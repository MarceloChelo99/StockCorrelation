"""Parse key 10-K sections from raw SEC submission text."""
from __future__ import annotations

from dataclasses import dataclass
import html
import re


_DOCUMENT_BLOCK_RE = re.compile(r"(?is)<DOCUMENT>(.*?)</DOCUMENT>")
_TYPE_RE = re.compile(r"(?im)<TYPE>\s*([^\n\r<]+)")
_TEXT_RE = re.compile(r"(?is)<TEXT>(.*?)</TEXT>")
_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)[^>]*>.*?</\1>")
_INLINE_HEADER_RE = re.compile(r"(?is)<ix:header[^>]*>.*?</ix:header>")
_COMMENT_RE = re.compile(r"(?is)<!--.*?-->")
_TAG_RE = re.compile(r"(?s)<[^>]+>")
_BLOCK_TAG_RE = re.compile(
    r"(?i)</?(address|article|aside|blockquote|br|caption|div|dl|dt|dd|figcaption|figure|footer|"
    r"h[1-6]|header|hr|li|main|nav|ol|p|section|table|tbody|td|tfoot|th|thead|title|tr|ul)[^>]*>"
)
_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTILINE_BREAK_RE = re.compile(r"\n{3,}")
_GENERIC_ITEM_RE = re.compile(r"(?im)^\s*ITEM\s+\d+\s*[A-Z]?\s*(?:\.|\||:|-|–|—)?\s*(?:[^\n]*)$")


def _loose_word(value: str) -> str:
    """Return a regex fragment that tolerates stray spaces inside heading words."""
    return r"\s*".join(re.escape(character) for character in value)


_ITEM_SEPARATOR_PATTERN = r"(?:\s*(?:\.|\||:|-|–|—)\s*){0,4}"
_BUSINESS_PATTERN = _loose_word("BUSINESS")
_RISK_PATTERN = _loose_word("RISK")
_FACTORS_PATTERN = _loose_word("FACTORS")
_UNRESOLVED_PATTERN = _loose_word("UNRESOLVED")
_STAFF_PATTERN = _loose_word("STAFF")
_COMMENTS_PATTERN = _loose_word("COMMENTS")
_CYBERSECURITY_PATTERN = rf"{_loose_word('CYBER')}\s*{_loose_word('SECURITY')}"
_PROPERTIES_PATTERN = _loose_word("PROPERTIES")
_LEGAL_PATTERN = _loose_word("LEGAL")
_PROCEEDINGS_PATTERN = _loose_word("PROCEEDINGS")
_MINE_PATTERN = _loose_word("MINE")
_SAFETY_PATTERN = _loose_word("SAFETY")
_MANAGEMENT_PATTERN = rf"{_loose_word('MANAGEMENT')}\s*(?:[’']\s*S)?"
_DISCUSSION_PATTERN = _loose_word("DISCUSSION")
_ANALYSIS_PATTERN = _loose_word("ANALYSIS")
_OPERATIONS_PATTERN = _loose_word("OPERATIONS")
_QUANTITATIVE_PATTERN = _loose_word("QUANTITATIVE")
_QUALITATIVE_PATTERN = _loose_word("QUALITATIVE")
_DISCLOSURES_PATTERN = _loose_word("DISCLOSURES")
_FINANCIAL_PATTERN = _loose_word("FINANCIAL")
_STATEMENTS_PATTERN = _loose_word("STATEMENTS")
_SUPPLEMENTARY_PATTERN = _loose_word("SUPPLEMENTARY")
_DATA_PATTERN = _loose_word("DATA")
_MDA_PATTERN = (
    rf"{_MANAGEMENT_PATTERN}\s+{_DISCUSSION_PATTERN}\s+AND\s+{_ANALYSIS_PATTERN}"
    rf"[\s\S]{{0,180}}?{_OPERATIONS_PATTERN}"
)
_MANAGEMENT_DISCUSSION_PATTERN = rf"{_MANAGEMENT_PATTERN}\s+{_DISCUSSION_PATTERN}"
_NOTES_PATTERN = _loose_word("NOTES")
_CONSOLIDATED_PATTERN = _loose_word("CONSOLIDATED")
_REPORT_PATTERN = _loose_word("REPORT")
_OF_PATTERN = _loose_word("OF")
_ENTRY_PATTERN = _loose_word("ENTRY")
_MATERIAL_PATTERN = _loose_word("MATERIAL")
_DEFINITIVE_PATTERN = _loose_word("DEFINITIVE")
_AGREEMENT_PATTERN = _loose_word("AGREEMENT")
_RESULTS_PATTERN = _loose_word("RESULTS")
_CONDITION_PATTERN = _loose_word("CONDITION")
_PROSPECTUS_PATTERN = _loose_word("PROSPECTUS")
_SUMMARY_PATTERN = _loose_word("SUMMARY")
_CORPORATE_PATTERN = _loose_word("CORPORATE")
_GOVERNANCE_PATTERN = _loose_word("GOVERNANCE")
_DIRECTOR_PATTERN = _loose_word("DIRECTOR")
_DIRECTORS_PATTERN = _loose_word("DIRECTORS")
_ELECTION_PATTERN = _loose_word("ELECTION")
_COMPENSATION_PATTERN = _loose_word("COMPENSATION")
_EXECUTIVE_PATTERN = _loose_word("EXECUTIVE")
_PAY_PATTERN = _loose_word("PAY")
_VERSUS_PATTERN = _loose_word("VERSUS")
_PERFORMANCE_PATTERN = _loose_word("PERFORMANCE")


@dataclass(frozen=True)
class SectionSpan:
    """Parsed text span for one filing section."""

    key: str
    title: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class ParsedTenK:
    """Parsed 10-K business, risk-factor, and MD&A sections."""

    document_text: str
    sections: dict[str, SectionSpan]

    def get(self, key: str) -> str | None:
        section = self.sections.get(key)
        return None if section is None else section.text


_HEADING_SPECS = {
    "business": {
        "title": "ITEM 1. BUSINESS",
        "patterns": [
            re.compile(rf"(?im)^\s*ITEM\s+1\s*{_ITEM_SEPARATOR_PATTERN}{_BUSINESS_PATTERN}\s*(?:\.|:)?\s*$"),
            re.compile(
                rf"(?im)^\s*(?:{_BUSINESS_PATTERN}|{_BUSINESS_PATTERN}\s+SUMMARY|OUR\s+{_BUSINESS_PATTERN}|ABOUT\s+.+)\s*(?:\.|:)?\s*$"
            ),
        ],
        "end_keys": ["risk_factors", "mda", "item_1b", "item_1c", "item_2"],
    },
    "risk_factors": {
        "title": "ITEM 1A. RISK FACTORS",
        "patterns": [
            re.compile(
                rf"(?im)^\s*ITEM\s+1\s*A\s*{_ITEM_SEPARATOR_PATTERN}{_RISK_PATTERN}\s+{_FACTORS_PATTERN}\s*(?:\.|:)?\s*$"
            ),
            re.compile(rf"(?im)^\s*{_RISK_PATTERN}\s+{_FACTORS_PATTERN}\s*(?:\.|:)?\s*$"),
        ],
        "end_keys": ["item_1b", "item_1c", "item_2", "mda"],
    },
    "mda": {
        "title": "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "patterns": [
            re.compile(
                rf"(?im)^\s*ITEM\s+7\s*{_ITEM_SEPARATOR_PATTERN}(?:MD&A|{_MDA_PATTERN}|{_MANAGEMENT_DISCUSSION_PATTERN})[^\n]*$"
            ),
            re.compile(rf"(?im)^\s*{_MDA_PATTERN}[^\n]*$"),
            re.compile(
                rf"(?im)^\s*{_MANAGEMENT_PATTERN}\s+{_DISCUSSION_PATTERN}\s+AND\s+{_ANALYSIS_PATTERN}[^\n]*$"
            ),
            re.compile(rf"(?im)^\s*{_MANAGEMENT_DISCUSSION_PATTERN}\s*(?:\.|:)?\s*$"),
        ],
        "end_keys": ["item_7a", "item_8"],
    },
    "item_1b": {
        "title": "ITEM 1B. UNRESOLVED STAFF COMMENTS",
        "patterns": [
            re.compile(
                rf"(?im)^\s*ITEM\s+1\s*B\s*{_ITEM_SEPARATOR_PATTERN}{_UNRESOLVED_PATTERN}\s+{_STAFF_PATTERN}\s+{_COMMENTS_PATTERN}\s*(?:\.|:)?\s*$"
            ),
            re.compile(rf"(?im)^\s*{_UNRESOLVED_PATTERN}\s+{_STAFF_PATTERN}\s+{_COMMENTS_PATTERN}\s*(?:\.|:)?\s*$"),
        ],
        "end_keys": [],
    },
    "item_1c": {
        "title": "ITEM 1C. CYBERSECURITY",
        "patterns": [
            re.compile(rf"(?im)^\s*ITEM\s+1\s*C\s*{_ITEM_SEPARATOR_PATTERN}{_CYBERSECURITY_PATTERN}\s*(?:\.|:)?\s*$"),
            re.compile(rf"(?im)^\s*{_CYBERSECURITY_PATTERN}\s*(?:\.|:)?\s*$"),
        ],
        "end_keys": ["item_2", "mda"],
    },
    "item_2": {
        "title": "ITEM 2. PROPERTIES",
        "patterns": [
            re.compile(rf"(?im)^\s*ITEM\s+2\s*{_ITEM_SEPARATOR_PATTERN}{_PROPERTIES_PATTERN}\s*(?:\.|:)?\s*$"),
            re.compile(rf"(?im)^\s*{_PROPERTIES_PATTERN}\s*(?:\.|:)?\s*$"),
        ],
        "end_keys": ["item_3", "item_4", "mda"],
    },
    "item_3": {
        "title": "ITEM 3. LEGAL PROCEEDINGS",
        "patterns": [
            re.compile(rf"(?im)^\s*ITEM\s+3\s*{_ITEM_SEPARATOR_PATTERN}{_LEGAL_PATTERN}\s+{_PROCEEDINGS_PATTERN}\s*(?:\.|:)?\s*$"),
            re.compile(rf"(?im)^\s*{_LEGAL_PATTERN}\s+{_PROCEEDINGS_PATTERN}\s*(?:\.|:)?\s*$"),
        ],
        "end_keys": ["item_4", "mda"],
    },
    "item_4": {
        "title": "ITEM 4. MINE SAFETY DISCLOSURES",
        "patterns": [
            re.compile(rf"(?im)^\s*ITEM\s+4\s*{_ITEM_SEPARATOR_PATTERN}{_MINE_PATTERN}\s+{_SAFETY_PATTERN}.*$"),
            re.compile(rf"(?im)^\s*{_MINE_PATTERN}\s+{_SAFETY_PATTERN}.*$"),
        ],
        "end_keys": ["mda"],
    },
    "item_7a": {
        "title": "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES",
        "patterns": [
            re.compile(
                rf"(?im)^\s*ITEM\s+7\s*A\s*{_ITEM_SEPARATOR_PATTERN}{_QUANTITATIVE_PATTERN}\s+AND\s+{_QUALITATIVE_PATTERN}\s+{_DISCLOSURES_PATTERN}.*$"
            ),
            re.compile(rf"(?im)^\s*{_QUANTITATIVE_PATTERN}\s+AND\s+{_QUALITATIVE_PATTERN}\s+{_DISCLOSURES_PATTERN}.*$"),
        ],
        "end_keys": ["item_8"],
    },
    "item_8": {
        "title": "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        "patterns": [
            re.compile(rf"(?im)^\s*ITEM\s+8\s*{_ITEM_SEPARATOR_PATTERN}{_FINANCIAL_PATTERN}\s+{_STATEMENTS_PATTERN}.*$"),
            re.compile(
                rf"(?im)^\s*{_FINANCIAL_PATTERN}\s+{_STATEMENTS_PATTERN}\s+AND\s+{_SUPPLEMENTARY_PATTERN}\s+{_DATA_PATTERN}\s*(?:\.|:)?\s*$"
            ),
            re.compile(
                rf"(?im)^\s*{_CONSOLIDATED_PATTERN}\s+{_FINANCIAL_PATTERN}\s+{_STATEMENTS_PATTERN}\s*(?:\.|:)?\s*$"
            ),
            re.compile(
                rf"(?im)^\s*{_NOTES_PATTERN}\s+TO\s+{_CONSOLIDATED_PATTERN}\s+{_FINANCIAL_PATTERN}\s+{_STATEMENTS_PATTERN}\s*(?:\.|:)?\s*$"
            ),
            re.compile(rf"(?im)^\s*{_REPORT_PATTERN}\s+{_OF_PATTERN}\s+{_MANAGEMENT_PATTERN}\s*(?:\.|:)?\s*$"),
        ],
        "end_keys": [],
    },
}


_OUTPUT_SECTION_HEADING_KEYS = {
    "business": "business",
    "risk_factors": "risk_factors",
    "mda": "mda",
    "item_1c_cybersecurity": "item_1c",
    "item_2_properties": "item_2",
    "item_3_legal_proceedings": "item_3",
    "item_7a_market_risk": "item_7a",
}
_OUTPUT_SECTION_KEYS = tuple(_OUTPUT_SECTION_HEADING_KEYS.keys())


def parse_10k_sections(submission_text: str) -> ParsedTenK:
    """Parse business, risk-factor, MD&A, and optional cybersecurity sections from one raw 10-K."""
    return _parse_configured_sections(
        submission_text,
        form_type="10-K",
        heading_specs=_HEADING_SPECS,
        output_heading_keys=_OUTPUT_SECTION_HEADING_KEYS,
        include_annual_exhibits=True,
    )


def parse_10q_sections(submission_text: str) -> ParsedTenK:
    """Parse useful 10-Q sections: MD&A, market risk, legal proceedings, and risk factors."""
    specs = {
        "q_legal": {
            "title": "PART II ITEM 1. LEGAL PROCEEDINGS",
            "patterns": [
                re.compile(rf"(?im)^\s*ITEM\s+1\s*{_ITEM_SEPARATOR_PATTERN}{_LEGAL_PATTERN}\s+{_PROCEEDINGS_PATTERN}\s*(?:\.|:)?\s*$"),
            ],
            "end_keys": ["q_risk", "q_mda", "q_market"],
        },
        "q_risk": {
            "title": "PART II ITEM 1A. RISK FACTORS",
            "patterns": [
                re.compile(rf"(?im)^\s*ITEM\s+1\s*A\s*{_ITEM_SEPARATOR_PATTERN}{_RISK_PATTERN}\s+{_FACTORS_PATTERN}\s*(?:\.|:)?\s*$"),
                re.compile(rf"(?im)^\s*{_RISK_PATTERN}\s+{_FACTORS_PATTERN}\s*(?:\.|:)?\s*$"),
            ],
            "end_keys": ["q_mda", "q_market"],
        },
        "q_mda": {
            "title": "PART I ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS",
            "patterns": [
                re.compile(rf"(?im)^\s*ITEM\s+2\s*{_ITEM_SEPARATOR_PATTERN}(?:MD&A|{_MDA_PATTERN}|{_MANAGEMENT_DISCUSSION_PATTERN})[^\n]*$"),
                re.compile(rf"(?im)^\s*{_MDA_PATTERN}[^\n]*$"),
            ],
            "end_keys": ["q_market", "q_legal", "q_risk"],
        },
        "q_market": {
            "title": "PART I ITEM 3. QUANTITATIVE AND QUALITATIVE DISCLOSURES",
            "patterns": [
                re.compile(rf"(?im)^\s*ITEM\s+3\s*{_ITEM_SEPARATOR_PATTERN}{_QUANTITATIVE_PATTERN}\s+AND\s+{_QUALITATIVE_PATTERN}\s+{_DISCLOSURES_PATTERN}.*$"),
            ],
            "end_keys": ["q_legal", "q_risk"],
        },
    }
    return _parse_configured_sections(
        submission_text,
        form_type="10-Q",
        heading_specs=specs,
        output_heading_keys={
            "q_mda": "q_mda",
            "q_market_risk": "q_market",
            "q_legal_proceedings": "q_legal",
            "q_risk_factors": "q_risk",
        },
        min_heading_score=-12,
    )


def parse_8k_sections(submission_text: str) -> ParsedTenK:
    """Parse useful 8-K event sections for agreements and earnings narratives."""
    specs = {
        "item_1_01": {
            "title": "ITEM 1.01 ENTRY INTO A MATERIAL DEFINITIVE AGREEMENT",
            "patterns": [
                re.compile(
                    rf"(?im)^\s*ITEM\s+1\.01\s*{_ITEM_SEPARATOR_PATTERN}{_ENTRY_PATTERN}.*{_MATERIAL_PATTERN}.*{_AGREEMENT_PATTERN}.*$"
                ),
                re.compile(r"(?im)^\s*ITEM\s+1\.01\b.*$"),
            ],
            "end_keys": ["item_2_02"],
        },
        "item_2_02": {
            "title": "ITEM 2.02 RESULTS OF OPERATIONS AND FINANCIAL CONDITION",
            "patterns": [
                re.compile(
                    rf"(?im)^\s*ITEM\s+2\.02\s*{_ITEM_SEPARATOR_PATTERN}{_RESULTS_PATTERN}.*{_OPERATIONS_PATTERN}.*{_CONDITION_PATTERN}.*$"
                ),
                re.compile(r"(?im)^\s*ITEM\s+2\.02\b.*$"),
            ],
            "end_keys": ["item_1_01"],
        },
    }
    return _parse_configured_sections(
        submission_text,
        form_type="8-K",
        heading_specs=specs,
        output_heading_keys={
            "eightk_item_1_01_material_agreement": "item_1_01",
            "eightk_item_2_02_results": "item_2_02",
        },
        generic_item_fallback=True,
    )


def parse_s1_sections(submission_text: str) -> ParsedTenK:
    """Parse useful S-1 prospectus sections for newer-company business models."""
    specs = {
        "summary": {
            "title": "PROSPECTUS SUMMARY",
            "patterns": [
                re.compile(rf"(?im)^\s*{_PROSPECTUS_PATTERN}\s+{_SUMMARY_PATTERN}\s*(?:\.|:)?\s*$"),
                re.compile(rf"(?im)^\s*{_SUMMARY_PATTERN}\s*(?:\.|:)?\s*$"),
            ],
            "end_keys": ["risk", "business", "mda"],
        },
        "risk": {
            "title": "RISK FACTORS",
            "patterns": [re.compile(rf"(?im)^\s*{_RISK_PATTERN}\s+{_FACTORS_PATTERN}\s*(?:\.|:)?\s*$")],
            "end_keys": ["business", "mda"],
        },
        "business": {
            "title": "BUSINESS",
            "patterns": [re.compile(rf"(?im)^\s*{_BUSINESS_PATTERN}\s*(?:\.|:)?\s*$")],
            "end_keys": ["mda"],
        },
        "mda": {
            "title": "MANAGEMENT'S DISCUSSION AND ANALYSIS",
            "patterns": [
                re.compile(rf"(?im)^\s*(?:{_MDA_PATTERN}|{_MANAGEMENT_DISCUSSION_PATTERN})[^\n]*$"),
            ],
            "end_keys": [],
        },
    }
    return _parse_configured_sections(
        submission_text,
        form_type="S-1",
        heading_specs=specs,
        output_heading_keys={
            "s1_summary": "summary",
            "s1_risk_factors": "risk",
            "s1_business": "business",
            "s1_mda": "mda",
        },
        max_section_chars=180000,
    )


def parse_proxy_sections(submission_text: str) -> ParsedTenK:
    """Parse useful DEF 14A proxy sections for governance and compensation signals."""
    specs = {
        "governance": {
            "title": "CORPORATE GOVERNANCE",
            "patterns": [
                re.compile(rf"(?im)^\s*{_CORPORATE_PATTERN}\s+{_GOVERNANCE_PATTERN}\s*(?:\.|:)?\s*$"),
                re.compile(rf"(?im)^\s*{_GOVERNANCE_PATTERN}\s*(?:\.|:)?\s*$"),
            ],
            "end_keys": ["directors", "compensation", "pay_performance"],
        },
        "directors": {
            "title": "ELECTION OF DIRECTORS",
            "patterns": [
                re.compile(rf"(?im)^\s*(?:{_ELECTION_PATTERN}\s+OF\s+{_DIRECTORS_PATTERN}|{_DIRECTOR_PATTERN}\s+NOMINEES)[^\n]*$"),
                re.compile(rf"(?im)^\s*PROPOSAL\s+NO\.\s*1\s*.*{_ELECTION_PATTERN}\s+OF\s+{_DIRECTORS_PATTERN}.*$"),
                re.compile(rf"(?im)^\s*{_DIRECTORS_PATTERN}\s*(?:\.|:)?\s*$"),
            ],
            "end_keys": ["governance", "compensation", "pay_performance"],
        },
        "compensation": {
            "title": "EXECUTIVE COMPENSATION",
            "patterns": [
                re.compile(rf"(?im)^\s*{_COMPENSATION_PATTERN}\s+{_DISCUSSION_PATTERN}\s+AND\s+{_ANALYSIS_PATTERN}\s*(?:\.|:)?\s*$"),
                re.compile(rf"(?im)^\s*{_EXECUTIVE_PATTERN}\s+{_COMPENSATION_PATTERN}\s*(?:\.|:)?\s*$"),
                re.compile(rf"(?im)^\s*{_COMPENSATION_PATTERN}\s*(?:\.|:)?\s*$"),
            ],
            "end_keys": ["pay_performance", "governance", "directors"],
        },
        "pay_performance": {
            "title": "PAY VERSUS PERFORMANCE",
            "patterns": [
                re.compile(rf"(?im)^\s*{_PAY_PATTERN}\s+{_VERSUS_PATTERN}\s+{_PERFORMANCE_PATTERN}\s*(?:\.|:)?\s*$"),
            ],
            "end_keys": ["governance", "directors", "compensation"],
        },
    }
    return _parse_configured_sections(
        submission_text,
        form_type="DEF 14A",
        heading_specs=specs,
        output_heading_keys={
            "proxy_governance": "governance",
            "proxy_directors": "directors",
            "proxy_compensation": "compensation",
            "proxy_pay_vs_performance": "pay_performance",
        },
        max_section_chars=140000,
    )


def parse_filing_sections(submission_text: str, form_type: str) -> ParsedTenK:
    """Dispatch raw filing text to the section parser for its SEC form type."""
    form = _normalize_form_type(form_type)
    if form.startswith("10-K"):
        return parse_10k_sections(submission_text)
    if form.startswith("10-Q"):
        return parse_10q_sections(submission_text)
    if form.startswith("8-K"):
        return parse_8k_sections(submission_text)
    if form.startswith("S-1"):
        return parse_s1_sections(submission_text)
    if form == "DEF14A":
        return parse_proxy_sections(submission_text)
    return ParsedTenK(document_text=normalize_document_text(extract_primary_document(submission_text, form_type=form_type)), sections={})


def _parse_configured_sections(
    submission_text: str,
    *,
    form_type: str,
    heading_specs: dict[str, dict[str, object]],
    output_heading_keys: dict[str, str],
    include_annual_exhibits: bool = False,
    generic_item_fallback: bool = False,
    max_section_chars: int | None = None,
    min_heading_score: int = -6,
) -> ParsedTenK:
    """Parse sections using caller-provided heading specs and output names."""
    if include_annual_exhibits:
        document = extract_parse_document(submission_text, form_type=form_type)
    else:
        document = extract_primary_document(submission_text, form_type=form_type)
    cleaned_text = normalize_document_text(document)

    starts = {
        key: _select_heading_start(cleaned_text, spec["patterns"], min_score=min_heading_score)
        for key, spec in heading_specs.items()
    }
    sections: dict[str, SectionSpan] = {}
    for key, heading_key in output_heading_keys.items():
        start = starts.get(heading_key)
        if start is None:
            continue

        end = len(cleaned_text)
        for end_key in heading_specs[heading_key]["end_keys"]:
            candidate = _select_boundary_start(
                cleaned_text,
                heading_specs[end_key]["patterns"],
                start,
                min_score=min_heading_score,
            )
            if candidate is not None:
                end = min(end, candidate)
        if generic_item_fallback or end == len(cleaned_text):
            candidate = _select_next_generic_item_start(cleaned_text, start)
            if candidate is not None:
                end = min(end, candidate)
        if max_section_chars is not None:
            end = min(end, start + max_section_chars)

        section_text = cleaned_text[start:end].strip()
        section_text = _cleanup_section_text(section_text)
        if not section_text:
            continue
        sections[key] = SectionSpan(
            key=key,
            title=str(heading_specs[heading_key]["title"]),
            start=start,
            end=end,
            text=section_text,
        )

    return ParsedTenK(document_text=cleaned_text, sections=sections)


def extract_primary_document(submission_text: str, *, form_type: str = "10-K") -> str:
    """Extract the main filing document body from a raw SEC submission text payload."""
    normalized_target = _normalize_form_type(form_type)
    best_fallback: str | None = None

    for block_match in _DOCUMENT_BLOCK_RE.finditer(submission_text):
        block = block_match.group(1)
        type_match = _TYPE_RE.search(block)
        if type_match is None:
            continue

        document_type = _normalize_form_type(type_match.group(1))
        text_match = _TEXT_RE.search(block)
        document_text = text_match.group(1) if text_match is not None else block
        if best_fallback is None:
            best_fallback = document_text

        if document_type == normalized_target or document_type.startswith(normalized_target):
            return document_text

    if best_fallback is not None:
        return best_fallback

    text_match = _TEXT_RE.search(submission_text)
    if text_match is not None:
        return text_match.group(1)
    return submission_text


def extract_parse_document(submission_text: str, *, form_type: str = "10-K") -> str:
    """Extract the filing document plus annual-report exhibits useful for section parsing."""
    primary = extract_primary_document(submission_text, form_type=form_type)
    annual_exhibits: list[str] = []

    for block_match in _DOCUMENT_BLOCK_RE.finditer(submission_text):
        block = block_match.group(1)
        type_match = _TYPE_RE.search(block)
        document_type = _normalize_form_type(type_match.group(1)) if type_match is not None else ""
        if document_type not in {"EX-13", "EX-99", "EX-99.1"}:
            continue
        text_match = _TEXT_RE.search(block)
        document_text = text_match.group(1) if text_match is not None else block
        annual_exhibits.append(document_text)

    if not annual_exhibits:
        return primary
    return "\n\n".join([primary, *annual_exhibits])


def normalize_document_text(document_text: str) -> str:
    """Convert filing HTML and inline XBRL into normalized plain text."""
    text = document_text
    text = _INLINE_HEADER_RE.sub(" ", text)
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _COMMENT_RE.sub(" ", text)
    text = re.sub(r"(?is)<\?.*?\?>", " ", text)
    text = _BLOCK_TAG_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = text.replace("\r", "\n")

    normalized_lines: list[str] = []
    for raw_line in text.splitlines():
        line = _WHITESPACE_RE.sub(" ", raw_line).strip()
        if line:
            normalized_lines.append(line)
        elif normalized_lines and normalized_lines[-1] != "":
            normalized_lines.append("")

    normalized = "\n".join(normalized_lines)
    normalized = _MULTILINE_BREAK_RE.sub("\n\n", normalized)
    return normalized.strip()


def _select_heading_start(text: str, patterns: list[re.Pattern[str]], *, min_score: int = -6) -> int | None:
    matches: list[re.Match[str]] = []
    for pattern in patterns:
        matches.extend(pattern.finditer(text))
    if not matches:
        return None
    matches = sorted(matches, key=lambda match: match.start())

    scored_matches: list[tuple[int, int]] = []
    for index, match in enumerate(matches):
        scored_matches.append((_score_heading_candidate(text, matches, index), match.start()))

    best_score, best_start = max(scored_matches, key=lambda item: (item[0], -item[1]))
    if best_score < min_score:
        return None
    return best_start


def _select_boundary_start(
    text: str,
    patterns: list[re.Pattern[str]],
    section_start: int,
    *,
    min_score: int = -6,
) -> int | None:
    """Return the first plausible boundary heading after a section start."""
    matches: list[re.Match[str]] = []
    for pattern in patterns:
        matches.extend(pattern.finditer(text))
    matches = sorted(matches, key=lambda match: match.start())

    for index, match in enumerate(matches):
        if match.start() <= section_start:
            continue
        if _score_heading_candidate(text, matches, index) >= min_score:
            return match.start()
    return None


def _select_next_generic_item_start(text: str, section_start: int) -> int | None:
    """Return the next generic Item heading after a section start."""
    for match in _GENERIC_ITEM_RE.finditer(text):
        if match.start() > section_start:
            return match.start()
    return None


def _score_heading_candidate(text: str, matches: list[re.Match[str]], index: int) -> int:
    match = matches[index]
    start = match.start()
    score = 0

    if index > 0 and start - matches[index - 1].start() <= 120:
        score += 6
    if index + 1 < len(matches) and matches[index + 1].start() - start <= 120:
        score += 6

    window_after = text[start : start + 200].lower()
    window_before = text[max(0, start - 200) : start].lower()
    if "table of contents" in window_after:
        score -= 5
    if "table of contents" in window_before:
        score -= 1
    if "part i" in window_before or "part ii" in window_before:
        score += 2
    if start < max(5000, len(text) // 100):
        score -= 1
    prose_window = text[match.end() : match.end() + 400]
    lowercase_count = sum(character.islower() for character in prose_window)
    if lowercase_count > 120:
        score += 6
    elif lowercase_count > 60:
        score += 3
    elif lowercase_count < 20:
        score -= 3
    score -= _index_like_penalty(text, match.end())

    current_item_code = _item_code(match.group(0))
    distance_to_next_item = _distance_to_next_item_heading(text, start, current_item_code)
    if distance_to_next_item is None:
        score += 8
    elif distance_to_next_item > 5000:
        score += 8
    elif distance_to_next_item > 1000:
        score += 5
    elif distance_to_next_item > 300:
        score += 2
    else:
        score -= 8
    return score


def _cleanup_section_text(section_text: str) -> str:
    text = section_text
    text = re.sub(
        r"(?im)^\s*TABLE OF CONTENTS\s*$",
        "",
        text,
    )
    text = re.sub(r"(?im)^\s*\d+\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_form_type(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()


def _index_like_penalty(text: str, start: int) -> int:
    lines = [
        line.strip()
        for line in text[start : start + 700].splitlines()
        if line.strip()
    ][:14]
    if not lines:
        return 0

    penalty = 0
    if _is_page_marker(lines[0]):
        penalty += 10

    page_markers = sum(_is_page_marker(line) for line in lines)
    section_titles = sum(_is_index_title(line) for line in lines)
    acronym_lines = sum(_is_acronym_line(line) for line in lines)
    short_lines = sum(len(line) <= 45 for line in lines)
    if page_markers >= 2 and section_titles >= 2:
        penalty += 8
    if short_lines >= 10 and section_titles >= 4:
        penalty += 6
    if short_lines >= 10 and acronym_lines >= 3:
        penalty += 8
    return penalty


def _is_page_marker(line: str) -> bool:
    return bool(re.fullmatch(r"(?:page|pages)?\s*(?:\d+|none|n/a|not applicable)(?:\s*[-,]\s*\d+)?", line, flags=re.I))


def _is_index_title(line: str) -> bool:
    return bool(
        re.search(
            r"business|risk factors|cybersecurity|properties|legal proceedings|management.s discussion|"
            r"management discussion|notes to consolidated|overview|basis & policies|year in review|"
            r"financial statements|market risk|controls and procedures|executive compensation",
            line,
            flags=re.I,
        )
    )


def _is_acronym_line(line: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9&/.-]{1,12}", line))


def _distance_to_next_item_heading(text: str, start: int, current_item_code: str | None) -> int | None:
    for next_match in _GENERIC_ITEM_RE.finditer(text, pos=start + 1):
        next_code = _item_code(next_match.group(0))
        if current_item_code is not None and next_code == current_item_code:
            continue
        return next_match.start() - start
    return None


def _item_code(value: str) -> str | None:
    match = re.search(r"ITEM\s+(\d+)\s*([A-Z]?)\s*(?:\.|\||:)?", value, flags=re.I)
    if match is None:
        return None
    return f"{match.group(1)}{match.group(2)}".upper()

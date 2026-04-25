"""Extract sparse company relationship edges from filing section text."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


RELATIONSHIP_TYPES = ["customer", "supplier", "competitor", "partner", "agreement", "generic"]

CUSTOMER_WORDS = [
    "customer",
    "customers",
    "client",
    "clients",
]
SUPPLIER_WORDS = [
    "supplier",
    "suppliers",
    "vendor",
    "vendors",
    "sourced",
    "purchase",
    "purchases",
    "purchased",
    "procure",
    "procures",
    "procured",
]
COMPETITOR_WORDS = [
    "compete",
    "competes",
    "competing",
    "competition",
    "competitive",
    "competitor",
    "competitors",
    "rival",
    "rivals",
]
PARTNER_WORDS = [
    "partner",
    "partners",
    "partnership",
    "collaboration",
    "collaborate",
    "alliance",
    "joint venture",
    "reseller",
    "distributor",
    "distribution agreement",
]
AGREEMENT_WORDS = [
    "agreement",
    "agreements",
    "contract",
    "contracts",
    "license",
    "licenses",
    "licensed",
    "arrangement",
    "arrangements",
]

GENERIC_ALIASES = {
    "apple",
    "target",
    "general",
    "public",
    "first",
    "brown",
    "international",
    "global",
    "american",
    "united",
    "new",
    "old",
    "principal",
    "progressive",
    "travelers",
    "regions",
    "citizens",
    "key",
    "ball",
    "pool",
    "crown",
    "steel",
    "huntington",
    "monster",
    "host",
}

COMPANY_SUFFIXES = [
    "incorporated",
    "inc",
    "corporation",
    "corp",
    "company",
    "co",
    "limited",
    "ltd",
    "plc",
    "holdings",
    "holding",
    "group",
    "class a",
    "class b",
    "class c",
    "the",
]


@dataclass(frozen=True)
class CompanyAlias:
    """Searchable name variant for one public company."""

    ticker: str
    company_name: str
    alias: str
    pattern: re.Pattern[str]


def extract_relationships(
    sections: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    max_section_chars: int = 60000,
    min_confidence: float = 0.45,
) -> pd.DataFrame:
    """Extract relationship edges from parsed 10-K sections.

    Input sections must contain ticker, accession_no, filing_date, business_text,
    risk_factors_text, and mda_text. Output is one row per source-target filing
    relationship mention with a context snippet and confidence score.
    """
    aliases = build_company_aliases(metadata)
    rows: list[dict[str, object]] = []
    for _, filing in sections.iterrows():
        source_ticker = str(filing["ticker"]).upper()
        text_parts = [
            str(filing.get("business_text", ""))[:max_section_chars],
            str(filing.get("risk_factors_text", ""))[:max_section_chars],
            str(filing.get("mda_text", ""))[:max_section_chars],
        ]
        text = "\n".join(part for part in text_parts if part and part != "nan")
        if not text.strip():
            continue

        for match in find_company_mentions(text, aliases, source_ticker):
            relationship_type, confidence = classify_relationship(match["context"])
            if confidence < min_confidence:
                continue
            rows.append(
                {
                    "source_ticker": source_ticker,
                    "target_ticker": match["target_ticker"],
                    "relationship_type": relationship_type,
                    "filing_date": pd.Timestamp(filing["filing_date"]).strftime("%Y-%m-%d"),
                    "form": "10-K",
                    "accession_no": filing["accession_no"],
                    "confidence": confidence,
                    "matched_alias": match["alias"],
                    "context_snippet": compact_snippet(match["context"]),
                }
            )

    if not rows:
        return empty_relationship_frame()

    relationships = pd.DataFrame(rows)
    relationships = relationships.sort_values(
        ["source_ticker", "target_ticker", "relationship_type", "confidence"],
        ascending=[True, True, True, False],
    )
    relationships = relationships.drop_duplicates(
        subset=["source_ticker", "target_ticker", "relationship_type", "accession_no"],
        keep="first",
    )
    return relationships.reset_index(drop=True)


def build_company_aliases(metadata: pd.DataFrame) -> list[CompanyAlias]:
    """Build conservative company-name aliases from ticker metadata."""
    required = {"ticker", "company_name"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    aliases: list[CompanyAlias] = []
    seen: set[tuple[str, str]] = set()
    for _, row in metadata.iterrows():
        ticker = str(row["ticker"]).upper()
        company_name = str(row["company_name"])
        for alias in candidate_aliases(company_name):
            key = (ticker, alias.lower())
            if key in seen:
                continue
            seen.add(key)
            pattern = re.compile(rf"(?<![A-Za-z0-9&]){re.escape(alias)}(?![A-Za-z0-9&])", re.IGNORECASE)
            aliases.append(CompanyAlias(ticker=ticker, company_name=company_name, alias=alias, pattern=pattern))
    aliases.sort(key=lambda item: len(item.alias), reverse=True)
    return aliases


def candidate_aliases(company_name: str) -> list[str]:
    """Return high-precision aliases for one company name."""
    cleaned = normalize_company_name(company_name)
    aliases = {cleaned}

    without_parentheses = re.sub(r"\([^)]*\)", " ", company_name)
    aliases.add(normalize_company_name(without_parentheses))

    suffixless = strip_company_suffixes(cleaned)
    aliases.add(suffixless)

    if "," in company_name:
        aliases.add(normalize_company_name(company_name.split(",", 1)[0]))

    result = []
    for alias in aliases:
        alias = " ".join(alias.split())
        if is_usable_alias(alias):
            result.append(alias)
    return sorted(set(result), key=len, reverse=True)


def normalize_company_name(value: str) -> str:
    """Normalize punctuation while preserving readable company words."""
    value = value.replace("&", " and ")
    value = value.replace("’", "'")
    value = re.sub(r"[^A-Za-z0-9 .'-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .-'")


def strip_company_suffixes(value: str) -> str:
    """Remove common legal suffixes from the end of a company name."""
    words = value.split()
    while words and words[-1].lower().strip(".") in COMPANY_SUFFIXES:
        words = words[:-1]
    return " ".join(words)


def is_usable_alias(alias: str) -> bool:
    """Keep aliases that are unlikely to create obvious false positives."""
    if len(alias) < 5:
        return False
    words = alias.lower().replace(".", "").split()
    if not words:
        return False
    if len(words) == 1:
        return len(words[0]) >= 6 and words[0] not in GENERIC_ALIASES
    if len(alias) < 8:
        return False
    return not all(word in COMPANY_SUFFIXES for word in words)


def find_company_mentions(text: str, aliases: list[CompanyAlias], source_ticker: str) -> list[dict[str, str]]:
    """Find target-company mentions in filing text."""
    mentions = []
    matched_spans: list[tuple[int, int]] = []
    lowered_text = text.lower()
    for alias in aliases:
        if alias.ticker == source_ticker:
            continue
        if alias.alias.lower() not in lowered_text:
            continue
        for match in alias.pattern.finditer(text):
            span = match.span()
            if overlaps_existing_span(span, matched_spans):
                continue
            matched_spans.append(span)
            mentions.append(
                {
                    "target_ticker": alias.ticker,
                    "alias": alias.alias,
                    "context": context_window(text, span[0], span[1]),
                }
            )
    return mentions


def overlaps_existing_span(span: tuple[int, int], existing: list[tuple[int, int]]) -> bool:
    """Return true if a new alias span overlaps a previously accepted longer alias."""
    start, end = span
    for old_start, old_end in existing:
        if start < old_end and end > old_start:
            return True
    return False


def context_window(text: str, start: int, end: int, *, radius: int = 280) -> str:
    """Return local context around a matched company mention."""
    sentence_left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start))
    sentence_right_candidates = [index for index in [text.find(".", end), text.find("\n", end)] if index != -1]
    if sentence_right_candidates:
        left = 0 if sentence_left == -1 else sentence_left + 1
        right = min(sentence_right_candidates) + 1
        return text[left:right]

    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right]


def classify_relationship(context: str) -> tuple[str, float]:
    """Classify a mention context into a coarse relationship type."""
    lowered = context.lower()
    scores = {
        "customer": keyword_score(lowered, CUSTOMER_WORDS),
        "supplier": keyword_score(lowered, SUPPLIER_WORDS),
        "competitor": keyword_score(lowered, COMPETITOR_WORDS),
        "partner": keyword_score(lowered, PARTNER_WORDS),
        "agreement": keyword_score(lowered, AGREEMENT_WORDS),
    }

    # Competition lists often also contain generic customer/supplier words.
    # Prefer competitor unless another relationship has much stronger evidence.
    if scores["competitor"] > 0:
        strongest_non_competitor = max(score for key, score in scores.items() if key != "competitor")
        if scores["competitor"] >= strongest_non_competitor:
            relationship_type = "competitor"
        else:
            relationship_type = max(scores, key=scores.get)
    else:
        relationship_type = max(scores, key=scores.get)
    score = scores[relationship_type]
    if score == 0:
        return "generic", 0.35
    confidence = min(0.95, 0.45 + 0.12 * score)
    return relationship_type, confidence


def keyword_score(text: str, keywords: list[str]) -> int:
    """Count relationship keywords in a local context window."""
    score = 0
    for keyword in keywords:
        score += len(re.findall(rf"\b{re.escape(keyword)}\b", text))
    return score


def compact_snippet(value: str, *, max_chars: int = 600) -> str:
    """Collapse whitespace and trim a context snippet."""
    snippet = re.sub(r"\s+", " ", value).strip()
    if len(snippet) <= max_chars:
        return snippet
    return snippet[: max_chars - 3].rstrip() + "..."


def empty_relationship_frame() -> pd.DataFrame:
    """Return an empty relationship frame with the stable output schema."""
    return pd.DataFrame(
        columns=[
            "source_ticker",
            "target_ticker",
            "relationship_type",
            "filing_date",
            "form",
            "accession_no",
            "confidence",
            "matched_alias",
            "context_snippet",
        ]
    )

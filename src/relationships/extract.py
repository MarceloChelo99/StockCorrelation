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
    "procure",
    "procures",
    "procured",
    "supplied",
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
    "booking",
    "southern",
    "trimble",
    "old dominion",
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


@dataclass(frozen=True)
class RelationshipClassification:
    """Coarse relationship label plus explicit supplier/customer direction."""

    relationship_type: str
    confidence: float
    source_role: str
    target_role: str
    supply_chain_direction: str
    direction_confidence: float


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
            classification = classify_relationship_detail(match["context"], matched_alias=match["alias"])
            if classification.confidence < min_confidence:
                continue
            direction = directed_supply_chain_fields(
                source_ticker=source_ticker,
                target_ticker=match["target_ticker"],
                classification=classification,
            )
            rows.append(
                {
                    "source_ticker": source_ticker,
                    "target_ticker": match["target_ticker"],
                    "relationship_type": classification.relationship_type,
                    "filing_date": pd.Timestamp(filing["filing_date"]).strftime("%Y-%m-%d"),
                    "form": "10-K",
                    "accession_no": filing["accession_no"],
                    "confidence": classification.confidence,
                    "matched_alias": match["alias"],
                    **direction,
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
    lower_alias = alias.lower().replace(".", "")
    if lower_alias in GENERIC_ALIASES:
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
            if not is_probable_company_mention(alias, match.group(0)):
                continue
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


def is_probable_company_mention(alias: CompanyAlias, matched_text: str) -> bool:
    """Reject obvious common-phrase matches while preserving named companies.

    We prefer precision over recall for graph edges. A lowercase phrase like
    "applied materials" is usually descriptive text, not a public-company name.
    """
    alias_words = alias.alias.lower().replace(".", "").split()
    has_legal_suffix = any(word in COMPANY_SUFFIXES for word in alias_words)
    if has_legal_suffix:
        return True
    if matched_text.islower():
        return False
    return True


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
    classification = classify_relationship_detail(context)
    return classification.relationship_type, classification.confidence


def classify_relationship_detail(context: str, matched_alias: str | None = None) -> RelationshipClassification:
    """Classify context and attach source/target role semantics."""
    lowered = context.lower()
    if is_non_operating_context(lowered):
        return RelationshipClassification(
            relationship_type="generic",
            confidence=0.35,
            source_role="",
            target_role="",
            supply_chain_direction="",
            direction_confidence=0.0,
        )

    scores = {
        "customer": keyword_score(lowered, CUSTOMER_WORDS),
        "supplier": supplier_context_score(lowered),
        "competitor": keyword_score(lowered, COMPETITOR_WORDS),
        "partner": keyword_score(lowered, PARTNER_WORDS),
        "agreement": keyword_score(lowered, AGREEMENT_WORDS),
    }
    has_competitor_context = has_competitor_list_pattern(lowered)
    if has_competitor_context:
        scores["competitor"] += 4
    if matched_alias and target_appears_as_buyer(lowered, matched_alias):
        scores["customer"] += 2
    if matched_alias and target_appears_as_supplier(lowered, matched_alias):
        scores["supplier"] += 2

    # Competition lists often also contain generic customer/supplier words.
    # Prefer competitor unless another relationship has much stronger evidence.
    if scores["competitor"] > 0:
        strongest_non_competitor = max(score for key, score in scores.items() if key != "competitor")
        if has_competitor_context or scores["competitor"] >= strongest_non_competitor:
            relationship_type = "competitor"
        else:
            relationship_type = max(scores, key=scores.get)
    else:
        relationship_type = max(scores, key=scores.get)
    score = scores[relationship_type]
    if score == 0:
        return RelationshipClassification(
            relationship_type="generic",
            confidence=0.35,
            source_role="",
            target_role="",
            supply_chain_direction="",
            direction_confidence=0.0,
        )
    confidence = min(0.95, 0.45 + 0.12 * score)
    source_role = ""
    target_role = ""
    supply_chain_direction = ""
    direction_confidence = 0.0
    if relationship_type == "customer":
        source_role = "supplier"
        target_role = "customer"
        supply_chain_direction = "source_supplies_target"
        direction_confidence = 0.75 if has_customer_list_pattern(lowered, matched_alias) else 0.60
    elif relationship_type == "supplier":
        source_role = "customer"
        target_role = "supplier"
        supply_chain_direction = "target_supplies_source"
        direction_confidence = 0.75 if has_supplier_purchase_pattern(lowered) else 0.60
    elif relationship_type == "competitor":
        source_role = "competitor"
        target_role = "competitor"
    elif relationship_type in {"partner", "agreement"}:
        source_role = "counterparty"
        target_role = "counterparty"

    return RelationshipClassification(
        relationship_type=relationship_type,
        confidence=confidence,
        source_role=source_role,
        target_role=target_role,
        supply_chain_direction=supply_chain_direction,
        direction_confidence=direction_confidence,
    )


def supplier_context_score(text: str) -> int:
    """Score supplier evidence while avoiding generic purchase-language traps."""
    score = keyword_score(text, SUPPLIER_WORDS)
    if has_supplier_purchase_pattern(text):
        score += 2
    return score


def has_competitor_list_pattern(text: str) -> bool:
    """Return true when vendor-like words appear inside competitor lists."""
    patterns = [
        r"\b(?:compete|competes|competing|competition|competitors?)\b.{0,140}\b(?:including|include|includes|with|from)\b",
        r"\bcompetitors?\b.{0,120}\b(?:are|were|such\s+as|like)\b",
        r"\bcompetition\b.{0,120}\b(?:coming\s+from|from|with)\b.{0,120}\b(?:vendors?|providers?|companies|solutions?)\b",
        r"\bvendors?\b.{0,160}\bcompetitive\s+solutions?\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def target_appears_as_buyer(text: str, matched_alias: str) -> bool:
    """Return true when the matched company is described as buying from the filer."""
    alias = re.escape(matched_alias.lower())
    if re.search(rf"\b{alias}\b.{{0,80}}\bpurchase\s+price\b", text):
        return False
    patterns = [
        rf"\b{alias}\b.{{0,80}}\b(?:will\s+)?(?:purchase|purchases|purchased|buy|buys|bought|procure|procures|procured)\b",
        rf"\bwe\b.{{0,40}}\b(?:supply|supplied|provide|provided|sell|sold|deliver|delivered|license|licensed)\b.{{0,140}}\b(?:to|for)\b.{{0,80}}\b{alias}\b",
        rf"\b(?:sold|sales|revenue|revenues)\s+(?:to|with)\b.{{0,80}}\b{alias}\b",
        rf"\b(?:customers?|clients?)\b.{{0,80}}\b(?:include|includes|included|are|were|was|is)\b.{{0,100}}\b{alias}\b",
        rf"\b{alias}\b.{{0,100}}\b(?:accounted\s+for|represented|represents|comprised|comprises)\b.{{0,100}}\b(?:sales|revenue|revenues)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def target_appears_as_supplier(text: str, matched_alias: str) -> bool:
    """Return true when the matched company is described as supplying the filer."""
    alias = re.escape(matched_alias.lower())
    patterns = [
        rf"\b{alias}\b.{{0,80}}\b(?:exclusive\s+)?(?:supplier|suppliers|vendor|vendors)\b",
        rf"\b(?:supplier|suppliers|vendor|vendors)\b.{{0,80}}\b{alias}\b",
        rf"\b(?:supplied|provided|manufactured)\s+by\b.{{0,80}}\b{alias}\b",
        rf"\b(?:purchase|purchases|purchased|procure|procures|procured|source|sourced)\b.{{0,80}}\bfrom\b.{{0,80}}\b{alias}\b",
        rf"\b(?:use|uses|using|utilize|utilizes|utilizing|rely|relies|relying|depend|depends|dependent|host|hosts|hosted|hosting|run|runs|running|operate|operates|operating)\b.{{0,180}}\b{alias}\b",
        rf"\bour\s+(?:third-party\s+)?(?:provider|providers|service\s+provider|service\s+providers)\b.{{0,100}}\b{alias}\b",
        rf"\b(?:cloud|infrastructure|ai|technology|payment|data\s+center)\s+(?:providers?|platforms?|services?)\b.{{0,120}}\b(?:including|include|includes|such\s+as|like|primarily)\b.{{0,100}}\b{alias}\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def has_customer_list_pattern(text: str, matched_alias: str | None = None) -> bool:
    """Return true when context says the matched firm buys from the filer."""
    patterns = [
        r"\b(?:our|the company's|company's)?\s*(?:customers?|clients?)\s+(?:include|includes|included|are|were)\b",
        r"\b(?:sales|revenue|revenues)\s+(?:to|from)\b",
        r"\baccounts?\s+receivable\s+from\b",
    ]
    if matched_alias:
        alias = re.escape(matched_alias.lower())
        patterns.append(
            rf"\bwe\b.{{0,40}}\b(?:supply|supplied|provide|provided|sell|sold|deliver|delivered|license|licensed)\b.{{0,140}}\b(?:to|for)\b.{{0,80}}\b{alias}\b"
        )
    return any(re.search(pattern, text) for pattern in patterns)


def has_supplier_purchase_pattern(text: str) -> bool:
    """Return true when context says the matched firm supplies the filer."""
    patterns = [
        r"\b(?:purchase|purchases|purchased|procure|procures|procured|source|sourced)\b.{0,80}\bfrom\b",
        r"\b(?:supplied|provided|manufactured)\s+by\b",
        r"\b(?:depend|depends|dependent|rely|relies|reliant)\b.{0,80}\b(?:supplier|suppliers|vendor|vendors)\b",
        r"\b(?:supplier|suppliers|vendor|vendors)\s+(?:include|includes|included|are|were)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def is_non_operating_context(text: str) -> bool:
    """Return true for biographical mentions that are not company relationships."""
    patterns = [
        r"\b(?:served|serves|previously\s+served|currently\s+serves)\s+as\b.{0,160}\b(?:chief|executive|president|chairman|director|board)\b",
        r"\b(?:board\s+of\s+directors|executive\s+officer|chairman\s+of\s+the\s+board)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def directed_supply_chain_fields(
    *,
    source_ticker: str,
    target_ticker: str,
    classification: RelationshipClassification,
) -> dict[str, object]:
    """Return explicit supplier/customer columns for a classified edge."""
    supplier_ticker = ""
    customer_ticker = ""
    if classification.supply_chain_direction == "source_supplies_target":
        supplier_ticker = source_ticker
        customer_ticker = target_ticker
    elif classification.supply_chain_direction == "target_supplies_source":
        supplier_ticker = target_ticker
        customer_ticker = source_ticker

    return {
        "source_role": classification.source_role,
        "target_role": classification.target_role,
        "supplier_ticker": supplier_ticker,
        "customer_ticker": customer_ticker,
        "supply_chain_direction": classification.supply_chain_direction,
        "direction_confidence": classification.direction_confidence,
    }


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
            "source_role",
            "target_role",
            "supplier_ticker",
            "customer_ticker",
            "supply_chain_direction",
            "direction_confidence",
            "context_snippet",
        ]
    )

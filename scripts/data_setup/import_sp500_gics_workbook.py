"""Import a curated S&P 500 historical-member GICS workbook.

The workbook is not full point-in-time GICS history. It gives one cleaned GICS
classification per ticker for current and former S&P 500 members from 2010
onward. That is still useful because it fills the sector-label gap for deleted
constituents in historical backtests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import re
from pathlib import Path
import sys
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import _bootstrap  # noqa: F401
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log


REPO_ROOT = _bootstrap.REPO_ROOT
DEFAULT_INPUT = Path.home() / "Downloads" / "sp500_members_gics_2010_present.xlsx"
DEFAULT_GICS_OUTPUT = REPO_ROOT / "data" / "processed" / "metadata" / "sp500_members_gics_2010_present.parquet"
DEFAULT_MEMBERSHIP_PATH = REPO_ROOT / "data" / "processed" / "metadata" / "sp500_membership_history.parquet"

SHEET_NAME = "All Members 2010+"
EXCEL_EPOCH = datetime(1899, 12, 30)
TEXT_COLUMNS = {
    "ticker",
    "company_name",
    "status",
    "gics_sector",
    "gics_industry_group",
    "gics_industry",
    "gics_sub_industry",
    "source",
    "notes",
}
DATE_COLUMNS = {"first_in_sp500", "last_out"}
INTEGER_COLUMNS = {
    "stints",
    "gics_sector_code",
    "gics_industry_group_code",
    "gics_industry_code",
    "gics_sub_industry_code",
}
REQUIRED_COLUMNS = {
    "ticker",
    "company_name",
    "status",
    "first_in_sp500",
    "gics_sector",
    "gics_sub_industry",
}

# These map workbook post-rename/bankruptcy tickers onto the project ticker used
# by the membership and price artifacts. Keep this list deliberately small and
# auditable; exact ticker matches still cover most rows.
WORKBOOK_TO_PROJECT_TICKER = {
    "AABA": "YHOO",
    "ANRZQ": "ANR",
    "ANTM": "WLP",
    "BBT": "BBT",
    "BHGE": "BHI",
    "BTUUQ": "BTU",
    "EKDKQ": "EK",
    "RSHCQ": "RSH",
    "WYND": "WYN",
}


def parse_args() -> argparse.Namespace:
    """Parse workbook import arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to sp500_members_gics_2010_present.xlsx.")
    parser.add_argument("--gics-output", default=str(DEFAULT_GICS_OUTPUT), help="Clean GICS parquet output path.")
    parser.add_argument(
        "--membership-path",
        default=str(DEFAULT_MEMBERSHIP_PATH),
        help="Historical membership parquet to enrich in place.",
    )
    parser.add_argument(
        "--no-update-membership",
        action="store_true",
        help="Only write the standalone GICS parquet; do not update membership history.",
    )
    return parser.parse_args()


def normalize_ticker(value: object) -> str | None:
    """Normalize spreadsheet tickers to the project convention."""
    if value is None or pd.isna(value):
        return None
    ticker = str(value).strip().upper().replace(".", "-")
    if not ticker or ticker == "NAN":
        return None
    return ticker


def project_ticker(value: object) -> str | None:
    """Return the project ticker, applying known rename/bankruptcy aliases."""
    ticker = normalize_ticker(value)
    if not ticker:
        return None
    return WORKBOOK_TO_PROJECT_TICKER.get(ticker, ticker)


def canonical_column_name(value: object) -> str:
    """Return snake_case names for workbook columns."""
    text = str(value).strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    aliases = {
        "company": "company_name",
        "first_in_sandp_500": "first_in_sp500",
        "last_out": "last_out",
        "blank_current": "last_out",
        "last_out_blank_current": "last_out",
        "sector_code": "gics_sector_code",
        "gics_sub_industry_code_8_digit": "gics_sub_industry_code",
    }
    return aliases.get(text, text)


def load_shared_strings(workbook: ZipFile) -> list[str]:
    """Return workbook shared strings."""
    path = "xl/sharedStrings.xml"
    if path not in workbook.namelist():
        return []
    root = ET.fromstring(workbook.read(path))
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings = []
    for item in root.findall("m:si", namespace):
        parts = [node.text or "" for node in item.findall(".//m:t", namespace)]
        strings.append("".join(parts))
    return strings


def column_index(cell_reference: str) -> int:
    """Return zero-based column index from an Excel cell reference."""
    letters = "".join(character for character in cell_reference if character.isalpha())
    index = 0
    for character in letters:
        index = index * 26 + (ord(character.upper()) - ord("A") + 1)
    return index - 1


def cell_value(cell: ET.Element, shared_strings: list[str], namespace: dict[str, str]) -> object:
    """Return a scalar value from one XLSX cell."""
    cell_type = cell.attrib.get("t")
    if cell_type == "s":
        raw = cell.findtext("m:v", namespaces=namespace)
        if raw is None:
            return None
        return shared_strings[int(raw)]
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//m:t", namespace))
    raw = cell.findtext("m:v", namespaces=namespace)
    if raw is None:
        return None
    return raw


def workbook_sheet_path(workbook: ZipFile, sheet_name: str) -> str:
    """Resolve a worksheet XML path by visible sheet name."""
    main_ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
    relationship_id = None
    for sheet in workbook_root.findall(".//m:sheet", main_ns):
        if sheet.attrib.get("name") == sheet_name:
            relationship_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            break
    if relationship_id is None:
        raise ValueError(f"Workbook does not contain sheet {sheet_name!r}.")

    rel_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    for relationship in rel_root.findall("r:Relationship", rel_ns):
        if relationship.attrib.get("Id") == relationship_id:
            target = relationship.attrib["Target"].lstrip("/")
            return target if target.startswith("xl/") else f"xl/{target}"
    raise ValueError(f"Could not resolve worksheet relationship for {sheet_name!r}.")


def read_xlsx_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read one simple worksheet using only the Python standard library."""
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as workbook:
        shared_strings = load_shared_strings(workbook)
        sheet_path = workbook_sheet_path(workbook, sheet_name)
        root = ET.fromstring(workbook.read(sheet_path))

    rows = []
    max_column = 0
    for row in root.findall(".//m:sheetData/m:row", namespace):
        values: dict[int, object] = {}
        for cell in row.findall("m:c", namespace):
            reference = cell.attrib.get("r", "")
            if not reference:
                continue
            index = column_index(reference)
            values[index] = cell_value(cell, shared_strings, namespace)
            max_column = max(max_column, index)
        if values:
            rows.append([values.get(index) for index in range(max_column + 1)])
    return pd.DataFrame(rows)


def excel_date(value: object) -> pd.Timestamp | pd.NaT:
    """Parse either an ISO date string or an Excel serial date."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return pd.NaT
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return pd.to_datetime(text, errors="coerce")
    return pd.Timestamp(EXCEL_EPOCH + timedelta(days=number))


def integer_value(value: object) -> pd.NA | int:
    """Parse nullable integer-like workbook values."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return pd.NA
    return int(float(str(value).strip()))


def load_gics_workbook(path: Path) -> pd.DataFrame:
    """Return normalized workbook rows with one classification per ticker."""
    raw = read_xlsx_sheet(path, SHEET_NAME)
    header_matches = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Ticker")]
    if len(header_matches) == 0:
        raise ValueError("Could not find the Ticker header row in the workbook.")
    header_row = int(header_matches[0])
    headers = [canonical_column_name(value) for value in raw.iloc[header_row].tolist()]
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = headers
    data = data.loc[:, [column for column in data.columns if column]]
    data = data.dropna(subset=["ticker"], how="all").copy()

    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"GICS workbook is missing required columns: {sorted(missing)}")

    for column in TEXT_COLUMNS.intersection(data.columns):
        data[column] = data[column].where(data[column].notna(), pd.NA)
        data[column] = data[column].astype("string").str.strip()
        data.loc[data[column].isin(["", "nan", "NaN"]), column] = pd.NA
    for column in DATE_COLUMNS.intersection(data.columns):
        data[column] = data[column].map(excel_date)
    for column in INTEGER_COLUMNS.intersection(data.columns):
        data[column] = data[column].map(integer_value).astype("Int64")

    data["ticker_raw"] = data["ticker"].map(normalize_ticker)
    data["ticker"] = data["ticker"].map(project_ticker)
    data["gics_source"] = data.get("source", pd.Series(pd.NA, index=data.index))
    data["gics_notes"] = data.get("notes", pd.Series(pd.NA, index=data.index))
    data["gics_asof_note"] = "Workbook uses March 17, 2023 GICS taxonomy / closest equivalent."
    data = data.drop_duplicates("ticker", keep="last").sort_values("ticker").reset_index(drop=True)
    return data


def enrich_membership(membership: pd.DataFrame, gics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fill missing membership GICS fields from the imported workbook."""
    result = membership.copy()
    result["ticker"] = result["ticker"].astype(str).str.upper()
    before_sector_tickers = int(result.dropna(subset=["gics_sector"])["ticker"].nunique())

    columns = [
        "ticker",
        "gics_sector",
        "gics_sub_industry",
        "gics_industry_group",
        "gics_industry",
        "gics_sector_code",
        "gics_industry_group_code",
        "gics_industry_code",
        "gics_sub_industry_code",
        "gics_source",
        "gics_notes",
        "gics_asof_note",
    ]
    lookup = gics.loc[:, [column for column in columns if column in gics.columns]].copy()
    merged = result.merge(lookup, on="ticker", how="left", suffixes=("", "_workbook"))

    fill_columns = [
        "gics_sector",
        "gics_sub_industry",
        "gics_industry_group",
        "gics_industry",
        "gics_sector_code",
        "gics_industry_group_code",
        "gics_industry_code",
        "gics_sub_industry_code",
        "gics_source",
        "gics_notes",
        "gics_asof_note",
    ]
    for column in fill_columns:
        workbook_column = f"{column}_workbook"
        if workbook_column not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = pd.NA
        merged[column] = merged[column].where(merged[column].notna(), merged[workbook_column])

    drop_columns = [column for column in merged.columns if column.endswith("_workbook")]
    merged = merged.drop(columns=drop_columns)
    after_sector_tickers = int(merged.dropna(subset=["gics_sector"])["ticker"].nunique())
    stats = {
        "membership_tickers_with_gics_before": before_sector_tickers,
        "membership_tickers_with_gics_after": after_sector_tickers,
        "membership_tickers_gics_added": after_sector_tickers - before_sector_tickers,
        "gics_workbook_tickers": int(gics["ticker"].nunique()),
        "matched_workbook_tickers": int(gics[gics["ticker"].isin(set(result["ticker"]))]["ticker"].nunique()),
    }
    return merged, stats


def main() -> None:
    """Import workbook, write clean GICS metadata, and optionally enrich membership."""
    args = parse_args()
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"GICS workbook not found: {input_path}")

    gics = load_gics_workbook(input_path)
    gics_output = Path(args.gics_output)
    ensure_dir(gics_output.parent)
    gics.to_parquet(gics_output, index=False)

    stats = {
        "input_path": str(input_path),
        "gics_output": str(gics_output),
        "rows": int(len(gics)),
        "unique_tickers": int(gics["ticker"].nunique()),
        "current_members": int(gics["status"].eq("Current").sum()) if "status" in gics.columns else None,
        "former_members": int(gics["status"].eq("Former").sum()) if "status" in gics.columns else None,
        "source_imported_at": datetime.now(timezone.utc).isoformat(),
        "point_in_time_note": (
            "This is not full historical GICS. It is one cleaned classification per 2010-present S&P member "
            "using the March 17, 2023 taxonomy or closest equivalent."
        ),
    }

    if not args.no_update_membership:
        membership_path = Path(args.membership_path)
        if not membership_path.exists():
            raise FileNotFoundError(f"Membership parquet not found: {membership_path}")
        membership = pd.read_parquet(membership_path)
        enriched, enrich_stats = enrich_membership(membership, gics)
        enriched.to_parquet(membership_path, index=False)
        stats.update(enrich_stats)
        stats["membership_path"] = str(membership_path)
        log(
            "Enriched membership GICS coverage from "
            f"{enrich_stats['membership_tickers_with_gics_before']} to "
            f"{enrich_stats['membership_tickers_with_gics_after']} tickers.",
            tag="gics-import",
        )

    write_json(stats, gics_output.with_suffix(".manifest.json"))
    log(f"Wrote {len(gics):,} GICS rows to {gics_output}.", tag="gics-import")
    log(json.dumps(stats, indent=2), tag="gics-import")


if __name__ == "__main__":
    main()

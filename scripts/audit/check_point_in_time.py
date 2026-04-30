"""Run point-in-time diagnostics against current local artifacts.

This audit is intentionally slower and more artifact-aware than the unit tests.
It does not prove the absence of every possible leak, but it makes the major
point-in-time assumptions visible and writes a reviewer-friendly report.
"""
from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd



from src.config import load_config  # noqa: E402
from src.features.registry import build_active_producers  # noqa: E402
from src.features.fundamentals.growth_lifecycle import annual_fundamental_panel  # noqa: E402
from src.utils.dates import as_of_merge  # noqa: E402
from src.utils.io import ensure_dir  # noqa: E402


@dataclass
class AuditCheck:
    name: str
    status: str
    detail: str


def main(config_name: str = "baseline", output_path: str = "report/point_in_time_audit.md") -> None:
    """Run PIT checks and write a markdown report."""
    config = load_config(config_name)
    checks: list[AuditCheck] = []
    checks.append(check_as_of_merge_invariant())
    checks.extend(check_feature_artifacts(config))
    checks.extend(check_historical_text_artifacts())
    checks.extend(check_relationship_artifact(config))
    checks.extend(check_fundamentals_artifact(config))
    checks.extend(check_registered_producer_contracts(config))

    output = Path(output_path)
    ensure_dir(output.parent)
    output.write_text(render_report(checks), encoding="utf-8")
    print(f"Wrote point-in-time audit to {output}")


def check_as_of_merge_invariant() -> AuditCheck:
    """Verify the low-level as-of merge does not select future rows."""
    left = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "BBB"],
            "date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31", "2024-02-29"]),
        }
    )
    right = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "BBB"],
            "date": pd.to_datetime(["2024-01-15", "2024-02-29", "2024-04-01", "2024-03-01"]),
            "source_date": pd.to_datetime(["2024-01-15", "2024-02-29", "2024-04-01", "2024-03-01"]),
            "value": [1.0, 2.0, 999.0, 5.0],
        }
    )
    merged = as_of_merge(left, right, by=["ticker"])
    known_sources = merged.dropna(subset=["source_date"])
    if (known_sources["source_date"] <= known_sources["date"]).all() and pd.isna(merged.loc[3, "value"]):
        return AuditCheck("as_of_merge invariant", "pass", "Synthetic merge never selected future feature rows.")
    return AuditCheck("as_of_merge invariant", "fail", "Synthetic merge selected a future feature row.")


def check_feature_artifacts(config: dict) -> list[AuditCheck]:
    """Check current feature parquets for basic PIT-auditable invariants."""
    feature_dir = Path(config["paths"]["feature_dir"])
    checks: list[AuditCheck] = []
    if not feature_dir.exists():
        return [AuditCheck("feature artifacts", "fail", f"Feature directory missing: {feature_dir}")]

    for path in sorted(feature_dir.glob("*.parquet")):
        if path.name == "feature_manifest.parquet":
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            checks.append(AuditCheck(path.name, "fail", f"Could not read parquet: {exc}"))
            continue
        missing = {"ticker", "date"} - set(frame.columns)
        if missing:
            checks.append(AuditCheck(path.name, "warn", f"Missing ticker/date columns: {sorted(missing)}"))
            continue
        duplicates = int(frame.duplicated(["ticker", "date"]).sum())
        min_date = pd.to_datetime(frame["date"], errors="coerce").min()
        max_date = pd.to_datetime(frame["date"], errors="coerce").max()
        status = "pass" if duplicates == 0 else "fail"
        detail = f"rows={len(frame):,}, tickers={frame['ticker'].nunique():,}, date_range={date_text(min_date)} to {date_text(max_date)}, duplicate_keys={duplicates}"
        checks.append(AuditCheck(f"feature artifact {path.name}", status, detail))
    return checks


def check_historical_text_artifacts() -> list[AuditCheck]:
    """Check compact historical text artifacts and date-bearing columns."""
    checks: list[AuditCheck] = []
    for directory in ["data/processed/historical_text", "data/processed/historical_text_10k_10q"]:
        root = Path(directory)
        manifest_path = root / "historical_text_manifest.json"
        embeddings_path = root / "historical_section_embeddings.parquet"
        counts_path = root / "historical_section_topic_counts.parquet"
        if not manifest_path.exists():
            checks.append(AuditCheck(f"{directory} manifest", "warn", "Manifest is missing."))
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        detail = (
            f"forms={manifest.get('forms')}, filings={manifest.get('filings'):,}, "
            f"embedding_rows={manifest.get('embedding_rows'):,}, failures={manifest.get('failures')}"
        )
        checks.append(AuditCheck(f"{directory} manifest", "pass", detail))
        for path in [embeddings_path, counts_path]:
            if not path.exists():
                checks.append(AuditCheck(str(path), "warn", "Expected historical text artifact is missing."))
                continue
            frame = pd.read_parquet(path, columns=["ticker", "filing_date", "section"])
            dates = pd.to_datetime(frame["filing_date"], errors="coerce")
            missing_dates = int(dates.isna().sum())
            status = "pass" if missing_dates == 0 else "fail"
            checks.append(
                AuditCheck(
                    path.name,
                    status,
                    f"rows={len(frame):,}, tickers={frame['ticker'].nunique():,}, sections={frame['section'].nunique():,}, filing_date_missing={missing_dates}",
                )
            )
    return checks


def check_relationship_artifact(config: dict) -> list[AuditCheck]:
    """Check relationship graph has filing dates needed for PIT graph construction."""
    path = Path(config["paths"]["relationships_path"])
    if not path.exists():
        return [AuditCheck("relationships", "warn", f"Relationship graph missing: {path}")]
    frame = pd.read_parquet(path)
    if "filing_date" not in frame.columns:
        return [AuditCheck("relationships", "fail", "Relationship graph lacks filing_date.")]
    dates = pd.to_datetime(frame["filing_date"], errors="coerce")
    missing = int(dates.isna().sum())
    status = "pass" if missing == 0 else "fail"
    return [
        AuditCheck(
            "relationships filing_date",
            status,
            f"rows={len(frame):,}, missing_filing_date={missing}, date_range={date_text(dates.min())} to {date_text(dates.max())}",
        )
    ]


def check_fundamentals_artifact(config: dict) -> list[AuditCheck]:
    """Check fundamentals have both filing_date and end_date so lag is auditable."""
    path = Path(config["paths"]["fundamentals_path"])
    if not path.exists():
        return [AuditCheck("fundamentals", "warn", f"Fundamentals parquet missing: {path}")]
    frame = pd.read_parquet(path)
    required = {"ticker", "concept", "value", "end_date", "filing_date", "form"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        return [AuditCheck("fundamentals schema", "fail", f"Missing columns: {sorted(missing_columns)}")]
    filing_dates = pd.to_datetime(frame["filing_date"], errors="coerce")
    end_dates = pd.to_datetime(frame["end_date"], errors="coerce")
    missing_dates = int(filing_dates.isna().sum())
    period_after_filing = int((end_dates > filing_dates).fillna(False).sum())
    annual_panel = annual_fundamental_panel(frame)
    status = "pass" if missing_dates == 0 and period_after_filing == 0 else "warn"
    detail = (
        f"rows={len(frame):,}, tickers={frame['ticker'].nunique():,}, concepts={frame['concept'].nunique():,}, "
        f"missing_filing_date={missing_dates}, raw_end_date_after_filing={period_after_filing}, "
        f"annual_panel_rows_after_pit_filter={len(annual_panel):,}"
    )
    return [AuditCheck("fundamentals filing lag fields", status, detail)]


def check_registered_producer_contracts(config: dict) -> list[AuditCheck]:
    """List registered producer schemas so audit output captures active contracts."""
    checks: list[AuditCheck] = []
    producers = build_active_producers(config)
    for name, producer in producers.items():
        spec = producer.spec
        checks.append(
            AuditCheck(
                f"registered producer {name}",
                "pass",
                f"source={spec.source}, keys={spec.key_columns}, feature_columns={len(spec.columns)}",
            )
        )
    return checks


def render_report(checks: list[AuditCheck]) -> str:
    """Render the audit checks and known limitations as markdown."""
    status_counts = {status: sum(1 for check in checks if check.status == status) for status in ["pass", "warn", "fail"]}
    lines = [
        "# Point-In-Time Audit",
        "",
        "This report checks whether the current local artifacts expose the fields and invariants needed for point-in-time evaluation. It is a diagnostic, not a formal proof that no leakage exists.",
        "",
        "## Summary",
        "",
        f"- Pass: `{status_counts['pass']}`",
        f"- Warn: `{status_counts['warn']}`",
        f"- Fail: `{status_counts['fail']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in checks:
        lines.append(f"| {check.name} | {check.status} | {check.detail.replace('|', '/')} |")

    lines.extend(
        [
            "",
            "## Known Limitations",
            "",
            "- `report/theme_labels.csv` contains manually curated theme labels assigned with full-sample knowledge. Theme loadings may be point-in-time, but the human-readable labels are interpretive and full-sample.",
            "- Dashboard PCA projections are fit on all available points for a stable visual map. The projection therefore knows the future geometry, even when the underlying firm-date loadings are point-in-time.",
            "- GICS metadata is currently static over the sample and may not reflect historical sector or sub-industry classifications.",
            "- Model retraining on a longer sample can change embeddings for earlier observations through learned weights and normalization. For strict PIT backtests, train models only on data available up to the training cutoff and freeze them for that evaluation window.",
            "- The audit verifies artifact structure and selected invariants. It does not inspect every row-level source dependency in raw SEC filings.",
            "",
        ]
    )
    return "\n".join(lines)


def date_text(value: object) -> str:
    """Format a possibly-missing timestamp for markdown."""
    if pd.isna(value):
        return "NA"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    output = sys.argv[2] if len(sys.argv) > 2 else "report/point_in_time_audit.md"
    main(config, output)

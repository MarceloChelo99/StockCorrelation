from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import sys
from pathlib import Path

import pandas as pd



from src.utils.io import ensure_dir
from src.utils.logging import log


def main(slice_results_path: str, output_path: str = "report/hybrid_covariance_policy.md") -> None:
    """Summarize an in-sample routing policy from covariance slice results."""
    slices = pd.read_csv(slice_results_path)
    usable = slices[slices["status"] == "ok"].copy()
    usable["preferred_method"] = usable.apply(preferred_method, axis=1)
    usable["annual_variance_improvement"] = usable["ledoit_wolf_annual_variance"] - usable[
        "embedding_annual_variance"
    ]
    usable["relative_variance_improvement"] = usable["annual_variance_improvement"] / usable[
        "ledoit_wolf_annual_variance"
    ]

    output = Path(output_path)
    ensure_dir(output.parent)
    usable.to_csv(output.with_suffix(".csv"), index=False)
    output.write_text(markdown_summary(usable), encoding="utf-8")
    log(f"Wrote hybrid covariance policy summary to {output}.", tag="hybrid-cov")


def preferred_method(row: pd.Series) -> str:
    """Choose the lower-realized-variance estimator for a slice."""
    if float(row["embedding_annual_variance"]) < float(row["ledoit_wolf_annual_variance"]):
        return "embedding_prior"
    return "ledoit_wolf"


def markdown_summary(usable: pd.DataFrame) -> str:
    """Format the covariance routing policy as markdown."""
    embedding_rows = usable[usable["preferred_method"] == "embedding_prior"]
    ledoit_rows = usable[usable["preferred_method"] == "ledoit_wolf"]

    lines = [
        "# Hybrid Covariance Policy",
        "",
        "This is an in-sample routing summary from the covariance slice analysis, not an out-of-sample production estimator.",
        "It asks where embedding-prior shrinkage realized lower variance than Ledoit-Wolf in the current evaluation window.",
        "",
        f"- Slices favoring embedding-prior shrinkage: `{len(embedding_rows)}`",
        f"- Slices favoring Ledoit-Wolf: `{len(ledoit_rows)}`",
        "",
        "Embedding-prior slices:",
        "",
        "| Slice Type | Slice | LW Ann Var | Emb Ann Var | Relative Improvement |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in embedding_rows.sort_values("relative_variance_improvement", ascending=False).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["slice_type"]),
                    str(row["slice"]),
                    f"{float(row['ledoit_wolf_annual_variance']):.5f}",
                    f"{float(row['embedding_annual_variance']):.5f}",
                    f"{100.0 * float(row['relative_variance_improvement']):.2f}%",
                ]
            )
            + " |"
        )

    lines.extend(["", "Ledoit-Wolf slices:", "", "| Slice Type | Slice | LW Ann Var | Emb Ann Var | Relative Improvement |", "| --- | --- | ---: | ---: | ---: |"])
    for _, row in ledoit_rows.sort_values("relative_variance_improvement").iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["slice_type"]),
                    str(row["slice"]),
                    f"{float(row['ledoit_wolf_annual_variance']):.5f}",
                    f"{float(row['embedding_annual_variance']):.5f}",
                    f"{100.0 * float(row['relative_variance_improvement']):.2f}%",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: .venv/bin/python -m scripts.analysis.hybrid_covariance_policy "
            "<covariance_slices.csv> [output_path]"
        )
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "report/hybrid_covariance_policy.md")

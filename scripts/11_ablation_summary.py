from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_dir
from src.utils.logging import log


KNOWN_RUNS = {
    "semantic_embedding",
    "risk_embedding",
    "ae_events_only",
    "ae_price_only",
    "ae_price_minilm",
    "ae_minilm",
    "ae_minilm_text_only",
    "ae_text_baseline",
}


def main(output_path: str = "report/feature_ablation_summary.csv") -> None:
    """Summarize existing feature-ablation experiments into one table."""
    rows = []
    for root in sorted(Path("experiments").glob("*")):
        config = read_json(root / "config.yaml")
        metrics = read_json(root / "metrics.json")
        training = read_json(root / "training_history.json")
        name = config.get("experiment_name", root.name)
        if name not in KNOWN_RUNS:
            continue

        clustering = metrics.get("clustering", {})
        peers = metrics.get("peers", {})
        covariance = metrics.get("covariance", {})
        covariance_methods = covariance.get("methods", {})
        rows.append(
            {
                "experiment_name": name,
                "label": feature_label(config),
                "features_enabled": ",".join(config.get("features", {}).get("enabled", [])),
                "final_reconstruction_mse": training.get("final_reconstruction_mse"),
                "clustering_ari": clustering.get("ari"),
                "clustering_nmi": clustering.get("nmi"),
                "peer_embedding_corr": peers.get("mean_embedding_peer_corr"),
                "peer_benchmark_corr": peers.get("mean_benchmark_peer_corr"),
                "peer_corr_diff": peers.get("mean_corr_diff"),
                "cov_embedding_annual_variance": covariance_methods.get("embedding_prior", {}).get("realized_annual_variance"),
                "cov_ledoit_wolf_annual_variance": covariance_methods.get("ledoit_wolf", {}).get("realized_annual_variance"),
                "path": str(root),
            }
        )

    if not rows:
        raise ValueError("No known ablation runs found under experiments/.")

    result = pd.DataFrame(rows)
    result["has_covariance"] = result["cov_embedding_annual_variance"].notna()
    result = result.sort_values(["label", "has_covariance"], ascending=[True, False])
    result = result.drop_duplicates(subset=["label", "features_enabled"], keep="first")
    result = result.drop(columns=["has_covariance"])
    result = result.sort_values("clustering_nmi", ascending=False, na_position="last")
    output = Path(output_path)
    ensure_dir(output.parent)
    result.to_csv(output, index=False)
    output.with_suffix(".md").write_text(markdown_summary(result), encoding="utf-8")
    log(f"Wrote ablation summary to {output}.", tag="ablation")


def read_json(path: Path) -> dict:
    """Read a JSON dict if it exists, otherwise return an empty dict."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def feature_label(config: dict) -> str:
    """Return a readable feature-group label from a resolved config."""
    enabled = set(config.get("features", {}).get("enabled", []))
    has_price = any(name.startswith("price_") for name in enabled)
    has_text = "text_business" in enabled or "text_risk" in enabled
    has_events = "event_item_frequency" in enabled
    text_method = config.get("features", {}).get("text", {}).get("method", "hashed")

    parts = []
    if has_price:
        parts.append("price")
    if has_text:
        parts.append(f"{text_method} text")
    if has_events:
        parts.append("8-K counts")
    return " + ".join(parts) if parts else "unknown"


def markdown_summary(result: pd.DataFrame) -> str:
    """Format ablation results for the report."""
    lines = [
        "# Feature Ablation Summary",
        "",
        "| Experiment | Features | NMI | Peer Diff | Cov Annual Var |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in result.iterrows():
        cov_value = row.get("cov_embedding_annual_variance")
        cov_text = "" if pd.isna(cov_value) else f"{float(cov_value):.5f}"
        peer_value = row.get("peer_corr_diff")
        peer_text = "" if pd.isna(peer_value) else f"{float(peer_value):.3f}"
        nmi_value = row.get("clustering_nmi")
        nmi_text = "" if pd.isna(nmi_value) else f"{float(nmi_value):.3f}"
        lines.append(
            f"| {row['label']} | {row['features_enabled']} | {nmi_text} | {peer_text} | {cov_text} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "report/feature_ablation_summary.csv")

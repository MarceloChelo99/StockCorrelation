"""Run a small temporal-autoencoder lambda/alpha sweep.

The sweep trains one temporal business-view model per parameter combination,
then evaluates the validation checks that decide whether the model is useful
for trajectory analysis rather than merely smooth.
"""
from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.temporal_validation.ai_drift_test import AI_MOVERS, CONTROL_FIRMS
from scripts.temporal_validation.common import (
    displacement,
    load_embeddings,
    monthly_velocities,
    standardized_embeddings,
    write_markdown,
)
from scripts.temporal_validation.event_alignment_test import EVENTS
from scripts.temporal_validation.stability_test import STABLE_FIRMS
from src.config import load_config, save_config
from src.evaluation.clustering import ClusteringEvaluator
from src.models.train import train_embedding_model
from src.utils.io import read_json
from src.utils.logging import log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="temporal_business_view")
    parser.add_argument("--vanilla-embeddings", required=True)
    parser.add_argument("--vanilla-metrics", default="")
    parser.add_argument("--output-dir", default="report/temporal_validation")
    parser.add_argument("--sweep-root", default="experiments/_runtime/temporal_sweep")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.1, 0.5, 1.0, 2.0])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--max-combos", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"]["epochs"] = int(args.epochs)
    dataset_path = Path(config["paths"]["dataset_dir"]) / f"{config['assembly']['dataset_name']}.parquet"
    dataset = pd.read_parquet(dataset_path)
    metadata = load_metadata(config)
    vanilla = load_embeddings(args.vanilla_embeddings)
    baseline_metrics = baseline_clustering(config, metadata, vanilla, args.vanilla_metrics)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = Path(args.sweep_root) / timestamp
    sweep_root.mkdir(parents=True, exist_ok=False)

    combos = [(lambda_temp, alpha) for lambda_temp in args.lambdas for alpha in args.alphas]
    if args.max_combos > 0:
        combos = combos[: args.max_combos]

    rows = []
    for combo_index, (lambda_temp, alpha) in enumerate(combos, start=1):
        name = f"lambda_{lambda_temp:g}_alpha_{alpha:g}".replace(".", "p")
        output_dir = sweep_root / name
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "model").mkdir(exist_ok=False)
        combo_config = dict(config)
        combo_config["model"] = dict(config["model"])
        combo_config["model"]["lambda_temp"] = float(lambda_temp)
        combo_config["model"]["alpha"] = float(alpha)
        combo_config["experiment_name"] = f"temporal_sweep_{name}"
        save_config(combo_config, output_dir / "config.yaml")

        log(f"Sweep {combo_index}/{len(combos)}: lambda={lambda_temp}, alpha={alpha}.", tag="temporal-sweep")
        _, embeddings, history = train_embedding_model(
            dataset,
            combo_config,
            model_dir=output_dir / "model",
            embeddings_path=output_dir / "embeddings.parquet",
            history_path=output_dir / "training_history.json",
        )
        metrics = evaluate_candidate(
            embeddings=embeddings,
            vanilla_embeddings=vanilla,
            metadata=metadata,
            config=combo_config,
            baseline_nmi=float(baseline_metrics["nmi"]),
            start_year=2018,
            end_year=2024,
        )
        rows.append(
            {
                "lambda_temp": float(lambda_temp),
                "alpha": float(alpha),
                "run_dir": str(output_dir),
                "epochs_trained": int(history.get("epochs_trained", 0)),
                "val_temporal_loss": float(history.get("final_temporal_loss", np.nan)),
                **metrics,
            }
        )

    results = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "hyperparameter_sweep.csv"
    results.to_csv(results_path, index=False)
    write_summary(results, baseline_metrics, output_dir / "hyperparameter_sweep.md")
    log(f"Wrote sweep results to {results_path}.", tag="temporal-sweep")


def load_metadata(config: dict) -> pd.DataFrame:
    """Load sector metadata for clustering validation."""
    metadata_path = Path(config["paths"].get("metadata_path", ""))
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata path not found: {metadata_path}")
    metadata = pd.read_parquet(metadata_path)
    if "ticker" not in metadata.columns:
        raise ValueError(f"Metadata file {metadata_path} must contain a ticker column.")
    metadata = metadata.copy()
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
    return metadata


def baseline_clustering(
    config: dict,
    metadata: pd.DataFrame,
    vanilla_embeddings: pd.DataFrame,
    vanilla_metrics_path: str,
) -> dict:
    """Return vanilla clustering metrics from disk or recompute them."""
    if vanilla_metrics_path:
        payload = read_json(vanilla_metrics_path)
        if "clustering" in payload:
            return payload["clustering"]
    return ClusteringEvaluator().run(vanilla_embeddings, metadata, config, output_dir=None)


def evaluate_candidate(
    *,
    embeddings: pd.DataFrame,
    vanilla_embeddings: pd.DataFrame,
    metadata: pd.DataFrame,
    config: dict,
    baseline_nmi: float,
    start_year: int,
    end_year: int,
) -> dict:
    """Compute validation metrics for one trained temporal candidate."""
    temporal = standardized_embeddings(embeddings)
    vanilla = standardized_embeddings(vanilla_embeddings)
    clustering = ClusteringEvaluator().run(embeddings, metadata, config, output_dir=None)
    ai = ai_drift_metrics(temporal)
    stability = stability_metrics(temporal, vanilla, start_year=start_year, end_year=end_year)
    event = event_alignment_metrics(temporal)
    nmi_ratio = float(clustering["nmi"]) / baseline_nmi if baseline_nmi > 0 else float("nan")
    nmi_pass = bool(nmi_ratio >= 0.90)
    overall_pass = bool(ai["ai_pass"] and stability["stability_pass"] and event["event_pass"] and nmi_pass)
    score = (
        min(float(ai["ai_ratio"]) / 2.0, 1.5)
        + min(float(stability["stability_improvement_ratio"]), 1.5)
        + min(float(event["event_pass_rate"]) / 0.60, 1.5)
        + min(nmi_ratio / 0.90, 1.5)
    )
    return {
        "clustering_ari": float(clustering["ari"]),
        "clustering_nmi": float(clustering["nmi"]),
        "baseline_nmi": baseline_nmi,
        "nmi_ratio_to_vanilla": nmi_ratio,
        "nmi_pass": nmi_pass,
        **ai,
        **stability,
        **event,
        "overall_pass": overall_pass,
        "selection_score": float(score),
    }


def ai_drift_metrics(embeddings: pd.DataFrame) -> dict:
    """Measure whether AI movers displaced more than stable controls."""
    rows = []
    for group, tickers in [("ai_mover", AI_MOVERS), ("control", CONTROL_FIRMS)]:
        for ticker in tickers:
            rows.append(
                {
                    "ticker": ticker,
                    "group": group,
                    "displacement": displacement(embeddings, ticker, "2022-12-31", "2024-12-31"),
                }
            )
    frame = pd.DataFrame(rows)
    ai_median = frame.loc[frame["group"] == "ai_mover", "displacement"].median(skipna=True)
    control_median = frame.loc[frame["group"] == "control", "displacement"].median(skipna=True)
    ratio = ai_median / control_median if pd.notna(control_median) and control_median > 0 else np.nan
    return {
        "ai_median_displacement": float(ai_median),
        "control_median_displacement": float(control_median),
        "ai_ratio": float(ratio),
        "ai_pass": bool(pd.notna(ratio) and ratio >= 2.0),
    }


def stability_metrics(
    temporal_embeddings: pd.DataFrame,
    vanilla_embeddings: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> dict:
    """Measure whether stable firms move less under temporal smoothing."""
    rows = []
    for ticker in STABLE_FIRMS:
        for year in range(start_year, end_year):
            start = f"{year}-12-31"
            end = f"{year + 1}-12-31"
            rows.append(
                {
                    "ticker": ticker,
                    "temporal": displacement(temporal_embeddings, ticker, start, end),
                    "vanilla": displacement(vanilla_embeddings, ticker, start, end),
                }
            )
    frame = pd.DataFrame(rows)
    temporal_median = frame["temporal"].median(skipna=True)
    vanilla_median = frame["vanilla"].median(skipna=True)
    improvement = 1.0 - temporal_median / vanilla_median if pd.notna(vanilla_median) and vanilla_median > 0 else np.nan
    return {
        "stable_temporal_median": float(temporal_median),
        "stable_vanilla_median": float(vanilla_median),
        "stability_improvement_ratio": float(improvement),
        "stability_pass": bool(pd.notna(improvement) and improvement > 0.0),
    }


def event_alignment_metrics(embeddings: pd.DataFrame) -> dict:
    """Measure whether known events create velocity spikes."""
    passed = []
    ratios = []
    for event in EVENTS:
        velocities = monthly_velocities(embeddings, event["ticker"])
        if velocities.empty:
            passed.append(False)
            continue
        event_date = pd.Timestamp(event["event_date"])
        event_mask = (velocities["end_date"] - event_date).abs().dt.days <= 45
        baseline_mask = (velocities["end_date"] - event_date).abs().dt.days > 180
        event_velocity = velocities.loc[event_mask, "velocity"].max()
        baseline = velocities.loc[baseline_mask, "velocity"]
        baseline = baseline.loc[baseline > 1e-12]
        baseline_median = baseline.median()
        ratio = event_velocity / baseline_median if pd.notna(event_velocity) and baseline_median > 0 else np.nan
        ratios.append(float(ratio) if pd.notna(ratio) else np.nan)
        passed.append(bool(pd.notna(ratio) and ratio > 1.5))
    pass_rate = float(np.mean(passed)) if passed else 0.0
    return {
        "event_pass_rate": pass_rate,
        "event_median_spike_ratio": float(np.nanmedian(ratios)) if ratios else float("nan"),
        "event_pass": bool(pass_rate >= 0.60),
    }


def write_summary(results: pd.DataFrame, baseline_metrics: dict, path: Path) -> None:
    """Write a compact Markdown summary of the sweep."""
    if results.empty:
        write_markdown(path, "Temporal Hyperparameter Sweep", ["No candidates were run."])
        return
    passed = results.loc[results["overall_pass"]]
    if not passed.empty:
        best = passed.sort_values(["selection_score", "clustering_nmi"], ascending=False).iloc[0]
        selection_note = "At least one candidate passed all validation gates."
    else:
        best = results.sort_values(["selection_score", "clustering_nmi"], ascending=False).iloc[0]
        selection_note = "No candidate passed all validation gates; best row is ranked by diagnostic score."
    top_columns = [
        "lambda_temp",
        "alpha",
        "overall_pass",
        "selection_score",
        "ai_ratio",
        "stability_improvement_ratio",
        "event_pass_rate",
        "nmi_ratio_to_vanilla",
        "clustering_nmi",
        "run_dir",
    ]
    top = results.sort_values(["overall_pass", "selection_score"], ascending=False).head(5)
    table = markdown_table(top.loc[:, top_columns])
    write_markdown(
        path,
        "Temporal Hyperparameter Sweep",
        [
            f"Vanilla baseline NMI: `{float(baseline_metrics['nmi']):.4f}`",
            selection_note,
            "",
            "Best candidate:",
            f"- `lambda_temp = {float(best['lambda_temp']):g}`",
            f"- `alpha = {float(best['alpha']):g}`",
            f"- Overall pass: `{bool(best['overall_pass'])}`",
            f"- AI ratio: `{float(best['ai_ratio']):.3f}`",
            f"- Stability improvement ratio: `{float(best['stability_improvement_ratio']):.3f}`",
            f"- Event pass rate: `{float(best['event_pass_rate']):.2%}`",
            f"- NMI ratio to vanilla: `{float(best['nmi_ratio_to_vanilla']):.3f}`",
            f"- Run dir: `{best['run_dir']}`",
            "",
            "Top candidates:",
            "",
            table,
        ],
    )


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without optional tabulate dependency."""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

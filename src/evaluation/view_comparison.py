"""Compare whether view-specific soft clusters encode distinct structures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.base import Evaluator
from src.evaluation.clustering import normalized_mutual_info
from src.utils.io import ensure_dir


class ViewComparisonEvaluator(Evaluator):
    """Compute cross-view NMI between hard GMM assignments."""

    name = "view_comparison"

    def run(self, embeddings, metadata, config, db=None, output_dir=None) -> dict:
        if output_dir is None:
            raise ValueError("View comparison requires experiment output_dir with view loadings.")
        root = Path(output_dir)
        labels = {}
        for view_name in config.get("views_enabled", list(config["views"].keys())):
            path = root / "views" / view_name / "loadings.parquet"
            if not path.exists():
                continue
            loadings = pd.read_parquet(path)
            labels[view_name] = latest_hard_labels(loadings)
        if len(labels) < 2:
            raise ValueError("Need at least two view loading files for view comparison.")

        views = sorted(labels)
        matrix = pd.DataFrame(np.eye(len(views)), index=views, columns=views)
        for left_index, left in enumerate(views):
            for right in views[left_index + 1 :]:
                joined = labels[left].rename("left").to_frame().join(labels[right].rename("right"), how="inner")
                score = normalized_mutual_info(joined["left"].astype(str).to_numpy(), joined["right"].astype(str).to_numpy())
                matrix.loc[left, right] = score
                matrix.loc[right, left] = score

        if output_dir is not None:
            artifacts_dir = ensure_dir(root / "evaluation")
            matrix.to_csv(artifacts_dir / "view_nmi_matrix.csv")
        off_diagonal = matrix.to_numpy()[~np.eye(len(matrix), dtype=bool)]
        return {
            "views": views,
            "mean_off_diagonal_nmi": float(off_diagonal.mean()) if len(off_diagonal) else 0.0,
            "max_off_diagonal_nmi": float(off_diagonal.max()) if len(off_diagonal) else 0.0,
        }


def latest_hard_labels(loadings: pd.DataFrame) -> pd.Series:
    """Return hard theme labels from latest soft loadings."""
    frame = loadings.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    theme_columns = [column for column in frame.columns if column.startswith("theme_")]
    labels = frame.loc[:, theme_columns].astype(float).to_numpy().argmax(axis=1)
    return pd.Series(labels, index=frame["ticker"], name="label")

"""Point-in-time historical filing-text embedding features.

This producer promotes the compact historical MiniLM stream into the standard
feature pipeline. It reads section-level filing embeddings, selects the latest
available business/risk-style section vectors as of each firm-month, concatenates
those views, and PCA-reduces them into ``text_hist_emb_*`` columns.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.features.assembly import build_observation_panel
from src.features.base import FeatureProducer, FeatureSpec
from src.utils.dates import as_of_merge


class HistoricalTextFeatureProducer(FeatureProducer):
    """Build PCA-reduced point-in-time features from historical section embeddings."""

    def __init__(
        self,
        target_dim: int,
        *,
        input_path: str | Path,
        aggregation: str = "concat_business_risk",
        business_sections: list[str] | None = None,
        risk_sections: list[str] | None = None,
        pca_fit_end_date: str | None = None,
    ) -> None:
        self.target_dim = int(target_dim)
        self.input_path = Path(input_path)
        self.aggregation = str(aggregation)
        self.business_sections = business_sections or ["business"]
        self.risk_sections = risk_sections or ["risk_factors", "q_risk_factors"]
        self.pca_fit_end_date = pca_fit_end_date
        self.pca_: PCA | None = None
        self.pca_metadata_: dict[str, object] = {}

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="text_historical",
            key_columns=["ticker", "date"],
            columns=[f"text_hist_emb_{index}" for index in range(self.target_dim)],
            source="historical_section_embeddings",
            description="PCA-reduced point-in-time MiniLM section embeddings from historical filings.",
        )

    def compute(self, db, config: dict) -> pd.DataFrame:
        """Compute monthly point-in-time text features from historical section embeddings."""
        if self.aggregation != "concat_business_risk":
            raise ValueError("Only aggregation='concat_business_risk' is currently supported.")
        if not self.input_path.exists():
            raise FileNotFoundError(f"Historical text embeddings not found at {self.input_path}.")

        panel = build_observation_panel(db, config)
        if panel.empty:
            return pd.DataFrame(columns=self.spec.all_columns)
        panel["ticker"] = panel["ticker"].astype(str).str.upper()
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce")

        embeddings = load_historical_embeddings(self.input_path)
        embedding_columns = embedding_column_names(embeddings)
        wide = panel.loc[:, ["ticker", "date"]].copy()
        group_specs = {
            "business": self.business_sections,
            "risk": self.risk_sections,
        }
        for group_name, sections in group_specs.items():
            group = latest_section_group_frame(embeddings, embedding_columns, sections, group_name)
            wide = as_of_merge(wide, group, by=["ticker"])

        feature_columns = [column for column in wide.columns if column.startswith("text_hist_raw_")]
        if not feature_columns:
            raise ValueError("No historical text raw feature columns were produced.")
        raw_matrix = wide.loc[:, feature_columns].astype(float).fillna(0.0).to_numpy()
        if raw_matrix.shape[1] < self.target_dim:
            raise ValueError(
                f"text_historical target_dim={self.target_dim} exceeds raw dimension {raw_matrix.shape[1]}."
            )

        fit_mask = pca_fit_mask(wide["date"], self.pca_fit_end_date)
        signal_mask = np.abs(raw_matrix).sum(axis=1) > 0.0
        fit_mask = fit_mask & signal_mask
        if int(fit_mask.sum()) < self.target_dim:
            raise ValueError(
                f"Need at least {self.target_dim} non-empty PCA training rows, got {int(fit_mask.sum())}."
            )

        pca = PCA(n_components=self.target_dim, random_state=int(config["random_seed"]))
        pca.fit(raw_matrix[fit_mask])
        transformed = pca.transform(raw_matrix)
        transformed[~signal_mask, :] = np.nan
        self.pca_ = pca
        self.pca_metadata_ = {
            "input_path": str(self.input_path),
            "aggregation": self.aggregation,
            "business_sections": list(self.business_sections),
            "risk_sections": list(self.risk_sections),
            "raw_feature_columns": feature_columns,
            "embedding_columns": embedding_columns,
            "target_dim": self.target_dim,
            "pca_fit_end_date": self.pca_fit_end_date,
            "pca_training_rows": int(fit_mask.sum()),
            "total_rows": int(len(wide)),
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        }

        output = wide.loc[:, ["ticker", "date"]].copy()
        for index, column in enumerate(self.spec.columns):
            output[column] = transformed[:, index].astype(float)
        return output.loc[:, self.spec.all_columns].reset_index(drop=True)

    def run(self, db, config: dict, output_dir: str | Path) -> pd.DataFrame:
        """Compute features and persist the PCA object beside the feature parquet."""
        frame = self.compute(db, config)
        self._validate(frame)
        output_path = self.output_path(output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)
        if self.pca_ is not None:
            with (output_path.parent / "text_historical_pca.pkl").open("wb") as handle:
                pickle.dump(self.pca_, handle)
            (output_path.parent / "text_historical_pca_meta.json").write_text(
                json.dumps(self.pca_metadata_, indent=2),
                encoding="utf-8",
            )
        return frame


def load_historical_embeddings(path: Path) -> pd.DataFrame:
    """Load and canonicalize the historical section embedding parquet."""
    frame = pd.read_parquet(path)
    required = {"ticker", "filing_date", "section"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Historical text embeddings are missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["filing_date"], errors="coerce")
    frame["section"] = frame["section"].astype(str)
    frame = frame.dropna(subset=["ticker", "date", "section"])
    return frame.sort_values(["ticker", "date", "section"]).reset_index(drop=True)


def embedding_column_names(frame: pd.DataFrame) -> list[str]:
    """Return embedding columns sorted by numeric suffix."""
    columns = [column for column in frame.columns if column.startswith("embedding_")]
    if not columns:
        raise ValueError("Historical text embeddings must contain embedding_* columns.")
    return sorted(columns, key=lambda column: int(column.rsplit("_", 1)[1]))


def latest_section_group_frame(
    embeddings: pd.DataFrame,
    embedding_columns: list[str],
    sections: list[str],
    group_name: str,
) -> pd.DataFrame:
    """Return dated vectors for the latest available section group."""
    frame = embeddings[embeddings["section"].isin(sections)].copy()
    raw_columns = [f"text_hist_raw_{group_name}_{index}" for index in range(len(embedding_columns))]
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "date", *raw_columns])

    # Multiple matching sections on the same filing date are averaged so one
    # ticker-date contributes one business/risk vector to the as-of merge.
    grouped = frame.groupby(["ticker", "date"], as_index=False)[embedding_columns].mean()
    grouped = grouped.rename(columns={old: new for old, new in zip(embedding_columns, raw_columns, strict=True)})
    grouped = grouped.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return grouped.loc[:, ["ticker", "date", *raw_columns]].reset_index(drop=True)


def pca_fit_mask(dates: pd.Series, pca_fit_end_date: str | None) -> np.ndarray:
    """Return rows eligible for fitting the PCA transformer."""
    parsed = pd.to_datetime(dates, errors="coerce")
    mask = parsed.notna().to_numpy()
    if pca_fit_end_date:
        mask = mask & (parsed <= pd.Timestamp(pca_fit_end_date)).to_numpy()
    return mask

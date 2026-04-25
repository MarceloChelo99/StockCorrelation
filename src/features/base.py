"""Abstract base classes for feature producers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    """Schema contract for one persisted feature group."""

    name: str
    columns: list[str]
    key_columns: list[str]
    source: str
    description: str

    @property
    def all_columns(self) -> list[str]:
        return [*self.key_columns, *self.columns]


class FeatureProducer(ABC):
    """Compute and persist one feature group."""

    @property
    @abstractmethod
    def spec(self) -> FeatureSpec:
        """Return the declared output schema."""

    @abstractmethod
    def compute(self, db, config: dict) -> pd.DataFrame:
        """Compute the feature dataframe."""

    def output_path(self, output_dir: str | Path) -> Path:
        return Path(output_dir) / f"{self.spec.name}.parquet"

    def run(self, db, config: dict, output_dir: str | Path) -> pd.DataFrame:
        frame = self.compute(db, config)
        self._validate(frame)
        path = self.output_path(output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return frame

    def _validate(self, frame: pd.DataFrame) -> None:
        actual_columns = list(frame.columns)
        expected_columns = self.spec.all_columns
        if actual_columns != expected_columns:
            raise ValueError(
                f"Feature {self.spec.name!r} produced columns {actual_columns}, "
                f"expected {expected_columns}."
            )
        if frame.duplicated(self.spec.key_columns).any():
            raise ValueError(f"Feature {self.spec.name!r} contains duplicate key rows.")

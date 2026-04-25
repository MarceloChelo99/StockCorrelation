"""Abstract evaluator interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Evaluator(ABC):
    """Uniform interface for downstream embedding evaluations."""

    name: str

    @abstractmethod
    def run(
        self,
        embeddings: pd.DataFrame,
        metadata: pd.DataFrame,
        config: dict,
        db=None,
        output_dir=None,
    ) -> dict:
        """Run the evaluator and return JSON-serializable metrics."""

"""Randomness helpers for reproducible experiments."""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python and NumPy, and torch if it is installed."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

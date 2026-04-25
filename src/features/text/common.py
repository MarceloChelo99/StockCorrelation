"""Shared helpers for filing-text feature producers."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9']+")
_STOP_WORDS = {
    "about",
    "after",
    "also",
    "among",
    "because",
    "been",
    "being",
    "between",
    "could",
    "company",
    "during",
    "financial",
    "following",
    "from",
    "have",
    "into",
    "item",
    "management",
    "other",
    "results",
    "risk",
    "that",
    "their",
    "these",
    "this",
    "those",
    "under",
    "which",
    "with",
    "would",
}


def load_sections_frame(config: dict) -> pd.DataFrame:
    """Load the parsed 10-K sections parquet produced by the section parser."""
    path = Path(config["paths"]["sections_dir"]) / "ten_k_sections.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Parsed sections not found at {path}. Run scripts/01_parse_sections.py first."
        )
    frame = pd.read_parquet(path)
    if "filing_date" in frame.columns:
        frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce")
    return frame


def compute_text_feature_frame(
    frame: pd.DataFrame,
    *,
    text_column: str,
    prefix: str,
    embedding_dim: int,
    min_token_length: int,
    method: str = "hashed",
    model_name: str | None = None,
    batch_size: int = 16,
    max_chars: int = 12000,
    normalize_embeddings: bool = True,
) -> pd.DataFrame:
    """Convert one parsed filing-text column into fixed-width text vectors."""
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive.")

    rows: list[dict[str, object]] = []
    working = frame.loc[:, ["ticker", "filing_date", text_column]].copy()
    working = working.dropna(subset=[text_column, "filing_date"]).reset_index(drop=True)
    working[text_column] = working[text_column].astype(str)
    working = working[working[text_column].str.len() > 0].reset_index(drop=True)
    if working.empty:
        columns = ["ticker", "date", *vector_column_names(prefix, embedding_dim)]
        return pd.DataFrame(columns=columns)

    vector_columns = vector_column_names(prefix, embedding_dim)
    if method == "minilm":
        vectors = minilm_text_vectors(
            working[text_column].tolist(),
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
            batch_size=batch_size,
            max_chars=max_chars,
            normalize_embeddings=normalize_embeddings,
        )
        if vectors.shape[1] != embedding_dim:
            raise ValueError(
                f"Expected {embedding_dim} MiniLM dimensions for {prefix}, got {vectors.shape[1]}."
            )
    elif method == "hashed":
        vectors = np.vstack(
            [
                hashed_text_vector(
                    text,
                    embedding_dim=embedding_dim,
                    min_token_length=min_token_length,
                )
                for text in working[text_column]
            ]
        )
    else:
        raise ValueError(f"Unknown text feature method {method!r}.")

    for _, record in working.iterrows():
        vector = vectors[len(rows)]
        row: dict[str, object] = {
            "ticker": record["ticker"],
            "date": record["filing_date"],
        }
        for index, column in enumerate(vector_columns):
            row[column] = float(vector[index])
        rows.append(row)

    feature_frame = pd.DataFrame(rows)
    feature_frame = feature_frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    feature_frame = feature_frame.drop_duplicates(subset=["ticker", "date"], keep="last")
    return feature_frame.reset_index(drop=True)


def vector_column_names(prefix: str, embedding_dim: int) -> list[str]:
    """Return the stable feature column names for one hashed text vector."""
    return [f"{prefix}_{index}" for index in range(embedding_dim)]


def hashed_text_vector(text: str, *, embedding_dim: int, min_token_length: int) -> np.ndarray:
    """Project text into a stable signed hashing vector and l2-normalize it."""
    vector = np.zeros(embedding_dim, dtype=float)
    tokens = tokenize_text(text, min_token_length=min_token_length)
    for token in tokens:
        bucket, sign = stable_hash_bucket(token, embedding_dim)
        vector[bucket] += sign

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def minilm_text_vectors(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int,
    max_chars: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    """Encode filing text with a SentenceTransformer MiniLM model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MiniLM text features require sentence-transformers. "
            "Install torch and sentence-transformers or set features.text.method='hashed'."
        ) from exc

    truncated = [text[:max_chars] for text in texts]
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        truncated,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings,
    )
    return np.asarray(vectors, dtype=float)


def tokenize_text(text: str, *, min_token_length: int) -> list[str]:
    """Tokenize filing text into lowercase word tokens."""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip("'")
        if len(token) < min_token_length:
            continue
        if token in _STOP_WORDS:
            continue
        tokens.append(token)
    return tokens


def stable_hash_bucket(token: str, embedding_dim: int) -> tuple[int, int]:
    """Map a token to a deterministic vector bucket and sign."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    bucket = value % embedding_dim
    sign = -1 if (value >> 8) % 2 else 1
    return bucket, sign

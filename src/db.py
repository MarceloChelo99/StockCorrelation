"""SQLite-backed access layer for downstream pipeline stages."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FilingsDB:
    """Thin stable wrapper around the ingestion SQLite database."""

    sqlite_path: Path
    storage_dir: Path
    filings_group_name: str
    prices_group_name: str

    @classmethod
    def from_config(cls, config: dict) -> "FilingsDB":
        return cls(
            sqlite_path=Path(config["paths"]["sqlite_path"]),
            storage_dir=Path(config["paths"]["sqlite_path"]).parent,
            filings_group_name=config["data"]["filings_group_name"],
            prices_group_name=config["data"]["prices_group_name"],
        )

    def load_tickers(self) -> pd.DataFrame:
        return self._query(
            """
            SELECT *
            FROM tickers
            WHERE group_name = ?
            ORDER BY ticker ASC
            """,
            [self.filings_group_name],
        )

    def load_filings(self, form: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM filings WHERE group_name = ?"
        params: list[object] = [self.filings_group_name]
        if form is not None:
            query += " AND UPPER(form) = ?"
            params.append(form.upper())
        query += " ORDER BY ticker ASC, filing_date ASC"
        frame = self._query(query, params)
        return _parse_date_columns(frame, ["filing_date", "period_start", "period_end"])

    def load_raw_filings(self, form: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM raw_filings WHERE group_name = ?"
        params: list[object] = [self.filings_group_name]
        if form is not None:
            query += " AND UPPER(form) = ?"
            params.append(form.upper())
        query += " ORDER BY ticker ASC, filing_date ASC"
        frame = self._query(query, params)
        return _parse_date_columns(frame, ["filing_date", "period_end"])

    def load_prices(
        self,
        *,
        ticker: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> pd.DataFrame:
        query = "SELECT * FROM prices WHERE group_name = ?"
        params: list[object] = [self.prices_group_name]
        if ticker is not None:
            query += " AND ticker = ?"
            params.append(ticker.upper())
        if date_from is not None:
            query += " AND date >= ?"
            params.append(date_from)
        if date_to is not None:
            query += " AND date <= ?"
            params.append(date_to)
        query += " ORDER BY ticker ASC, date ASC"
        frame = self._query(query, params)
        return _parse_date_columns(frame, ["date"])

    def load_raw_submission_text(self, accession_no: str) -> str | None:
        """Load the full submission text for one accession from the parquet shards."""
        raw_index = self._query(
            """
            SELECT submission_text, raw_shard_path, raw_shard_row_number
            FROM raw_filings
            WHERE group_name = ? AND accession_no = ?
            LIMIT 1
            """,
            [self.filings_group_name, accession_no],
        )
        if raw_index.empty:
            return None

        record = raw_index.iloc[0]
        submission_text = record.get("submission_text")
        if isinstance(submission_text, str) and submission_text:
            return submission_text

        shard_path = record.get("raw_shard_path")
        shard_row_number = record.get("raw_shard_row_number")
        if not isinstance(shard_path, str) or not shard_path:
            return None
        try:
            row_number = int(shard_row_number)
        except (TypeError, ValueError):
            return None

        full_shard_path = self._dataset_dir(self.filings_group_name) / shard_path
        if not full_shard_path.exists():
            return None

        shard = pd.read_parquet(full_shard_path)
        if row_number < 0 or row_number >= len(shard):
            return None
        value = shard.iloc[row_number].get("submission_text")
        return str(value) if isinstance(value, str) else None

    def _query(self, query: str, params: list[object]) -> pd.DataFrame:
        if not self.sqlite_path.exists():
            raise FileNotFoundError(f"SQLite database not found at {self.sqlite_path}.")
        with sqlite3.connect(self.sqlite_path) as connection:
            return pd.read_sql_query(query, connection, params=params)

    def _dataset_dir(self, group_name: str) -> Path:
        return self.storage_dir / _slugify(group_name)


def _parse_date_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    parsed = frame.copy()
    for column in columns:
        if column in parsed.columns:
            parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
    return parsed


def _slugify(value: str) -> str:
    safe = "".join(character.lower() if character.isalnum() else "_" for character in value.strip())
    return safe.strip("_") or "raw_filing_corpus"

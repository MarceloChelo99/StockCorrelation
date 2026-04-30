"""File-backed storage and query helpers for raw filing corpora."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from shutil import rmtree

import pandas as pd

from .corpus import RAW_FILING_INDEX_COLUMNS, RawFilingCorpus, RawFilingCorpusBuilder


@dataclass(frozen=True)
class StoredCorpus:
    """Resolved storage metadata for a persisted raw filing corpus."""

    group_name: str
    dataset_dir: Path
    dataset_format: str
    raw_filings_path: Path
    raw_filing_shards_dir: Path | None
    filings_path: Path | None
    tickers_path: Path | None
    failures_path: Path | None
    manifest_path: Path

    def summary(self) -> dict[str, object]:
        """Return a compact description of the stored corpus layout."""
        return {
            "group_name": self.group_name,
            "dataset_dir": self.dataset_dir,
            "dataset_format": self.dataset_format,
            "has_raw_filings": self.raw_filings_path.exists(),
            "raw_filing_shards_dir": self.raw_filing_shards_dir,
            "has_filings": self.filings_path is not None and self.filings_path.exists(),
            "has_tickers": self.tickers_path is not None and self.tickers_path.exists(),
            "has_failures": self.failures_path is not None and self.failures_path.exists(),
            "manifest_path": self.manifest_path,
        }


class DatabaseOperator:
    """Persist and query raw filing corpora using parquet or JSONL files."""

    def __init__(
        self,
        identity: str | None = None,
        *,
        storage_dir: str | Path = "data/raw_filing_corpora",
        sqlite_path: str | Path | None = None,
        corpus_builder: RawFilingCorpusBuilder | None = None,
        default_format: str = "parquet",
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.sqlite_path = Path(sqlite_path) if sqlite_path is not None else self.storage_dir / "raw_filing_corpora.sqlite"
        self.default_format = default_format.strip().lower()
        if self.default_format not in {"parquet", "jsonl"}:
            raise ValueError("default_format must be 'parquet' or 'jsonl'.")
        self.corpus_builder = corpus_builder or RawFilingCorpusBuilder(identity=identity)

    def populate_for_symbols(
        self,
        tickers: list[str],
        *,
        group_name: str = "Custom Raw Filing Corpus",
        format: str | None = None,
        include_supporting_data: bool = True,
        filing_options: dict | None = None,
        raw_filing_options: dict | None = None,
    ) -> tuple[RawFilingCorpus, StoredCorpus]:
        corpus = self.corpus_builder.build_for_symbols(
            tickers,
            group_name=group_name,
            filing_options=filing_options,
            raw_filing_options=raw_filing_options,
        )
        stored = self.persist_corpus(
            corpus,
            format=format,
            include_supporting_data=include_supporting_data,
        )
        return corpus, stored

    def populate_for_symbols_with_sqlite(
        self,
        tickers: list[str],
        *,
        group_name: str = "Custom Raw Filing Corpus",
        format: str | None = None,
        include_supporting_data: bool = True,
        filing_options: dict | None = None,
        raw_filing_options: dict | None = None,
        sqlite_path: str | Path | None = None,
    ) -> tuple[RawFilingCorpus, StoredCorpus, Path]:
        corpus, stored = self.populate_for_symbols(
            tickers,
            group_name=group_name,
            format=format,
            include_supporting_data=include_supporting_data,
            filing_options=filing_options,
            raw_filing_options=raw_filing_options,
        )
        sqlite_db_path = self.persist_corpus_to_sqlite(corpus, stored=stored, sqlite_path=sqlite_path)
        return corpus, stored, sqlite_db_path

    def populate_public_market(
        self,
        *,
        group_name: str = "All Public Tickers",
        force_refresh: bool = False,
        limit: int | None = None,
        format: str | None = None,
        include_supporting_data: bool = True,
        filing_options: dict | None = None,
        raw_filing_options: dict | None = None,
    ) -> tuple[RawFilingCorpus, StoredCorpus]:
        corpus = self.corpus_builder.build_public_market(
            force_refresh=force_refresh,
            limit=limit,
            filing_options=filing_options,
            raw_filing_options=raw_filing_options,
        )
        if group_name != corpus.group_name:
            corpus.group_name = group_name
        stored = self.persist_corpus(
            corpus,
            format=format,
            include_supporting_data=include_supporting_data,
        )
        return corpus, stored

    def populate_public_market_with_sqlite(
        self,
        *,
        group_name: str = "All Public Tickers",
        force_refresh: bool = False,
        limit: int | None = None,
        format: str | None = None,
        include_supporting_data: bool = True,
        filing_options: dict | None = None,
        raw_filing_options: dict | None = None,
        sqlite_path: str | Path | None = None,
    ) -> tuple[RawFilingCorpus, StoredCorpus, Path]:
        corpus, stored = self.populate_public_market(
            group_name=group_name,
            force_refresh=force_refresh,
            limit=limit,
            format=format,
            include_supporting_data=include_supporting_data,
            filing_options=filing_options,
            raw_filing_options=raw_filing_options,
        )
        sqlite_db_path = self.persist_corpus_to_sqlite(corpus, stored=stored, sqlite_path=sqlite_path)
        return corpus, stored, sqlite_db_path

    def persist_corpus(
        self,
        corpus: RawFilingCorpus,
        *,
        format: str | None = None,
        include_supporting_data: bool = True,
    ) -> StoredCorpus:
        dataset_dir = self._dataset_dir(corpus.group_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_format = self._normalize_format(format)
        written = corpus.save(
            dataset_dir,
            format=dataset_format,
            include_supporting_data=include_supporting_data,
        )
        return self._stored_from_written(corpus.group_name, dataset_format, written)

    def persist_corpus_to_sqlite(
        self,
        corpus: RawFilingCorpus,
        *,
        stored: StoredCorpus | None = None,
        sqlite_path: str | Path | None = None,
    ) -> Path:
        db_path = Path(sqlite_path) if sqlite_path is not None else self.sqlite_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        raw_index = self._build_raw_filing_index(corpus, stored=stored)

        with sqlite3.connect(db_path) as connection:
            self._ensure_sqlite_schema(connection)
            self._delete_sqlite_group(connection, corpus.group_name)
            self._write_sqlite_frame(connection, "tickers", corpus.ticker_index, corpus.group_name)
            self._write_sqlite_frame(connection, "filings", corpus.filings, corpus.group_name)
            self._write_sqlite_frame(connection, "raw_filings", raw_index, corpus.group_name)
            failures = self._failures_frame(corpus.failures)
            self._write_sqlite_frame(connection, "failures", failures, corpus.group_name)
            self._upsert_sqlite_corpus_metadata(connection, corpus)
            self._create_sqlite_indexes(connection)

        return db_path

    def list_corpora(self) -> list[StoredCorpus]:
        stored: list[StoredCorpus] = []
        if not self.storage_dir.exists():
            return stored
        for manifest_path in sorted(self.storage_dir.glob("*/*_manifest.json")):
            try:
                stored.append(self._load_stored_from_manifest(manifest_path))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return stored

    def load_corpus(self, group_name: str) -> StoredCorpus:
        manifest_path = self._manifest_path(group_name)
        if not manifest_path.exists():
            raise FileNotFoundError(f"No stored corpus manifest found for group {group_name!r}.")
        return self._load_stored_from_manifest(manifest_path)

    def describe_corpus(self, group_name: str) -> dict[str, object]:
        """Load the stored corpus metadata and return its summary."""
        return self.load_corpus(group_name).summary()

    def load_raw_filings(self, group_name: str) -> pd.DataFrame:
        stored = self.load_corpus(group_name)
        return self._read_frame_by_path(stored.raw_filings_path, fallback_format="jsonl")

    def load_raw_submission_text(
        self,
        group_name: str,
        accession_no: str,
    ) -> str | None:
        raw_index = self.find_raw_filings(group_name, accession_no=accession_no)
        if raw_index.empty:
            return None
        stored = self.load_corpus(group_name)
        shard_relative_path = raw_index.iloc[0].get("raw_shard_path")
        shard_row_number = raw_index.iloc[0].get("raw_shard_row_number")
        if stored.raw_filing_shards_dir is None or shard_relative_path is None or pd.isna(shard_relative_path):
            return None
        shard_path = stored.dataset_dir / str(shard_relative_path)
        if not shard_path.exists():
            return None
        shard = pd.read_parquet(shard_path)
        try:
            row_number = int(shard_row_number)
        except (TypeError, ValueError):
            return None
        if row_number < 0 or row_number >= len(shard):
            return None
        return str(shard.iloc[row_number].get("submission_text", ""))

    def load_filings(self, group_name: str) -> pd.DataFrame:
        stored = self.load_corpus(group_name)
        if stored.filings_path is None or not stored.filings_path.exists():
            return pd.DataFrame()
        return self._read_frame(stored.filings_path, stored.dataset_format)

    def load_tickers(self, group_name: str) -> pd.DataFrame:
        stored = self.load_corpus(group_name)
        if stored.tickers_path is None or not stored.tickers_path.exists():
            return pd.DataFrame()
        return self._read_frame(stored.tickers_path, stored.dataset_format)

    def load_failures(self, group_name: str) -> pd.DataFrame:
        stored = self.load_corpus(group_name)
        if stored.failures_path is None or not stored.failures_path.exists():
            return pd.DataFrame(columns=["ticker", "stage", "error"])
        return pd.read_json(stored.failures_path, orient="records", lines=True)

    def find_raw_filings(
        self,
        group_name: str,
        *,
        ticker: str | None = None,
        form: str | None = None,
        accession_no: str | None = None,
    ) -> pd.DataFrame:
        frame = self.load_raw_filings(group_name)
        return self._filter_filings(frame, ticker=ticker, form=form, accession_no=accession_no)

    def find_structured_filings(
        self,
        group_name: str,
        *,
        ticker: str | None = None,
        form: str | None = None,
        accession_no: str | None = None,
    ) -> pd.DataFrame:
        frame = self.load_filings(group_name)
        return self._filter_filings(frame, ticker=ticker, form=form, accession_no=accession_no)

    def list_tickers(self, group_name: str) -> list[str]:
        """Return the stored ticker universe for a corpus."""
        tickers = self.load_tickers(group_name)
        if tickers.empty or "ticker" not in tickers.columns:
            return []
        return tickers["ticker"].dropna().astype(str).str.upper().tolist()

    def list_sqlite_corpora(self, *, sqlite_path: str | Path | None = None) -> list[str]:
        db_path = Path(sqlite_path) if sqlite_path is not None else self.sqlite_path
        if not db_path.exists():
            return []
        with sqlite3.connect(db_path) as connection:
            rows = connection.execute(
                "SELECT group_name FROM corpora ORDER BY updated_at DESC, group_name ASC"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def latest_filings(
        self,
        group_name: str,
        *,
        raw: bool = False,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Return the most recent stored filings for quick inspection."""
        if limit <= 0:
            raise ValueError("limit must be positive.")
        frame = self.load_raw_filings(group_name) if raw else self.load_filings(group_name)
        if frame.empty:
            return frame.copy()
        if "filing_date" in frame.columns:
            frame = frame.sort_values("filing_date", ascending=False, kind="stable")
        return frame.head(limit).reset_index(drop=True)

    def delete_corpus(self, group_name: str, *, missing_ok: bool = False) -> None:
        """Delete a stored corpus directory and all of its artifacts."""
        dataset_dir = self._dataset_dir(group_name)
        if not dataset_dir.exists():
            if missing_ok:
                return
            raise FileNotFoundError(f"No stored corpus found for group {group_name!r}.")
        rmtree(dataset_dir)

    def _stored_from_written(
        self,
        group_name: str,
        dataset_format: str,
        written: dict[str, Path],
    ) -> StoredCorpus:
        return StoredCorpus(
            group_name=group_name,
            dataset_dir=written["raw_filings"].parent,
            dataset_format=dataset_format,
            raw_filings_path=written["raw_filings"],
            raw_filing_shards_dir=written.get("raw_filing_shards_dir"),
            filings_path=written.get("filings"),
            tickers_path=written.get("tickers"),
            failures_path=written.get("failures"),
            manifest_path=written["manifest"],
        )

    def _load_stored_from_manifest(self, manifest_path: Path) -> StoredCorpus:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        group_name = str(payload["group_name"])
        dataset_format = self._normalize_format(str(payload["dataset_format"]))
        dataset_dir = manifest_path.parent
        stem = _slugify(group_name)
        extension = dataset_format
        return StoredCorpus(
            group_name=group_name,
            dataset_dir=dataset_dir,
            dataset_format=dataset_format,
            raw_filings_path=dataset_dir / f"{stem}_raw_filing_index.jsonl",
            raw_filing_shards_dir=_optional_existing_path(dataset_dir / f"{stem}_raw_filing_shards"),
            filings_path=_optional_existing_path(dataset_dir / f"{stem}_filings.{extension}"),
            tickers_path=_optional_existing_path(dataset_dir / f"{stem}_tickers.{extension}"),
            failures_path=_optional_existing_path(dataset_dir / f"{stem}_failures.jsonl"),
            manifest_path=manifest_path,
        )

    def _dataset_dir(self, group_name: str) -> Path:
        return self.storage_dir / _slugify(group_name)

    def _manifest_path(self, group_name: str) -> Path:
        stem = _slugify(group_name)
        return self._dataset_dir(group_name) / f"{stem}_manifest.json"

    def _normalize_format(self, format: str | None) -> str:
        normalized = (format or self.default_format).strip().lower()
        if normalized not in {"parquet", "jsonl"}:
            raise ValueError("format must be 'parquet' or 'jsonl'.")
        return normalized

    def _build_raw_filing_index(
        self,
        corpus: RawFilingCorpus,
        *,
        stored: StoredCorpus | None,
    ) -> pd.DataFrame:
        if stored is not None and stored.raw_filings_path.exists():
            raw_index = self._read_frame_by_path(stored.raw_filings_path, fallback_format="jsonl")
        else:
            raw_index = corpus.raw_filings.drop(columns=["submission_text"], errors="ignore").copy()
        for column in RAW_FILING_INDEX_COLUMNS:
            if column not in raw_index.columns:
                raw_index[column] = pd.NA
        return raw_index.loc[:, RAW_FILING_INDEX_COLUMNS]

    @staticmethod
    def _read_frame(path: Path, format: str) -> pd.DataFrame:
        if format == "parquet":
            return pd.read_parquet(path)
        if format == "jsonl":
            return pd.read_json(path, orient="records", lines=True)
        raise ValueError(f"Unsupported dataset format: {format}")

    @classmethod
    def _read_frame_by_path(cls, path: Path, *, fallback_format: str) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return cls._read_frame(path, "parquet")
        if suffix == ".jsonl":
            return cls._read_frame(path, "jsonl")
        return cls._read_frame(path, fallback_format)

    @staticmethod
    def _filter_filings(
        frame: pd.DataFrame,
        *,
        ticker: str | None,
        form: str | None,
        accession_no: str | None,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        filtered = frame.copy()
        if ticker:
            filtered = filtered[filtered["ticker"].astype(str).str.upper() == ticker.upper()]
        if form:
            filtered = filtered[filtered["form"].astype(str).str.upper() == form.upper()]
        if accession_no:
            filtered = filtered[filtered["accession_no"].astype(str) == accession_no]
        sort_column = "filing_date" if "filing_date" in filtered.columns else None
        if sort_column is not None:
            filtered = filtered.sort_values(sort_column, ascending=False, kind="stable")
        return filtered.reset_index(drop=True)

    def find_sqlite_raw_filings(
        self,
        group_name: str,
        *,
        ticker: str | None = None,
        form: str | None = None,
        accession_no: str | None = None,
        sqlite_path: str | Path | None = None,
    ) -> pd.DataFrame:
        return self._find_sqlite_filings(
            "raw_filings",
            group_name,
            ticker=ticker,
            form=form,
            accession_no=accession_no,
            sqlite_path=sqlite_path,
        )

    def find_sqlite_structured_filings(
        self,
        group_name: str,
        *,
        ticker: str | None = None,
        form: str | None = None,
        accession_no: str | None = None,
        sqlite_path: str | Path | None = None,
    ) -> pd.DataFrame:
        return self._find_sqlite_filings(
            "filings",
            group_name,
            ticker=ticker,
            form=form,
            accession_no=accession_no,
            sqlite_path=sqlite_path,
        )

    def sqlite_summary(self, *, sqlite_path: str | Path | None = None) -> dict[str, int]:
        db_path = Path(sqlite_path) if sqlite_path is not None else self.sqlite_path
        if not db_path.exists():
            return {"corpora": 0, "tickers": 0, "filings": 0, "raw_filings": 0, "prices": 0, "failures": 0}
        with sqlite3.connect(db_path) as connection:
            return {
                "corpora": self._sqlite_count(connection, "corpora"),
                "tickers": self._sqlite_count(connection, "tickers"),
                "filings": self._sqlite_count(connection, "filings"),
                "raw_filings": self._sqlite_count(connection, "raw_filings"),
                "prices": self._sqlite_count(connection, "prices"),
                "failures": self._sqlite_count(connection, "failures"),
            }

    def populate_prices_for_symbols_with_sqlite(
        self,
        tickers: list[str],
        *,
        group_name: str = "Custom Price History",
        start: date | datetime | str = "2010-01-01",
        end: date | datetime | str | None = None,
        interval: str = "1d",
        include_prepost: bool = False,
        sqlite_path: str | Path | None = None,
    ) -> tuple[pd.DataFrame, Path]:
        downloader = _import_price_history_downloader()()
        frames: list[pd.DataFrame] = []
        for ticker in [str(ticker).upper() for ticker in tickers]:
            frame = downloader.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                lookback_days=None,
                include_prepost=include_prepost,
            )
            if not frame.empty:
                frames.append(frame)
        prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        db_path = self.persist_prices_to_sqlite(prices, group_name=group_name, sqlite_path=sqlite_path)
        return prices, db_path

    def persist_prices_to_sqlite(
        self,
        prices: pd.DataFrame,
        *,
        group_name: str,
        sqlite_path: str | Path | None = None,
    ) -> Path:
        db_path = Path(sqlite_path) if sqlite_path is not None else self.sqlite_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as connection:
            try:
                connection.execute("DELETE FROM prices WHERE group_name = ?", (group_name,))
            except sqlite3.OperationalError:
                pass
            self._write_sqlite_frame(connection, "prices", prices, group_name)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_prices_group_ticker_date ON prices(group_name, ticker, date)"
            )
        return db_path

    def find_sqlite_prices(
        self,
        group_name: str,
        *,
        ticker: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        sqlite_path: str | Path | None = None,
    ) -> pd.DataFrame:
        db_path = Path(sqlite_path) if sqlite_path is not None else self.sqlite_path
        if not db_path.exists():
            return pd.DataFrame()

        query = "SELECT * FROM prices WHERE group_name = ?"
        params: list[object] = [group_name]
        if ticker:
            query += " AND UPPER(ticker) = ?"
            params.append(ticker.upper())
        if date_from:
            query += " AND date >= ?"
            params.append(date_from)
        if date_to:
            query += " AND date <= ?"
            params.append(date_to)
        query += " ORDER BY ticker ASC, date ASC"

        with sqlite3.connect(db_path) as connection:
            return pd.read_sql_query(query, connection, params=params)

    @staticmethod
    def _sqlite_count(connection: sqlite3.Connection, table_name: str) -> int:
        try:
            row = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row[0]) if row is not None else 0

    def _find_sqlite_filings(
        self,
        table_name: str,
        group_name: str,
        *,
        ticker: str | None,
        form: str | None,
        accession_no: str | None,
        sqlite_path: str | Path | None,
    ) -> pd.DataFrame:
        db_path = Path(sqlite_path) if sqlite_path is not None else self.sqlite_path
        if not db_path.exists():
            return pd.DataFrame()

        query = f"SELECT * FROM {table_name} WHERE group_name = ?"
        params: list[object] = [group_name]
        if ticker:
            query += " AND UPPER(ticker) = ?"
            params.append(ticker.upper())
        if form:
            query += " AND UPPER(form) = ?"
            params.append(form.upper())
        if accession_no:
            query += " AND accession_no = ?"
            params.append(accession_no)
        query += " ORDER BY filing_date DESC"

        with sqlite3.connect(db_path) as connection:
            return pd.read_sql_query(query, connection, params=params)

    @staticmethod
    def _ensure_sqlite_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS corpora (
                group_name TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL,
                tickers_count INTEGER NOT NULL,
                ticker_index_rows INTEGER NOT NULL,
                filing_rows INTEGER NOT NULL,
                raw_filing_rows INTEGER NOT NULL,
                failed_tickers INTEGER NOT NULL
            )
            """
        )

    @staticmethod
    def _delete_sqlite_group(connection: sqlite3.Connection, group_name: str) -> None:
        for table_name in ("tickers", "filings", "raw_filings", "failures"):
            try:
                connection.execute(f"DELETE FROM {table_name} WHERE group_name = ?", (group_name,))
            except sqlite3.OperationalError:
                continue
        connection.execute("DELETE FROM corpora WHERE group_name = ?", (group_name,))

    @staticmethod
    def _write_sqlite_frame(
        connection: sqlite3.Connection,
        table_name: str,
        frame: pd.DataFrame,
        group_name: str,
    ) -> None:
        if frame.empty:
            return
        sqlite_frame = frame.copy()
        sqlite_frame.insert(0, "group_name", group_name)
        sqlite_frame = _sqlite_compatible_frame(sqlite_frame)
        _ensure_sqlite_table_columns(connection, table_name, sqlite_frame)
        sqlite_frame.to_sql(table_name, connection, if_exists="append", index=False)

    @staticmethod
    def _failures_frame(failures: dict[str, list[str]]) -> pd.DataFrame:
        rows = [
            {"ticker": ticker, "stage": stage, "error": error}
            for ticker, errors in failures.items()
            for stage, error in (_split_failure(message) for message in errors)
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def _upsert_sqlite_corpus_metadata(connection: sqlite3.Connection, corpus: RawFilingCorpus) -> None:
        summary = corpus.summary()
        connection.execute(
            """
            INSERT INTO corpora (
                group_name,
                updated_at,
                tickers_count,
                ticker_index_rows,
                filing_rows,
                raw_filing_rows,
                failed_tickers
            )
            VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
            ON CONFLICT(group_name) DO UPDATE SET
                updated_at = excluded.updated_at,
                tickers_count = excluded.tickers_count,
                ticker_index_rows = excluded.ticker_index_rows,
                filing_rows = excluded.filing_rows,
                raw_filing_rows = excluded.raw_filing_rows,
                failed_tickers = excluded.failed_tickers
            """,
            (
                corpus.group_name,
                summary["tickers"],
                summary["ticker_index_rows"],
                summary["filing_rows"],
                summary["raw_filing_rows"],
                summary["failed_tickers"],
            ),
        )

    @staticmethod
    def _create_sqlite_indexes(connection: sqlite3.Connection) -> None:
        statements = [
            "CREATE INDEX IF NOT EXISTS idx_tickers_group_ticker ON tickers(group_name, ticker)",
            "CREATE INDEX IF NOT EXISTS idx_filings_group_ticker_date ON filings(group_name, ticker, filing_date)",
            "CREATE INDEX IF NOT EXISTS idx_filings_group_accession ON filings(group_name, accession_no)",
            "CREATE INDEX IF NOT EXISTS idx_raw_filings_group_ticker_date ON raw_filings(group_name, ticker, filing_date)",
            "CREATE INDEX IF NOT EXISTS idx_raw_filings_group_accession ON raw_filings(group_name, accession_no)",
            "CREATE INDEX IF NOT EXISTS idx_failures_group_ticker ON failures(group_name, ticker)",
        ]
        for statement in statements:
            try:
                connection.execute(statement)
            except sqlite3.OperationalError:
                continue


def _slugify(value: str) -> str:
    safe = "".join(character.lower() if character.isalnum() else "_" for character in value.strip())
    return safe.strip("_") or "raw_filing_corpus"


def _optional_existing_path(path: Path) -> Path | None:
    return path if path.exists() else None


def _split_failure(message: str) -> tuple[str, str]:
    stage, _, error = message.partition(": ")
    return stage or "unknown", error or message


def _sqlite_compatible_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a frame containing only scalar values sqlite can bind directly."""
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].map(_sqlite_scalar)
        elif pd.api.types.is_object_dtype(result[column]):
            result[column] = result[column].map(_sqlite_scalar)
    return result.where(pd.notnull(result), None)


def _sqlite_scalar(value):
    """Convert pandas/Python temporal scalars before sqlite binding."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _ensure_sqlite_table_columns(
    connection: sqlite3.Connection,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    existing_rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchall()
    if not existing_rows:
        return

    existing_columns = {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    for column in frame.columns:
        if column in existing_columns:
            continue
        sqlite_type = _sqlite_type_for_series(frame[column])
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {sqlite_type}")


def _sqlite_type_for_series(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    return "TEXT"


def _import_price_history_downloader():
    try:
        from price_fetcher import PriceHistoryDownloader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "price_fetcher is required for price-history downloads. "
            "Install the price-fetcher package or provide a local price_fetcher module."
        ) from exc
    return PriceHistoryDownloader

"""Raw filing corpus schema, storage, and pipeline helpers."""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .collector import MarketDataBundle, MarketDataCollector
from .filings import RAW_FILING_DATASET_COLUMNS, normalize_raw_filing_dataset

_SUPPORTED_CORPUS_FORMATS = {"parquet", "jsonl"}
_FORMAT_SUFFIXES = {".parquet": "parquet", ".jsonl": "jsonl"}
_RAW_FILING_SHARD_SIZE_ROWS = 100
_RAW_FILING_SHARD_TARGET_TEXT_BYTES = 1_500_000_000
RAW_FILING_INDEX_COLUMNS = (
    "ticker",
    "cik",
    "accession_no",
    "form",
    "filing_date",
    "period_end",
    "submission_text_length",
    "source_url",
    "downloaded_at",
    "raw_shard_path",
    "raw_shard_row_number",
)


@dataclass
class RawFilingCorpus:
    """Structured filing index plus the matching raw filing submission texts."""

    group_name: str
    tickers: list[str]
    ticker_index: pd.DataFrame = field(default_factory=pd.DataFrame)
    filings: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_filings: pd.DataFrame = field(default_factory=pd.DataFrame)
    failures: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ticker_index = self.ticker_index.reset_index(drop=True)
        self.filings = self.filings.reset_index(drop=True)
        self.raw_filings = normalize_raw_filing_dataset(self.raw_filings)

    @property
    def collection_name(self) -> str:
        return self.group_name

    @property
    def dataset_columns(self) -> tuple[str, ...]:
        return RAW_FILING_DATASET_COLUMNS

    def summary(self) -> dict[str, int]:
        return {
            "tickers": len(self.tickers),
            "ticker_index_rows": len(self.ticker_index),
            "filing_rows": len(self.filings),
            "raw_filing_rows": len(self.raw_filings),
            "failed_tickers": len(self.failures),
        }

    def save(
        self,
        output_path: str | Path,
        *,
        format: str = "parquet",
        include_supporting_data: bool = True,
    ) -> dict[str, Path]:
        return RawFilingCorpusWriter().write(
            self,
            output_path,
            format=format,
            include_supporting_data=include_supporting_data,
        )

    @classmethod
    def from_market_data_bundle(cls, bundle: MarketDataBundle) -> RawFilingCorpus:
        return cls(
            group_name=bundle.group_name,
            tickers=list(bundle.tickers),
            ticker_index=bundle.ticker_index.copy(),
            filings=bundle.filings.copy(),
            raw_filings=bundle.raw_filings.copy(),
            failures=dict(bundle.failures),
        )


class RawFilingCorpusWriter:
    """Persist a raw filing corpus with parquet shards for raw submission text."""

    def write(
        self,
        corpus: RawFilingCorpus,
        output_path: str | Path,
        *,
        format: str = "parquet",
        include_supporting_data: bool = True,
    ) -> dict[str, Path]:
        output_dir, stem, dataset_path, dataset_format = _resolve_corpus_output(
            output_path,
            group_name=corpus.group_name,
            format=format,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_index_path, raw_shards_dir = self._write_raw_filing_shards(
            corpus.raw_filings,
            output_dir=output_dir,
            stem=stem,
        )
        written = {
            "raw_filings": raw_index_path,
            "raw_filing_index": raw_index_path,
            "raw_filing_shards_dir": raw_shards_dir,
        }

        if include_supporting_data:
            if not corpus.filings.empty:
                written["filings"] = self._write_frame(
                    corpus.filings,
                    output_dir / f"{stem}_filings.{_extension_for_format(dataset_format)}",
                    dataset_format,
                )
            if not corpus.ticker_index.empty:
                written["tickers"] = self._write_frame(
                    corpus.ticker_index,
                    output_dir / f"{stem}_tickers.{_extension_for_format(dataset_format)}",
                    dataset_format,
                )
            if corpus.failures:
                written["failures"] = self._write_failures(
                    corpus.failures,
                    output_dir / f"{stem}_failures.jsonl",
                )

        written["manifest"] = self._write_manifest(
            corpus,
            output_dir / f"{stem}_manifest.json",
            dataset_format=dataset_format,
            include_supporting_data=include_supporting_data,
        )
        return written

    def _write_frame(self, frame: pd.DataFrame, path: Path, format: str) -> Path:
        if format == "parquet":
            frame.to_parquet(path, index=False)
        elif format == "jsonl":
            frame.to_json(path, orient="records", lines=True, force_ascii=False)
        else:
            raise ValueError(f"Unsupported corpus format: {format}")
        return path

    def _write_failures(self, failures: dict[str, list[str]], path: Path) -> Path:
        rows = [
            {"ticker": ticker, "stage": stage, "error": error}
            for ticker, errors in failures.items()
            for stage, error in (_split_failure(message) for message in errors)
        ]
        pd.DataFrame(rows).to_json(path, orient="records", lines=True, force_ascii=False)
        return path

    def _write_manifest(
        self,
        corpus: RawFilingCorpus,
        path: Path,
        *,
        dataset_format: str,
        include_supporting_data: bool,
    ) -> Path:
        payload = {
            "group_name": corpus.group_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_format": dataset_format,
            "dataset_columns": list(corpus.dataset_columns),
            "raw_filing_index_columns": list(RAW_FILING_INDEX_COLUMNS),
            "raw_filing_storage": "parquet_shards",
            "include_supporting_data": include_supporting_data,
            "summary": corpus.summary(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _write_raw_filing_shards(
        self,
        frame: pd.DataFrame,
        *,
        output_dir: Path,
        stem: str,
    ) -> tuple[Path, Path]:
        shards_dir = output_dir / f"{stem}_raw_filing_shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        index_path = output_dir / f"{stem}_raw_filing_index.jsonl"

        normalized = normalize_raw_filing_dataset(frame)
        if normalized.empty:
            pd.DataFrame(columns=RAW_FILING_INDEX_COLUMNS).to_json(
                index_path,
                orient="records",
                lines=True,
                force_ascii=False,
            )
            return index_path, shards_dir

        index_parts: list[pd.DataFrame] = []
        for shard_number, shard in enumerate(self._iter_raw_filing_shards(normalized), start=1):
            shard_name = f"{stem}_raw_filings_{shard_number:05d}.parquet"
            shard_path = shards_dir / shard_name
            shard.to_parquet(shard_path, index=False)

            shard_index = shard.drop(columns=["submission_text"]).copy()
            shard_index["raw_shard_path"] = str(Path(shards_dir.name) / shard_name)
            shard_index["raw_shard_row_number"] = range(len(shard_index))
            index_parts.append(shard_index)

        index_frame = pd.concat(index_parts, ignore_index=True) if index_parts else pd.DataFrame()
        for column in RAW_FILING_INDEX_COLUMNS:
            if column not in index_frame.columns:
                index_frame[column] = pd.NA
        index_frame = index_frame.loc[:, RAW_FILING_INDEX_COLUMNS]
        index_frame.to_json(index_path, orient="records", lines=True, force_ascii=False)
        return index_path, shards_dir

    def _iter_raw_filing_shards(self, frame: pd.DataFrame) -> Iterable[pd.DataFrame]:
        start = 0
        frame_length = len(frame)
        while start < frame_length:
            end = start
            rows_in_shard = 0
            text_bytes_in_shard = 0

            while end < frame_length and rows_in_shard < _RAW_FILING_SHARD_SIZE_ROWS:
                row = frame.iloc[end]
                text_length = _coerce_submission_text_length(row.get("submission_text_length"))
                would_exceed_target = rows_in_shard > 0 and (
                    text_bytes_in_shard + text_length > _RAW_FILING_SHARD_TARGET_TEXT_BYTES
                )
                if would_exceed_target:
                    break

                text_bytes_in_shard += text_length
                rows_in_shard += 1
                end += 1

            if end == start:
                end += 1

            yield frame.iloc[start:end].reset_index(drop=True)
            start = end


class RawFilingCorpusBuilder:
    """Build and optionally persist raw filing corpora."""

    def __init__(
        self,
        identity: str | None = None,
        *,
        collector: MarketDataCollector | None = None,
    ) -> None:
        if collector is None:
            collector = MarketDataCollector(identity=identity)
        self.collector = collector

    def build_for_symbols(
        self,
        tickers: Iterable[str],
        *,
        group_name: str = "Custom Raw Filing Corpus",
        filing_options: dict | None = None,
        raw_filing_options: dict | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> RawFilingCorpus:
        bundle = self.collector.collect_for_symbols(
            tickers,
            group_name=group_name,
            include_filings=True,
            include_raw_filings=True,
            include_price_history=False,
            filing_options=dict(filing_options or {}),
            raw_filing_options=dict(raw_filing_options or {}),
            on_progress=on_progress,
        )
        return RawFilingCorpus.from_market_data_bundle(bundle)

    def build_public_market(
        self,
        *,
        force_refresh: bool = False,
        limit: int | None = None,
        filing_options: dict | None = None,
        raw_filing_options: dict | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> RawFilingCorpus:
        bundle = self.collector.collect_public_market(
            force_refresh=force_refresh,
            limit=limit,
            include_filings=True,
            include_raw_filings=True,
            include_price_history=False,
            filing_options=dict(filing_options or {}),
            raw_filing_options=dict(raw_filing_options or {}),
            on_progress=on_progress,
        )
        return RawFilingCorpus.from_market_data_bundle(bundle)

    def build_and_persist_for_symbols(
        self,
        tickers: Iterable[str],
        output_path: str | Path,
        *,
        group_name: str = "Custom Raw Filing Corpus",
        format: str = "parquet",
        include_supporting_data: bool = True,
        filing_options: dict | None = None,
        raw_filing_options: dict | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> tuple[RawFilingCorpus, dict[str, Path]]:
        corpus = self.build_for_symbols(
            tickers,
            group_name=group_name,
            filing_options=filing_options,
            raw_filing_options=raw_filing_options,
            on_progress=on_progress,
        )
        written = corpus.save(
            output_path,
            format=format,
            include_supporting_data=include_supporting_data,
        )
        return corpus, written

    def build_and_persist_public_market(
        self,
        output_path: str | Path,
        *,
        force_refresh: bool = False,
        limit: int | None = None,
        format: str = "parquet",
        include_supporting_data: bool = True,
        filing_options: dict | None = None,
        raw_filing_options: dict | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> tuple[RawFilingCorpus, dict[str, Path]]:
        corpus = self.build_public_market(
            force_refresh=force_refresh,
            limit=limit,
            filing_options=filing_options,
            raw_filing_options=raw_filing_options,
            on_progress=on_progress,
        )
        written = corpus.save(
            output_path,
            format=format,
            include_supporting_data=include_supporting_data,
        )
        return corpus, written


def _resolve_corpus_output(
    output_path: str | Path,
    *,
    group_name: str,
    format: str,
) -> tuple[Path, str, Path, str]:
    path = Path(output_path)
    normalized_format = format.strip().lower()
    suffix_format = _FORMAT_SUFFIXES.get(path.suffix.lower())

    if normalized_format == "auto":
        normalized_format = suffix_format or "parquet"
    elif normalized_format not in _SUPPORTED_CORPUS_FORMATS:
        raise ValueError(
            f"Unsupported corpus format: {format}. Expected one of {sorted(_SUPPORTED_CORPUS_FORMATS)}."
        )

    if suffix_format is not None and suffix_format != normalized_format:
        raise ValueError(
            f"Output suffix {path.suffix!r} does not match requested format {normalized_format!r}."
        )

    if suffix_format is not None:
        output_dir = path.parent
        stem = path.stem
        dataset_path = path
    else:
        output_dir = path
        stem = _slugify(group_name)
        dataset_path = output_dir / f"{stem}_raw_filings.{_extension_for_format(normalized_format)}"

    return output_dir, stem, dataset_path, normalized_format


def _extension_for_format(format: str) -> str:
    if format not in _SUPPORTED_CORPUS_FORMATS:
        raise ValueError(f"Unsupported corpus format: {format}")
    return format


def _slugify(value: str) -> str:
    safe = "".join(character.lower() if character.isalnum() else "_" for character in value.strip())
    return safe.strip("_") or "raw_filing_corpus"


def _split_failure(message: str) -> tuple[str, str]:
    stage, _, error = message.partition(": ")
    return stage or "unknown", error or message


def _coerce_submission_text_length(value: object) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return 0
    return max(numeric, 0)

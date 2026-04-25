"""High-level orchestration for price history, filing metrics, and raw filings."""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .filings import FilingMetricsDownloader, RawFilingDownloader
from .networking import RequestRateLimiter
from .tickers import PublicTickerDirectory

if TYPE_CHECKING:
    from price_fetcher import PriceHistoryDownloader


@dataclass
class MarketDataBundle:
    """Combined ticker index, filing metrics, raw filings, and price history payload."""

    group_name: str
    tickers: list[str]
    ticker_index: pd.DataFrame = field(default_factory=pd.DataFrame)
    filings: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_filings: pd.DataFrame = field(default_factory=pd.DataFrame)
    price_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    failures: dict[str, list[str]] = field(default_factory=dict)

    @property
    def collection_name(self) -> str:
        return self.group_name

    def summary(self) -> dict[str, int]:
        return {
            "tickers": len(self.tickers),
            "ticker_index_rows": len(self.ticker_index),
            "filing_rows": len(self.filings),
            "raw_filing_rows": len(self.raw_filings),
            "price_history_rows": len(self.price_history),
            "failed_tickers": len(self.failures),
        }

    def save_csv(self, output_dir: str | Path) -> dict[str, Path]:
        return MarketDataCsvWriter().write(self, output_dir)


class MarketDataCsvWriter:
    """Persist a :class:`MarketDataBundle` to CSV files."""

    def write(self, bundle: MarketDataBundle, output_dir: str | Path) -> dict[str, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        slug = _slugify(bundle.group_name)
        written: dict[str, Path] = {}

        frames = {
            "tickers": bundle.ticker_index,
            "filings": bundle.filings,
            "raw_filings": bundle.raw_filings,
            "price_history": bundle.price_history,
        }
        for label, frame in frames.items():
            if frame.empty:
                continue
            path = output_path / f"{slug}_{label}.csv"
            frame.to_csv(path, index=False)
            written[label] = path

        if bundle.failures:
            failure_rows = [
                {"ticker": ticker, "stage": stage, "error": error}
                for ticker, errors in bundle.failures.items()
                for stage, error in (_split_failure(message) for message in errors)
            ]
            failures_path = output_path / f"{slug}_failures.csv"
            pd.DataFrame(failure_rows).to_csv(failures_path, index=False)
            written["failures"] = failures_path

        return written


class MarketDataCollector:
    """Compose ticker-directory, filing, raw-filing, and price-history services."""

    def __init__(
        self,
        identity: str | None = None,
        *,
        sec_rate_limiter: RequestRateLimiter | None = None,
        yahoo_rate_limiter: RequestRateLimiter | None = None,
        rate_limiter: RequestRateLimiter | None = None,
        ticker_directory: PublicTickerDirectory | None = None,
        filings_downloader: FilingMetricsDownloader | None = None,
        raw_filing_downloader: RawFilingDownloader | None = None,
        price_history_downloader: PriceHistoryDownloader | None = None,
        cik_resolver: Callable[[str], str | None] | None = None,
    ) -> None:
        self.identity = identity
        self.sec_rate_limiter = (
            sec_rate_limiter
            or rate_limiter
            or RequestRateLimiter(max_requests=10, period_seconds=1.0)
        )
        self.yahoo_rate_limiter = yahoo_rate_limiter or RequestRateLimiter(max_requests=2, period_seconds=1.0)
        self.rate_limiter = self.sec_rate_limiter
        self.ticker_directory = ticker_directory
        self.filings_downloader = filings_downloader
        self.raw_filing_downloader = raw_filing_downloader
        self.price_history_downloader = price_history_downloader
        self.cik_resolver = cik_resolver
        self._ticker_directory_lock = threading.Lock()
        self._filings_downloader_lock = threading.Lock()
        self._raw_filing_downloader_lock = threading.Lock()

    def load_ticker_directory(self, force_refresh: bool = False) -> pd.DataFrame:
        return self.get_ticker_directory().load(force_refresh=force_refresh)

    def list_public_symbols(self, force_refresh: bool = False, limit: int | None = None) -> list[str]:
        return self.get_ticker_directory().symbols(force_refresh=force_refresh, limit=limit)

    def collect_for_symbols(
        self,
        tickers: Iterable[str],
        *,
        group_name: str = "Custom Ticker Set",
        include_filings: bool = True,
        include_raw_filings: bool = False,
        include_price_history: bool = True,
        filing_options: dict | None = None,
        raw_filing_options: dict | None = None,
        price_history_options: dict | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> MarketDataBundle:
        normalized_tickers = [str(ticker).upper() for ticker in tickers]
        ticker_index = self._load_optional_ticker_directory(normalized_tickers)
        return self._collect_collection(
            normalized_tickers,
            group_name=group_name,
            ticker_index=ticker_index,
            include_filings=include_filings,
            include_raw_filings=include_raw_filings,
            include_price_history=include_price_history,
            filing_options=dict(filing_options or {}),
            raw_filing_options=dict(raw_filing_options or {}),
            price_history_options=dict(price_history_options or {}),
            on_progress=on_progress,
        )

    def collect_public_market(
        self,
        *,
        force_refresh: bool = False,
        limit: int | None = None,
        include_filings: bool = True,
        include_raw_filings: bool = False,
        include_price_history: bool = True,
        filing_options: dict | None = None,
        raw_filing_options: dict | None = None,
        price_history_options: dict | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> MarketDataBundle:
        ticker_index = self.load_ticker_directory(force_refresh=force_refresh)
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive when provided.")
            ticker_index = ticker_index.head(limit).reset_index(drop=True)
        tickers = ticker_index["ticker"].dropna().astype(str).str.upper().tolist()
        return self._collect_collection(
            tickers,
            group_name="All Public Tickers",
            ticker_index=ticker_index,
            include_filings=include_filings,
            include_raw_filings=include_raw_filings,
            include_price_history=include_price_history,
            filing_options=dict(filing_options or {}),
            raw_filing_options=dict(raw_filing_options or {}),
            price_history_options=dict(price_history_options or {}),
            on_progress=on_progress,
        )

    def _collect_collection(
        self,
        tickers: list[str],
        *,
        group_name: str,
        ticker_index: pd.DataFrame,
        include_filings: bool,
        include_raw_filings: bool,
        include_price_history: bool,
        filing_options: dict,
        raw_filing_options: dict,
        price_history_options: dict,
        on_progress: Callable[[str, int, int], None] | None,
    ) -> MarketDataBundle:
        if not include_filings and not include_raw_filings and not include_price_history:
            raise ValueError(
                "At least one of include_filings, include_raw_filings, or include_price_history must be True."
            )

        filing_frames: list[pd.DataFrame] = []
        raw_filing_frames: list[pd.DataFrame] = []
        price_history_frames: list[pd.DataFrame] = []
        failures: dict[str, list[str]] = {}
        total = len(tickers)

        for index, ticker in enumerate(tickers, start=1):
            filing_frame, raw_filing_frame, history_frame, ticker_failures = self._download_ticker(
                ticker,
                include_filings=include_filings,
                include_raw_filings=include_raw_filings,
                include_price_history=include_price_history,
                filing_options=filing_options,
                raw_filing_options=raw_filing_options,
                price_history_options=price_history_options,
            )
            if not filing_frame.empty:
                filing_frames.append(filing_frame)
            if not raw_filing_frame.empty:
                raw_filing_frames.append(raw_filing_frame)
            if not history_frame.empty:
                price_history_frames.append(history_frame)
            if ticker_failures:
                failures[ticker] = ticker_failures
            if on_progress is not None:
                on_progress(ticker, index, total)

        return MarketDataBundle(
            group_name=group_name,
            tickers=tickers,
            ticker_index=ticker_index,
            filings=_concat_frames(filing_frames, sort_by="filing_date", ascending=False),
            raw_filings=_concat_frames(raw_filing_frames, sort_by="filing_date", ascending=False),
            price_history=_concat_frames(price_history_frames, sort_by="date", ascending=False),
            failures=failures,
        )

    def _download_ticker(
        self,
        ticker: str,
        *,
        include_filings: bool,
        include_raw_filings: bool,
        include_price_history: bool,
        filing_options: dict,
        raw_filing_options: dict,
        price_history_options: dict,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
        filing_frame = pd.DataFrame()
        raw_filing_frame = pd.DataFrame()
        history_frame = pd.DataFrame()
        errors: list[str] = []

        if include_filings or include_raw_filings:
            try:
                filing_frame = self.get_filings_downloader().download_for_ticker(ticker, **filing_options)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"filings: {exc}")

        if include_raw_filings:
            try:
                raw_filing_frame = self.get_raw_filing_downloader().download_for_ticker(
                    ticker,
                    filings_frame=filing_frame if not filing_frame.empty else None,
                    **raw_filing_options,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"raw_filings: {exc}")

        if include_price_history:
            try:
                history_frame = self.get_price_history_downloader().download(ticker, **price_history_options)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"price_history: {exc}")

        return filing_frame, raw_filing_frame, history_frame, errors

    def get_ticker_directory(self) -> PublicTickerDirectory:
        if self.ticker_directory is None:
            with self._ticker_directory_lock:
                if self.ticker_directory is None:
                    if self.identity is None:
                        raise ValueError("identity is required to load the public ticker directory.")
                    self.ticker_directory = PublicTickerDirectory(self.identity)
        return self.ticker_directory

    def get_filings_downloader(self) -> FilingMetricsDownloader:
        if self.filings_downloader is None:
            with self._filings_downloader_lock:
                if self.filings_downloader is None:
                    if self.identity is None:
                        raise ValueError("identity is required to download SEC filing metrics.")
                    resolver = self.cik_resolver or self._build_cik_resolver()
                    self.filings_downloader = FilingMetricsDownloader(
                        identity=self.identity,
                        rate_limiter=self.sec_rate_limiter,
                        cik_resolver=resolver,
                    )
        return self.filings_downloader

    def get_raw_filing_downloader(self) -> RawFilingDownloader:
        if self.raw_filing_downloader is None:
            with self._raw_filing_downloader_lock:
                if self.raw_filing_downloader is None:
                    if self.identity is None:
                        raise ValueError("identity is required to download raw SEC filings.")
                    resolver = self.cik_resolver or self._build_cik_resolver()
                    self.raw_filing_downloader = RawFilingDownloader(
                        identity=self.identity,
                        rate_limiter=self.sec_rate_limiter,
                        cik_resolver=resolver,
                        filings_downloader=self.get_filings_downloader(),
                    )
        return self.raw_filing_downloader

    def get_price_history_downloader(self):
        if self.price_history_downloader is None:
            downloader_cls = _import_price_history_downloader()
            self.price_history_downloader = downloader_cls(rate_limiter=self.yahoo_rate_limiter)
        return self.price_history_downloader

    def _build_cik_resolver(self) -> Callable[[str], str | None]:
        directory = self.get_ticker_directory()
        lookup = directory.cik_lookup()
        return lambda ticker: lookup.get(ticker.upper())

    def _load_optional_ticker_directory(self, tickers: list[str]) -> pd.DataFrame:
        if self.ticker_directory is None and self.identity is None:
            return pd.DataFrame(columns=["cik_str", "ticker", "title", "search_label"])
        ticker_index = self.load_ticker_directory(force_refresh=False)
        return ticker_index[ticker_index["ticker"].isin(tickers)].reset_index(drop=True)


MarketDataManager = MarketDataCollector
MarketDataDownload = MarketDataBundle


def _concat_frames(frames: list[pd.DataFrame], *, sort_by: str, ascending: bool) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if sort_by in combined.columns:
        combined = combined.sort_values(sort_by, ascending=ascending, kind="stable")
    return combined.reset_index(drop=True)


def _slugify(value: str) -> str:
    safe = "".join(character.lower() if character.isalnum() else "_" for character in value.strip())
    return safe.strip("_") or "market_data"


def _split_failure(message: str) -> tuple[str, str]:
    stage, _, error = message.partition(": ")
    return stage or "unknown", error or message


def _import_price_history_downloader():
    try:
        from price_fetcher import PriceHistoryDownloader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "price_fetcher is required for price-history collection. "
            "Install the price-fetcher package or call the collector with include_price_history=False."
        ) from exc
    return PriceHistoryDownloader

"""Load and search the SEC public ticker directory."""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .networking import urlopen


_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_CACHE_TTL_SECONDS = 24 * 3600
_DEFAULT_CACHE_PATH = "company_tickers_cache.json"


class SecTickerDirectoryClient:
    """Read the SEC ticker directory with a simple on-disk cache."""

    def __init__(
        self,
        identity: str,
        cache_path: str | Path = _DEFAULT_CACHE_PATH,
        cache_ttl_seconds: int = _CACHE_TTL_SECONDS,
    ) -> None:
        self.identity = identity
        self.cache_path = Path(cache_path)
        self.cache_ttl_seconds = cache_ttl_seconds

    def load_payload(self, force_refresh: bool = False) -> dict:
        payload = None if force_refresh else self._load_cached_payload()
        if payload is None:
            payload = self._download_payload()
            self._write_cached_payload(payload)
        return payload

    def refresh_payload(self) -> dict:
        return self.load_payload(force_refresh=True)

    def _download_payload(self) -> dict:
        request = urllib.request.Request(
            _SEC_TICKERS_URL,
            headers={"User-Agent": self.identity},
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def _load_cached_payload(self) -> dict | None:
        if not self.cache_path.exists():
            return None
        try:
            payload = json.loads(self.cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if time.time() - payload.get("ts", 0) < self.cache_ttl_seconds:
            return payload.get("data")
        return None

    def _write_cached_payload(self, payload: dict) -> None:
        cache_payload = {"ts": time.time(), "data": payload}
        try:
            self.cache_path.write_text(json.dumps(cache_payload))
        except OSError:
            pass


class TickerDirectoryFrameBuilder:
    """Convert the SEC ticker payload into a normalized DataFrame."""

    def build(self, payload: dict) -> pd.DataFrame:
        rows = [
            {
                "cik_str": str(entry["cik_str"]).zfill(10),
                "ticker": str(entry["ticker"]).upper(),
                "title": str(entry["title"]),
            }
            for entry in payload.values()
        ]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["cik_str", "ticker", "title", "search_label"])
        frame["search_label"] = frame["ticker"] + " - " + frame["title"]
        return frame.sort_values("ticker", kind="stable").reset_index(drop=True)


class TickerDirectorySearch:
    """Search helper for the public ticker directory DataFrame."""

    def search(self, frame: pd.DataFrame, query: str, max_results: int = 50) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        if not query:
            return frame.head(max_results).reset_index(drop=True)

        query_upper = query.strip().upper()
        ticker_match = frame["ticker"].str.startswith(query_upper, na=False)
        title_match = frame["title"].str.contains(query, case=False, na=False, regex=False)
        return frame[ticker_match | title_match].head(max_results).reset_index(drop=True)


class PublicTickerDirectory:
    """High-level interface for the SEC public ticker directory."""

    def __init__(
        self,
        identity: str | None = None,
        *,
        client: SecTickerDirectoryClient | None = None,
        frame_builder: TickerDirectoryFrameBuilder | None = None,
        search_engine: TickerDirectorySearch | None = None,
        cache_path: str | Path = _DEFAULT_CACHE_PATH,
    ) -> None:
        if client is None:
            if identity is None:
                raise ValueError("identity is required when client is not provided.")
            client = SecTickerDirectoryClient(identity=identity, cache_path=cache_path)
        self.client = client
        self.frame_builder = frame_builder or TickerDirectoryFrameBuilder()
        self.search_engine = search_engine or TickerDirectorySearch()
        self._frame_cache: pd.DataFrame | None = None

    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        if self._frame_cache is not None and not force_refresh:
            return self._frame_cache.copy()
        payload = self.client.load_payload(force_refresh=force_refresh)
        self._frame_cache = self.frame_builder.build(payload)
        return self._frame_cache.copy()

    def refresh(self) -> pd.DataFrame:
        return self.load(force_refresh=True)

    def search(self, query: str, max_results: int = 50) -> pd.DataFrame:
        return self.search_engine.search(self.load(), query=query, max_results=max_results)

    def symbols(self, force_refresh: bool = False, limit: int | None = None) -> list[str]:
        frame = self.load(force_refresh=force_refresh)
        symbols = frame["ticker"].dropna().astype(str).str.upper().tolist()
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive when provided.")
            symbols = symbols[:limit]
        return symbols

    def cik_lookup(self, force_refresh: bool = False) -> dict[str, str]:
        frame = self.load(force_refresh=force_refresh)
        return {
            str(row["ticker"]).upper(): str(row["cik_str"]).zfill(10)
            for _, row in frame.iterrows()
            if row.get("ticker") and row.get("cik_str")
        }

    # Backwards-compatible alias
    def fetch(self, force_refresh: bool = False) -> pd.DataFrame:
        return self.load(force_refresh=force_refresh)


class AllTickersFetcher(PublicTickerDirectory):
    """Compatibility wrapper for the previous public ticker interface."""

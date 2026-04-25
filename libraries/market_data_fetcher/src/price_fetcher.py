from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd

from market_research_base import RequestRateLimiter, urlopen


_YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_DEFAULT_USER_AGENT = "Market Research Bot/1.0"


@dataclass(frozen=True)
class PriceHistoryRequest:
    ticker: str
    interval: str = "1d"
    start: date | datetime | str | None = None
    end: date | datetime | str | None = None
    lookback_days: int | None = 365
    include_prepost: bool = False


class PriceHistoryPeriodResolver:
    """Translate a price-history request into Yahoo chart timestamps."""

    def resolve(self, request: PriceHistoryRequest) -> tuple[int, int]:
        if request.lookback_days is not None and (request.start is not None or request.end is not None):
            raise ValueError("Use either lookback_days or start/end, not both.")

        if request.lookback_days is not None:
            if request.lookback_days <= 0:
                raise ValueError("lookback_days must be positive.")
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=request.lookback_days)
            return int(start_dt.timestamp()), int(end_dt.timestamp())

        start_dt = _to_utc_datetime(request.start, default_to_day_start=True)
        end_dt = _to_utc_datetime(request.end, default_to_day_start=False)

        if start_dt is None:
            raise ValueError("start is required when lookback_days is not provided.")
        if end_dt is None:
            end_dt = datetime.now(timezone.utc)
        if start_dt >= end_dt:
            raise ValueError("start must be earlier than end.")

        return int(start_dt.timestamp()), int(end_dt.timestamp())


class YahooChartClient:
    """Download raw price-history payloads from Yahoo Finance."""

    def __init__(
        self,
        user_agent: str = _DEFAULT_USER_AGENT,
        rate_limiter: RequestRateLimiter | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.rate_limiter = rate_limiter or RequestRateLimiter(max_requests=2, period_seconds=1.0)

    def download_chart(self, request: PriceHistoryRequest, period: tuple[int, int]) -> dict:
        period_start, period_end = period
        params = urllib.parse.urlencode(
            {
                "interval": request.interval,
                "period1": str(period_start),
                "period2": str(period_end),
                "includePrePost": str(request.include_prepost).lower(),
                "events": "div,splits,capitalGains",
                "includeAdjustedClose": "true",
            }
        )
        url = _YAHOO_CHART_URL.format(ticker=urllib.parse.quote(request.ticker)) + f"?{params}"
        http_request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})

        self.rate_limiter.wait()
        with urlopen(http_request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))


class YahooChartResultParser:
    """Extract the chart result block from a Yahoo payload."""

    def parse(self, payload: dict) -> dict:
        chart = payload.get("chart") or {}
        error = chart.get("error")
        if error:
            description = error.get("description") or "Unknown chart API error"
            raise ValueError(description)

        results = chart.get("result") or []
        if not results:
            raise ValueError("No price history returned for ticker.")
        return results[0]


class PriceHistoryFrameBuilder:
    """Convert a Yahoo chart result into the project DataFrame shape."""

    def build(self, ticker: str, result: dict) -> pd.DataFrame:
        timestamps = result.get("timestamp") or []
        quote_rows = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        adjclose_rows = ((result.get("indicators") or {}).get("adjclose") or [{}])[0]

        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": quote_rows.get("open"),
                "high": quote_rows.get("high"),
                "low": quote_rows.get("low"),
                "close": quote_rows.get("close"),
                "adj_close": adjclose_rows.get("adjclose"),
                "volume": quote_rows.get("volume"),
            }
        )
        if frame.empty:
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"])

        frame.insert(0, "ticker", ticker.upper())
        frame["date"] = frame["timestamp"].dt.date
        numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
        frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=numeric_columns, how="all")
        frame = frame.drop(columns=["timestamp"])
        return frame.sort_values("date", kind="stable").reset_index(drop=True)


class PriceHistoryDownloader:
    """Download normalized OHLCV price history for a ticker."""

    def __init__(
        self,
        *,
        period_resolver: PriceHistoryPeriodResolver | None = None,
        chart_client: YahooChartClient | None = None,
        result_parser: YahooChartResultParser | None = None,
        frame_builder: PriceHistoryFrameBuilder | None = None,
        user_agent: str = _DEFAULT_USER_AGENT,
        rate_limiter: RequestRateLimiter | None = None,
    ) -> None:
        limiter = rate_limiter or RequestRateLimiter(max_requests=2, period_seconds=1.0)
        self.period_resolver = period_resolver or PriceHistoryPeriodResolver()
        self.chart_client = chart_client or YahooChartClient(user_agent=user_agent, rate_limiter=limiter)
        self.result_parser = result_parser or YahooChartResultParser()
        self.frame_builder = frame_builder or PriceHistoryFrameBuilder()

    def download(
        self,
        ticker: str,
        *,
        interval: str = "1d",
        start: date | datetime | str | None = None,
        end: date | datetime | str | None = None,
        lookback_days: int | None = 365,
        include_prepost: bool = False,
    ) -> pd.DataFrame:
        request = PriceHistoryRequest(
            ticker=ticker.upper(),
            interval=interval,
            start=start,
            end=end,
            lookback_days=lookback_days,
            include_prepost=include_prepost,
        )
        return self.download_request(request)

    def download_request(self, request: PriceHistoryRequest) -> pd.DataFrame:
        period = self.period_resolver.resolve(request)
        payload = self.chart_client.download_chart(request, period)
        result = self.result_parser.parse(payload)
        return self.frame_builder.build(request.ticker, result)

    def fetch(self, ticker: str, **kwargs) -> pd.DataFrame:
        return self.download(ticker, **kwargs)

    def fetch_request(self, request: PriceHistoryRequest) -> pd.DataFrame:
        return self.download_request(request)


class PriceHistoryFetcher(PriceHistoryDownloader):
    """Compatibility wrapper for the previous price-history interface."""


def _to_utc_datetime(
    value: date | datetime | str | None,
    *,
    default_to_day_start: bool,
) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, str):
        value = datetime.fromisoformat(value)

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt_time = time.min if default_to_day_start else time.max
        dt = datetime.combine(value, dt_time)
    else:
        raise TypeError("Date values must be date, datetime, ISO-8601 string, or None.")

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

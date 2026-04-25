"""Reusable market-data fetching package for price history and SEC filings."""

from .collector import (
    MarketDataBundle,
    MarketDataCollector,
    MarketDataCsvWriter,
    MarketDataDownload,
    MarketDataManager,
)
from .corpus import RAW_FILING_INDEX_COLUMNS, RawFilingCorpus, RawFilingCorpusBuilder, RawFilingCorpusWriter
from .database_operator import DatabaseOperator, StoredCorpus
from .filings import (
    CompanyFactsParser,
    DEI_CONCEPT_MAP,
    EdgarCompanyFactsClient,
    FilingMetricsDownloader,
    GAAP_CONCEPT_MAP,
    RAW_FILING_DATASET_COLUMNS,
    RawFilingDownloader,
    RawFilingSubmissionClient,
    TickerFinancialFetcher,
    normalize_raw_filing_dataset,
)
from .networking import RequestRateLimiter, create_ssl_context, urlopen
from .tickers import (
    AllTickersFetcher,
    PublicTickerDirectory,
    SecTickerDirectoryClient,
    TickerDirectoryFrameBuilder,
    TickerDirectorySearch,
)

__all__ = [
    "AllTickersFetcher",
    "CompanyFactsParser",
    "DatabaseOperator",
    "DEI_CONCEPT_MAP",
    "EdgarCompanyFactsClient",
    "FilingMetricsDownloader",
    "GAAP_CONCEPT_MAP",
    "MarketDataBundle",
    "MarketDataCollector",
    "MarketDataCsvWriter",
    "MarketDataDownload",
    "MarketDataManager",
    "PublicTickerDirectory",
    "RAW_FILING_DATASET_COLUMNS",
    "RAW_FILING_INDEX_COLUMNS",
    "RequestRateLimiter",
    "RawFilingCorpus",
    "RawFilingCorpusBuilder",
    "RawFilingCorpusWriter",
    "RawFilingDownloader",
    "RawFilingSubmissionClient",
    "SecTickerDirectoryClient",
    "StoredCorpus",
    "TickerDirectoryFrameBuilder",
    "TickerDirectorySearch",
    "TickerFinancialFetcher",
    "create_ssl_context",
    "normalize_raw_filing_dataset",
    "urlopen",
]

try:
    from price_fetcher import (
        PriceHistoryDownloader,
        PriceHistoryFetcher,
        PriceHistoryFrameBuilder,
        PriceHistoryPeriodResolver,
        PriceHistoryRequest,
        YahooChartClient,
        YahooChartResultParser,
    )
except ModuleNotFoundError:
    pass
else:
    __all__.extend(
        [
            "PriceHistoryDownloader",
            "PriceHistoryFetcher",
            "PriceHistoryFrameBuilder",
            "PriceHistoryPeriodResolver",
            "PriceHistoryRequest",
            "YahooChartClient",
            "YahooChartResultParser",
        ]
    )

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = REPO_ROOT / "libraries" / "market_data_fetcher" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from market_data_fetcher import DatabaseOperator, RawFilingCorpus, RawFilingCorpusWriter
from price_fetcher import PriceHistoryFrameBuilder


class FakeCorpusBuilder:
    def build_for_symbols(self, tickers, *, group_name, filing_options=None, raw_filing_options=None):
        tickers = [str(ticker).upper() for ticker in tickers]
        ticker_index = pd.DataFrame(
            [
                {"ticker": ticker, "cik_str": f"{index + 1:010d}", "title": f"{ticker} Corp", "search_label": ticker}
                for index, ticker in enumerate(tickers)
            ]
        )
        filings = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "form": "10-Q",
                    "filing_date": "2026-01-30",
                    "accession_no": "0000320193-26-000006",
                    "revenue": 143756000000,
                },
                {
                    "ticker": "MSFT",
                    "form": "10-Q",
                    "filing_date": "2026-01-28",
                    "accession_no": "0001193125-26-027207",
                    "revenue": 158946000000,
                },
            ]
        )
        raw_filings = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "cik": "0000320193",
                    "accession_no": "0000320193-26-000006",
                    "form": "10-Q",
                    "filing_date": "2026-01-30",
                    "period_end": "2025-12-27",
                    "submission_text": "apple filing text",
                    "submission_text_length": 17,
                    "source_url": "https://sec.example/aapl.txt",
                    "downloaded_at": "2026-04-23T00:00:00+00:00",
                },
                {
                    "ticker": "MSFT",
                    "cik": "0000789019",
                    "accession_no": "0001193125-26-027207",
                    "form": "10-Q",
                    "filing_date": "2026-01-28",
                    "period_end": "2025-12-31",
                    "submission_text": "microsoft filing text",
                    "submission_text_length": 21,
                    "source_url": "https://sec.example/msft.txt",
                    "downloaded_at": "2026-04-23T00:00:00+00:00",
                },
            ]
        )
        return RawFilingCorpus(
            group_name=group_name,
            tickers=tickers,
            ticker_index=ticker_index,
            filings=filings,
            raw_filings=raw_filings,
            failures={},
        )


class DatabaseOperatorTests(unittest.TestCase):
    def test_raw_filing_writer_splits_large_text_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = RawFilingCorpusWriter()
            raw_filings = pd.DataFrame(
                [
                    {
                        "ticker": "AAA",
                        "cik": "0000000001",
                        "accession_no": "0000000001-26-000001",
                        "form": "10-K",
                        "filing_date": "2026-01-01",
                        "period_end": "2025-12-31",
                        "submission_text": "a",
                        "submission_text_length": 900_000_000,
                        "source_url": "https://sec.example/aaa.txt",
                        "downloaded_at": "2026-04-23T00:00:00+00:00",
                    },
                    {
                        "ticker": "BBB",
                        "cik": "0000000002",
                        "accession_no": "0000000002-26-000001",
                        "form": "10-K",
                        "filing_date": "2026-01-02",
                        "period_end": "2025-12-31",
                        "submission_text": "b",
                        "submission_text_length": 900_000_000,
                        "source_url": "https://sec.example/bbb.txt",
                        "downloaded_at": "2026-04-23T00:00:00+00:00",
                    },
                ]
            )
            corpus = RawFilingCorpus(
                group_name="Shard Limit Test",
                tickers=["AAA", "BBB"],
                raw_filings=raw_filings,
            )

            written = writer.write(corpus, Path(temp_dir), format="jsonl", include_supporting_data=False)
            shard_files = sorted(written["raw_filing_shards_dir"].glob("*.parquet"))
            self.assertEqual(len(shard_files), 2)

            index = pd.read_json(written["raw_filing_index"], orient="records", lines=True)
            self.assertEqual(index["raw_shard_path"].nunique(), 2)

    def test_populate_query_and_delete_jsonl_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            operator = DatabaseOperator(
                storage_dir=temp_dir,
                corpus_builder=FakeCorpusBuilder(),
                default_format="jsonl",
            )

            corpus, stored = operator.populate_for_symbols(
                ["AAPL", "MSFT"],
                group_name="Unit Test Corpus",
                format="jsonl",
            )

            self.assertEqual(corpus.summary()["raw_filing_rows"], 2)
            self.assertTrue(stored.raw_filings_path.exists())
            self.assertEqual(operator.list_tickers("Unit Test Corpus"), ["AAPL", "MSFT"])

            raw = operator.find_raw_filings("Unit Test Corpus", ticker="AAPL")
            self.assertEqual(len(raw), 1)
            self.assertEqual(raw.iloc[0]["submission_text_length"], 17)
            self.assertTrue(str(raw.iloc[0]["raw_shard_path"]).endswith(".parquet"))

            submission_text = operator.load_raw_submission_text(
                "Unit Test Corpus",
                "0000320193-26-000006",
            )
            self.assertEqual(submission_text, "apple filing text")

            structured = operator.latest_filings("Unit Test Corpus", limit=1)
            self.assertEqual(len(structured), 1)
            self.assertEqual(structured.iloc[0]["ticker"], "AAPL")

            description = operator.describe_corpus("Unit Test Corpus")
            self.assertEqual(description["dataset_format"], "jsonl")

            listed = operator.list_corpora()
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0].group_name, "Unit Test Corpus")

            operator.delete_corpus("Unit Test Corpus")
            self.assertEqual(operator.list_corpora(), [])

    def test_populate_and_query_sqlite_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sqlite_path = Path(temp_dir) / "raw_filing_corpora.sqlite"
            operator = DatabaseOperator(
                storage_dir=temp_dir,
                sqlite_path=sqlite_path,
                corpus_builder=FakeCorpusBuilder(),
                default_format="jsonl",
            )

            _, _, written_sqlite_path = operator.populate_for_symbols_with_sqlite(
                ["AAPL", "MSFT"],
                group_name="SQLite Unit Test Corpus",
                format="jsonl",
            )

            self.assertEqual(written_sqlite_path, sqlite_path)
            self.assertTrue(sqlite_path.exists())
            self.assertEqual(operator.list_sqlite_corpora(), ["SQLite Unit Test Corpus"])

            summary = operator.sqlite_summary()
            self.assertEqual(summary["corpora"], 1)
            self.assertEqual(summary["filings"], 2)
            self.assertEqual(summary["raw_filings"], 2)

            structured = operator.find_sqlite_structured_filings("SQLite Unit Test Corpus", ticker="MSFT")
            self.assertEqual(len(structured), 1)
            self.assertEqual(structured.iloc[0]["accession_no"], "0001193125-26-027207")

            raw = operator.find_sqlite_raw_filings("SQLite Unit Test Corpus", ticker="AAPL")
            self.assertEqual(len(raw), 1)
            self.assertEqual(raw.iloc[0]["submission_text_length"], 17)
            self.assertTrue(str(raw.iloc[0]["raw_shard_path"]).endswith(".parquet"))

    def test_sqlite_persistence_serializes_pandas_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sqlite_path = Path(temp_dir) / "raw_filing_corpora.sqlite"
            operator = DatabaseOperator(storage_dir=temp_dir, sqlite_path=sqlite_path)
            corpus = RawFilingCorpus(
                group_name="Timestamp Corpus",
                tickers=["AAA"],
                ticker_index=pd.DataFrame(
                    [
                        {
                            "ticker": "AAA",
                            "cik": "0000000001",
                            "date_added": pd.Timestamp("2020-01-02"),
                        }
                    ]
                ),
                filings=pd.DataFrame(
                    [
                        {
                            "ticker": "AAA",
                            "cik": "0000000001",
                            "accession_no": "0000000001-26-000001",
                            "form": "10-K",
                            "filing_date": pd.Timestamp("2026-01-31"),
                            "period_end": pd.Timestamp("2025-12-31"),
                        }
                    ]
                ),
                raw_filings=pd.DataFrame(
                    [
                        {
                            "ticker": "AAA",
                            "cik": "0000000001",
                            "accession_no": "0000000001-26-000001",
                            "form": "10-K",
                            "filing_date": pd.Timestamp("2026-01-31"),
                            "period_end": pd.Timestamp("2025-12-31"),
                            "submission_text": "aaa filing text",
                            "submission_text_length": 15,
                            "source_url": "https://sec.example/aaa.txt",
                            "downloaded_at": pd.Timestamp("2026-04-28T20:37:32Z"),
                            "raw_shard_path": "shards/aaa.parquet",
                            "raw_shard_row_number": 0,
                        }
                    ]
                ),
            )

            operator.persist_corpus_to_sqlite(corpus)
            raw = operator.find_sqlite_raw_filings("Timestamp Corpus", ticker="AAA")

            self.assertEqual(len(raw), 1)
            self.assertIn("2026-01-31", str(raw.iloc[0]["filing_date"]))
            self.assertIn("2026-04-28", str(raw.iloc[0]["downloaded_at"]))

    def test_persist_and_query_sqlite_prices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sqlite_path = Path(temp_dir) / "raw_filing_corpora.sqlite"
            operator = DatabaseOperator(
                storage_dir=temp_dir,
                sqlite_path=sqlite_path,
                corpus_builder=FakeCorpusBuilder(),
                default_format="jsonl",
            )

            prices = pd.DataFrame(
                [
                    {
                        "ticker": "AAPL",
                        "date": "2010-01-04",
                        "open": 30.49,
                        "high": 30.64,
                        "low": 30.34,
                        "close": 30.57,
                        "adj_close": 26.54,
                        "volume": 123432400,
                    },
                    {
                        "ticker": "MSFT",
                        "date": "2010-01-04",
                        "open": 30.62,
                        "high": 31.10,
                        "low": 30.59,
                        "close": 30.95,
                        "adj_close": 23.43,
                        "volume": 38409100,
                    },
                ]
            )

            operator.persist_prices_to_sqlite(prices, group_name="Prices Unit Test")
            summary = operator.sqlite_summary()
            self.assertEqual(summary["prices"], 2)

            fetched = operator.find_sqlite_prices("Prices Unit Test", ticker="AAPL")
            self.assertEqual(len(fetched), 1)
            self.assertEqual(fetched.iloc[0]["close"], 30.57)

    def test_price_history_frame_builder_drops_all_null_rows(self) -> None:
        builder = PriceHistoryFrameBuilder()
        frame = builder.build(
            "FISV",
            {
                "timestamp": [1762732800, 1762819200, 1762905600],
                "indicators": {
                    "quote": [
                        {
                            "open": [63.91, 63.60, None],
                            "high": [64.18, 64.48, None],
                            "low": [62.67, 62.84, None],
                            "close": [63.80, 64.26, None],
                            "volume": [8431600, 5427200, None],
                        }
                    ],
                    "adjclose": [{"adjclose": [63.80, 64.26, None]}],
                },
            },
        )

        self.assertEqual(len(frame), 2)
        self.assertFalse(frame[["open", "high", "low", "close", "adj_close", "volume"]].isna().all(axis=1).any())


if __name__ == "__main__":
    unittest.main()

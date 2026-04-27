from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.features.fundamentals.valuation import ValuationProducer
from src.ingest.fundamentals import extract_companyfacts_rows


class FakePriceDB:
    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices

    def load_prices(self, *, ticker=None, date_from=None, date_to=None) -> pd.DataFrame:
        frame = self.prices.copy()
        if date_from is not None:
            frame = frame[frame["date"] >= pd.Timestamp(date_from)]
        if date_to is not None:
            frame = frame[frame["date"] <= pd.Timestamp(date_to)]
        return frame.reset_index(drop=True)


class ValuationFeatureTests(unittest.TestCase):
    def test_companyfacts_extraction_reads_dei_share_counts(self) -> None:
        payload = {
            "facts": {
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": {
                            "shares": [
                                {
                                    "val": 123.0,
                                    "end": "2024-01-31",
                                    "filed": "2024-02-15",
                                    "form": "10-K",
                                    "fp": "FY",
                                    "fy": 2023,
                                }
                            ]
                        }
                    }
                }
            }
        }

        rows = extract_companyfacts_rows(payload, ticker="AAA", cik="0000000001")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["concept"], "EntityCommonStockSharesOutstanding")
        self.assertEqual(rows[0]["unit"], "shares")

    def test_valuation_producer_uses_filing_date_as_of_merge(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fundamentals_path = root / "fundamentals.parquet"
            fundamentals = pd.DataFrame(
                [
                    fact("AAA", "Revenues", 1000.0),
                    fact("AAA", "NetIncomeLoss", 100.0),
                    fact("AAA", "OperatingIncomeLoss", 150.0),
                    fact("AAA", "GrossProfit", 400.0),
                    fact("AAA", "Assets", 1500.0),
                    fact("AAA", "StockholdersEquity", 500.0),
                    fact("AAA", "LongTermDebt", 200.0),
                    fact("AAA", "CashAndCashEquivalentsAtCarryingValue", 100.0),
                    fact("AAA", "NetCashProvidedByUsedInOperatingActivities", 180.0),
                    fact("AAA", "PaymentsToAcquirePropertyPlantAndEquipment", 50.0),
                    fact("AAA", "CommonStockSharesOutstanding", 100.0),
                    fact("AAA", "PaymentsOfDividends", 10.0),
                    fact("AAA", "PaymentsForRepurchaseOfCommonStock", 20.0),
                ]
            )
            fundamentals.to_parquet(fundamentals_path, index=False)
            prices = pd.DataFrame(
                {
                    "ticker": ["AAA", "AAA"],
                    "date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
                    "adj_close": [8.0, 10.0],
                }
            )
            config = {
                "paths": {"fundamentals_path": str(fundamentals_path)},
                "data": {"start_date": "2024-01-01", "end_date": "2024-02-29"},
                "features": {"price": {"min_history_days": 1}},
            }

            output = ValuationProducer().run(FakePriceDB(prices), config, root)

            self.assertEqual(len(output), 2)
            self.assertTrue((root / "valuation.parquet").exists())
            before_filing = output[output["date"] == pd.Timestamp("2024-01-31")].iloc[0]
            after_filing = output[output["date"] == pd.Timestamp("2024-02-29")].iloc[0]
            self.assertTrue(pd.isna(before_filing["valuation_sales_yield"]))
            self.assertFalse(bool(before_filing["valuation_has_full_valuation"]))
            self.assertAlmostEqual(float(after_filing["valuation_sales_yield"]), 1.0)
            self.assertAlmostEqual(float(after_filing["valuation_earnings_yield"]), 0.1)
            self.assertAlmostEqual(float(after_filing["valuation_book_to_market"]), 0.5)
            self.assertAlmostEqual(float(after_filing["valuation_debt_to_market"]), 0.2)
            self.assertAlmostEqual(float(after_filing["valuation_cash_to_market"]), 0.1)
            self.assertAlmostEqual(float(after_filing["valuation_free_cash_flow_yield"]), 0.13)
            self.assertAlmostEqual(float(after_filing["valuation_shareholder_yield"]), 0.03)
            self.assertAlmostEqual(float(after_filing["valuation_ev_to_sales"]), np.log1p(1.1))
            self.assertTrue(bool(after_filing["valuation_has_full_valuation"]))


def fact(ticker: str, concept: str, value: float) -> dict[str, object]:
    return {
        "ticker": ticker,
        "cik": "0000000001",
        "concept": concept,
        "unit": "USD",
        "value": value,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "filing_date": "2024-02-15",
        "form": "10-K",
        "fiscal_period": "FY",
        "fiscal_year": 2023,
    }


if __name__ == "__main__":
    unittest.main()

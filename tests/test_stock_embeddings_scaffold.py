from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.config import deep_merge, load_config
from src.utils.dates import as_of_merge, select_month_end_rows
from scripts_compat import load_script


class StockEmbeddingsScaffoldTests(unittest.TestCase):
    def test_deep_merge_replaces_lists_and_merges_dicts(self) -> None:
        merged = deep_merge(
            {"a": {"x": 1, "items": [1, 2]}, "b": 2},
            {"a": {"y": 3, "items": [9]}},
        )
        self.assertEqual(merged["a"]["x"], 1)
        self.assertEqual(merged["a"]["y"], 3)
        self.assertEqual(merged["a"]["items"], [9])

    def test_load_config_baseline(self) -> None:
        config = load_config("baseline")
        self.assertEqual(config["experiment_name"], "baseline")
        self.assertEqual(config["model"]["name"], "pca")
        self.assertIn("price_volatility", config["features"]["enabled"])

    def test_select_month_end_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "AAA", "AAA"],
                "date": pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-01", "2024-02-29"]),
                "value": [1, 2, 3, 4],
            }
        )
        result = select_month_end_rows(frame)
        self.assertEqual(result["date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-31", "2024-02-29"])

    def test_as_of_merge_uses_latest_prior_observation(self) -> None:
        left = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            }
        )
        right = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "date": pd.to_datetime(["2024-01-15", "2024-02-10"]),
                "value": [1.0, 2.0],
            }
        )
        merged = as_of_merge(left, right, by=["ticker"])
        self.assertEqual(merged["value"].tolist(), [1.0, 2.0])

    def test_evaluation_metadata_coalesces_external_labels(self) -> None:
        module = load_script("05_evaluate.py", "script_05_evaluate_for_test")
        with TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "metadata.parquet"
            pd.DataFrame(
                {
                    "ticker": ["AAA"],
                    "gics_sector": ["Information Technology"],
                    "gics_sub_industry": ["Software"],
                }
            ).to_parquet(metadata_path, index=False)
            db = FakeMetadataDB(
                pd.DataFrame(
                    {
                        "ticker": ["AAA"],
                        "gics_sector": [None],
                        "title": ["AAA Corp"],
                    }
                )
            )
            config = {"paths": {"metadata_path": str(metadata_path)}}

            metadata = module.load_metadata(db, config)

        self.assertEqual(metadata.loc[0, "gics_sector"], "Information Technology")
        self.assertEqual(metadata.loc[0, "gics_sub_industry"], "Software")


class FakeMetadataDB:
    def __init__(self, tickers: pd.DataFrame) -> None:
        self.tickers = tickers

    def load_tickers(self) -> pd.DataFrame:
        return self.tickers.copy()


if __name__ == "__main__":
    unittest.main()

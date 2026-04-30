from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.config import load_config
from src.features.assembly import assemble_dataset
from src.features.fundamentals.growth_lifecycle import annual_fundamental_panel
from src.features.registry import build_active_producers
from src.ingest.fundamentals import CONCEPT_GROUPS
from src.models.pca import PCAModel
from src.utils.dates import as_of_merge


class FakeDB:
    def __init__(self, prices: pd.DataFrame, tickers: pd.DataFrame) -> None:
        self.prices = prices.copy()
        self.tickers = tickers.copy()

    def load_prices(
        self,
        *,
        ticker: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> pd.DataFrame:
        frame = self.prices.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        if ticker is not None:
            frame = frame[frame["ticker"] == ticker.upper()]
        if date_from is not None:
            frame = frame[frame["date"] >= pd.Timestamp(date_from)]
        if date_to is not None:
            frame = frame[frame["date"] <= pd.Timestamp(date_to)]
        return frame.reset_index(drop=True)

    def load_tickers(self) -> pd.DataFrame:
        return self.tickers.copy()


class PointInTimeTests(unittest.TestCase):
    def test_registered_feature_outputs_do_not_change_when_future_data_is_added(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fixture = SyntheticPITFixture(root)
            early_cutoff = pd.Timestamp("2020-12-31")
            late_cutoff = pd.Timestamp("2021-06-30")
            early_config = fixture.config_for_cutoff(early_cutoff, root / "early")
            late_config = fixture.config_for_cutoff(late_cutoff, root / "late")
            early_db = fixture.db_for_cutoff(early_cutoff)
            late_db = fixture.db_for_cutoff(late_cutoff)

            for name, producer in build_active_producers(early_config).items():
                with self.subTest(producer=name):
                    early = producer.compute(early_db, early_config)
                    late = producer.compute(late_db, late_config)
                    producer._validate(early)
                    producer._validate(late)
                    assert_same_history(
                        early,
                        late,
                        cutoff=early_cutoff,
                        key_columns=producer.spec.key_columns,
                    )

    def test_assembled_dataset_is_stable_before_cutoff(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fixture = SyntheticPITFixture(root)
            early_cutoff = pd.Timestamp("2020-12-31")
            late_cutoff = pd.Timestamp("2021-06-30")
            feature_names = [
                "price_momentum",
                "event_item_frequency",
                "text_business",
                "growth_lifecycle",
                "network_position",
            ]

            early_config = fixture.config_for_cutoff(early_cutoff, root / "early")
            late_config = fixture.config_for_cutoff(late_cutoff, root / "late")
            early_config["assembly"]["required_feature_groups"] = feature_names
            late_config["assembly"]["required_feature_groups"] = feature_names
            early_db = fixture.db_for_cutoff(early_cutoff)
            late_db = fixture.db_for_cutoff(late_cutoff)

            write_feature_groups(early_db, early_config, feature_names)
            write_feature_groups(late_db, late_config, feature_names)
            early_dataset = assemble_dataset(
                early_db,
                early_config,
                feature_dir=early_config["paths"]["feature_dir"],
                output_path=root / "early_dataset.parquet",
            )
            late_dataset = assemble_dataset(
                late_db,
                late_config,
                feature_dir=late_config["paths"]["feature_dir"],
                output_path=root / "late_dataset.parquet",
            )

            assert_same_history(
                early_dataset,
                late_dataset,
                cutoff=early_cutoff,
                key_columns=["ticker", "date"],
            )

    def test_as_of_merge_never_uses_future_feature_rows(self) -> None:
        panel = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "AAA", "BBB"],
                "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-02-29"]),
            }
        )
        features = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "AAA", "BBB"],
                "date": pd.to_datetime(["2020-01-15", "2020-02-29", "2020-04-01", "2020-03-01"]),
                "source_date": pd.to_datetime(["2020-01-15", "2020-02-29", "2020-04-01", "2020-03-01"]),
                "value": [1.0, 2.0, 999.0, 5.0],
            }
        )

        merged = as_of_merge(panel, features, by=["ticker"])
        self.assertEqual(merged.loc[0, "value"], 1.0)
        self.assertEqual(merged.loc[1, "value"], 2.0)
        self.assertEqual(merged.loc[2, "value"], 2.0)
        self.assertTrue(pd.isna(merged.loc[3, "value"]))
        known_sources = merged.dropna(subset=["source_date"])
        self.assertTrue((known_sources["source_date"] <= known_sources["date"]).all())

    def test_embedding_projection_is_stable_when_future_rows_are_exact_replicates(self) -> None:
        early_features = np.array(
            [
                [1.0, 2.0, 0.5],
                [2.0, 1.5, 0.7],
                [3.0, 0.5, 1.1],
                [4.0, 0.2, 1.4],
                [5.0, -0.2, 1.8],
                [6.0, -1.0, 2.0],
            ],
            dtype=float,
        )
        extended_features = np.vstack([early_features, early_features])

        early_model = PCAModel(input_dim=3, embedding_dim=2)
        extended_model = PCAModel(input_dim=3, embedding_dim=2)
        early_model.fit(early_features)
        extended_model.fit(extended_features)

        self.assertTrue(
            np.allclose(
                early_model.encode(early_features),
                extended_model.encode(early_features),
                atol=1e-10,
            )
        )

    def test_fundamental_panel_drops_periods_ending_after_filing(self) -> None:
        rows = synthetic_fundamentals()
        unsafe = rows.iloc[[0]].copy()
        unsafe["ticker"] = "AAA"
        unsafe["concept"] = CONCEPT_GROUPS["revenue"][0]
        unsafe["value"] = 999999.0
        unsafe["end_date"] = pd.Timestamp("2021-12-31")
        unsafe["filing_date"] = pd.Timestamp("2021-01-15")
        unsafe["fiscal_year"] = 2021
        unsafe["form"] = "10-K"
        unsafe["fiscal_period"] = "FY"
        combined = pd.concat([rows, unsafe], ignore_index=True)

        panel = annual_fundamental_panel(combined)
        leaked = panel[
            (panel["ticker"] == "AAA")
            & (panel["fiscal_year"] == 2021)
            & (panel.get("revenue") == 999999.0)
        ]

        self.assertTrue(leaked.empty)


class SyntheticPITFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.tickers = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "cik_str": ["0000000001", "0000000002"],
                "title": ["AAA Corp", "BBB Corp"],
                "search_label": ["AAA Corp", "BBB Corp"],
            }
        )
        self.prices = synthetic_prices()
        self.events = synthetic_events()
        self.sections = synthetic_sections()
        self.historical_embeddings = synthetic_historical_embeddings()
        self.fundamentals = synthetic_fundamentals()
        self.relationships = synthetic_relationships()

    def db_for_cutoff(self, cutoff: pd.Timestamp) -> FakeDB:
        prices = self.prices[self.prices["date"] <= cutoff].copy()
        return FakeDB(prices=prices, tickers=self.tickers)

    def config_for_cutoff(self, cutoff: pd.Timestamp, artifact_root: Path) -> dict:
        artifact_root.mkdir(parents=True, exist_ok=True)
        sections_dir = artifact_root / "sections"
        feature_dir = artifact_root / "features"
        sections_dir.mkdir(parents=True, exist_ok=True)
        feature_dir.mkdir(parents=True, exist_ok=True)

        events_path = artifact_root / "events.parquet"
        fundamentals_path = artifact_root / "fundamentals.parquet"
        relationships_path = artifact_root / "relationships.parquet"
        historical_text_path = artifact_root / "historical_section_embeddings.parquet"
        self.events[self.events["filing_date"] <= cutoff].to_parquet(events_path, index=False)
        self.fundamentals[self.fundamentals["filing_date"] <= cutoff].to_parquet(fundamentals_path, index=False)
        self.relationships[self.relationships["filing_date"] <= cutoff].to_parquet(relationships_path, index=False)
        self.historical_embeddings[self.historical_embeddings["filing_date"] <= cutoff].to_parquet(historical_text_path, index=False)
        self.sections[self.sections["filing_date"] <= cutoff].to_parquet(sections_dir / "ten_k_sections.parquet", index=False)

        config = load_config("baseline")
        config["random_seed"] = 7
        config["data"]["start_date"] = "2018-01-01"
        config["data"]["end_date"] = cutoff.strftime("%Y-%m-%d")
        config["paths"]["events_path"] = str(events_path)
        config["paths"]["fundamentals_path"] = str(fundamentals_path)
        config["paths"]["relationships_path"] = str(relationships_path)
        config["paths"]["sections_dir"] = str(sections_dir)
        config["paths"]["feature_dir"] = str(feature_dir)
        config["features"]["price"]["min_history_days"] = 252
        config["features"]["events"]["items"] = ["1.01", "2.02"]
        config["features"]["events"]["windows"] = [30, 90]
        config["features"]["text"]["method"] = "hashed"
        config["features"]["text"]["business_embedding_dim"] = 4
        config["features"]["text"]["risk_embedding_dim"] = 4
        config["features"]["text"]["min_token_length"] = 3
        config["features"]["text_historical"]["input_path"] = str(historical_text_path)
        config["features"]["text_historical"]["target_dim"] = 2
        config["features"]["text_historical"]["business_sections"] = ["business"]
        config["features"]["text_historical"]["risk_sections"] = ["risk_factors", "q_risk_factors"]
        config["features"]["text_historical"]["pca_fit_end_date"] = "2020-12-31"
        config["features"]["graph"]["min_confidence"] = 0.0
        config["assembly"]["metadata_columns"] = []
        return config


def synthetic_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-02", "2021-06-30")
    rows = []
    for ticker_index, ticker in enumerate(["AAA", "BBB"], start=1):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "adj_close": 50.0 + ticker_index * 10.0 + index * (0.03 + ticker_index * 0.005),
                    "volume": 1_000_000 + ticker_index * 100_000 + index * 100,
                }
            )
    return pd.DataFrame(rows)


def synthetic_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": "AAA", "filing_date": pd.Timestamp("2019-02-15"), "item": "1.01"},
            {"ticker": "AAA", "filing_date": pd.Timestamp("2020-08-10"), "item": "2.02"},
            {"ticker": "AAA", "filing_date": pd.Timestamp("2021-03-01"), "item": "1.01"},
            {"ticker": "BBB", "filing_date": pd.Timestamp("2020-05-20"), "item": "1.01"},
            {"ticker": "BBB", "filing_date": pd.Timestamp("2021-04-20"), "item": "2.02"},
        ]
    )


def synthetic_sections() -> pd.DataFrame:
    rows = []
    for ticker in ["AAA", "BBB"]:
        for year in [2018, 2019, 2020, 2021]:
            rows.append(
                {
                    "ticker": ticker,
                    "filing_date": pd.Timestamp(f"{year}-02-20"),
                    "business_text": f"{ticker} business operations cloud software customers year {year}",
                    "risk_factors_text": f"{ticker} risks cybersecurity supply chain inflation year {year}",
                }
            )
    return pd.DataFrame(rows)


def synthetic_historical_embeddings() -> pd.DataFrame:
    rows = []
    for ticker_index, ticker in enumerate(["AAA", "BBB"], start=1):
        for year in [2018, 2019, 2020, 2021]:
            for section_index, section in enumerate(["business", "risk_factors"]):
                rows.append(
                    {
                        "ticker": ticker,
                        "filing_date": pd.Timestamp(f"{year}-02-20"),
                        "section": section,
                        "embedding_0": float(ticker_index + year * 0.001 + section_index),
                        "embedding_1": float(ticker_index * 0.5 + year * 0.002 + section_index),
                    }
                )
            rows.append(
                {
                    "ticker": ticker,
                    "filing_date": pd.Timestamp(f"{year}-08-20"),
                    "section": "q_risk_factors",
                    "embedding_0": float(ticker_index + year * 0.001 + 2.0),
                    "embedding_1": float(ticker_index * 0.5 + year * 0.002 + 2.0),
                }
            )
    return pd.DataFrame(rows)


def synthetic_fundamentals() -> pd.DataFrame:
    rows = []
    first_concepts = {canonical: concepts[0] for canonical, concepts in CONCEPT_GROUPS.items()}
    for ticker_index, ticker in enumerate(["AAA", "BBB"], start=1):
        cik = str(ticker_index).zfill(10)
        for fiscal_year in range(2016, 2022):
            filing_date = pd.Timestamp(f"{fiscal_year + 1}-02-15")
            values = {
                "revenue": 1000.0 + ticker_index * 100.0 + (fiscal_year - 2016) * 80.0,
                "gross_profit": 400.0 + ticker_index * 30.0 + (fiscal_year - 2016) * 30.0,
                "operating_income": 160.0 + ticker_index * 20.0 + (fiscal_year - 2016) * 12.0,
                "net_income": 120.0 + ticker_index * 15.0 + (fiscal_year - 2016) * 10.0,
                "assets": 2500.0 + ticker_index * 200.0 + (fiscal_year - 2016) * 100.0,
                "stockholders_equity": 1300.0 + ticker_index * 100.0 + (fiscal_year - 2016) * 50.0,
                "long_term_debt": 500.0 + ticker_index * 20.0 + (fiscal_year - 2016) * 15.0,
                "rd_expense": 40.0 + ticker_index * 5.0,
                "capex": -90.0 - ticker_index * 5.0,
                "buybacks": -20.0 - ticker_index,
                "dividends": -15.0 - ticker_index,
                "shares_outstanding": 100.0 + ticker_index * 10.0,
            }
            for canonical, value in values.items():
                rows.append(
                    {
                        "ticker": ticker,
                        "cik": cik,
                        "concept": first_concepts[canonical],
                        "unit": "USD",
                        "value": value,
                        "start_date": f"{fiscal_year}-01-01",
                        "end_date": f"{fiscal_year}-12-31",
                        "filing_date": filing_date,
                        "form": "10-K",
                        "fiscal_period": "FY",
                        "fiscal_year": fiscal_year,
                    }
                )
    return pd.DataFrame(rows)


def synthetic_relationships() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_ticker": "AAA",
                "target_ticker": "BBB",
                "relationship_type": "competitor",
                "confidence": 0.9,
                "filing_date": pd.Timestamp("2019-03-01"),
            },
            {
                "source_ticker": "BBB",
                "target_ticker": "AAA",
                "relationship_type": "supplier",
                "confidence": 0.8,
                "filing_date": pd.Timestamp("2021-03-01"),
            },
        ]
    )


def write_feature_groups(db: FakeDB, config: dict, feature_names: list[str]) -> None:
    producers = build_active_producers(config)
    feature_dir = Path(config["paths"]["feature_dir"])
    feature_dir.mkdir(parents=True, exist_ok=True)
    for name in feature_names:
        frame = producers[name].compute(db, config)
        producers[name]._validate(frame)
        frame.to_parquet(feature_dir / f"{name}.parquet", index=False)


def assert_same_history(
    early: pd.DataFrame,
    late: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    key_columns: list[str],
) -> None:
    early_part = canonical_before_cutoff(early, cutoff, key_columns)
    late_part = canonical_before_cutoff(late, cutoff, key_columns)
    assert_frame_equal(early_part, late_part, check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-10)


def canonical_before_cutoff(frame: pd.DataFrame, cutoff: pd.Timestamp, key_columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result[result["date"] <= cutoff]
    result = result.sort_values(key_columns).reset_index(drop=True)
    return result.loc[:, sorted(result.columns)].reset_index(drop=True)


if __name__ == "__main__":
    unittest.main()

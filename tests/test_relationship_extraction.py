from __future__ import annotations

import unittest

import pandas as pd

from src.relationships.extract import build_company_aliases, classify_relationship, extract_relationships


class RelationshipExtractionTests(unittest.TestCase):
    def test_build_company_aliases_keeps_useful_variants(self) -> None:
        metadata = pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "company_name": ["Microsoft Corporation"],
            }
        )

        aliases = build_company_aliases(metadata)
        values = {alias.alias for alias in aliases}

        self.assertIn("Microsoft", values)

    def test_classify_relationship_detects_competition_context(self) -> None:
        relationship_type, confidence = classify_relationship(
            "We compete with Microsoft Corporation and other competitors in cloud infrastructure."
        )

        self.assertEqual(relationship_type, "competitor")
        self.assertGreater(confidence, 0.45)

    def test_extract_relationships_returns_stable_schema(self) -> None:
        metadata = pd.DataFrame(
            {
                "ticker": ["AAA", "MSFT", "NVDA"],
                "company_name": ["Acme Analytics Inc.", "Microsoft Corporation", "Nvidia Corporation"],
            }
        )
        sections = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "accession_no": ["0001"],
                "filing_date": [pd.Timestamp("2025-01-31")],
                "business_text": [
                    "Our competitors include Microsoft Corporation. "
                    "We also purchase specialized chips from Nvidia Corporation suppliers."
                ],
                "risk_factors_text": [""],
                "mda_text": [""],
            }
        )

        relationships = extract_relationships(sections, metadata)

        self.assertEqual(set(relationships["target_ticker"]), {"MSFT", "NVDA"})
        self.assertIn("competitor", set(relationships["relationship_type"]))
        self.assertIn("supplier", set(relationships["relationship_type"]))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import pandas as pd

from src.relationships.extract import (
    build_company_aliases,
    classify_relationship,
    classify_relationship_detail,
    extract_relationships,
)


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

    def test_classify_relationship_adds_supplier_direction(self) -> None:
        classification = classify_relationship_detail(
            "We purchase specialized chips from Nvidia Corporation suppliers."
        )

        self.assertEqual(classification.relationship_type, "supplier")
        self.assertEqual(classification.source_role, "customer")
        self.assertEqual(classification.target_role, "supplier")
        self.assertEqual(classification.supply_chain_direction, "target_supplies_source")
        self.assertGreater(classification.direction_confidence, 0.0)

    def test_target_buyer_context_is_customer_direction(self) -> None:
        classification = classify_relationship_detail(
            "Under the agreement, Microsoft will purchase the output generated from the renewed plant.",
            matched_alias="Microsoft",
        )

        self.assertEqual(classification.relationship_type, "customer")
        self.assertEqual(classification.supply_chain_direction, "source_supplies_target")

    def test_target_supplier_context_is_supplier_direction(self) -> None:
        classification = classify_relationship_detail(
            "Coca-Cola will continue to be our exclusive beverage supplier.",
            matched_alias="Coca-Cola",
        )

        self.assertEqual(classification.relationship_type, "supplier")
        self.assertEqual(classification.supply_chain_direction, "target_supplies_source")

    def test_hosting_provider_context_is_supplier_direction(self) -> None:
        classification = classify_relationship_detail(
            "We currently host our Falcon platform and serve our customers using a mix of "
            "third-party data centers, primarily Amazon Web Services, Inc.",
            matched_alias="Amazon",
        )

        self.assertEqual(classification.relationship_type, "supplier")
        self.assertEqual(classification.source_role, "customer")
        self.assertEqual(classification.target_role, "supplier")
        self.assertEqual(classification.supply_chain_direction, "target_supplies_source")

    def test_marketplace_context_is_not_forced_to_supply_chain(self) -> None:
        classification = classify_relationship_detail(
            "Customers can access a free trial through AWS Marketplace and Microsoft Marketplace.",
            matched_alias="Microsoft",
        )

        self.assertIn(classification.relationship_type, {"partner", "agreement", "generic"})
        self.assertEqual(classification.supply_chain_direction, "")

    def test_collaboration_context_is_partner_not_supplier(self) -> None:
        classification = classify_relationship_detail(
            "We maintain collaborative relationships with PepsiCo for branded distribution programs.",
            matched_alias="PepsiCo",
        )

        self.assertEqual(classification.relationship_type, "partner")
        self.assertEqual(classification.supply_chain_direction, "")

    def test_internal_platform_list_is_not_supplier_direction(self) -> None:
        classification = classify_relationship_detail(
            "The market includes suppliers of Arm-based CPUs and companies that incorporate hardware "
            "and software for internal solutions or platforms, such as Amazon and Microsoft.",
            matched_alias="Amazon",
        )

        self.assertNotEqual(classification.relationship_type, "supplier")
        self.assertEqual(classification.supply_chain_direction, "")

    def test_filer_supplied_to_target_is_customer_direction(self) -> None:
        classification = classify_relationship_detail(
            "We have supplied engines to PACCAR for 81 years.",
            matched_alias="PACCAR",
        )

        self.assertEqual(classification.relationship_type, "customer")
        self.assertEqual(classification.source_role, "supplier")
        self.assertEqual(classification.target_role, "customer")
        self.assertEqual(classification.supply_chain_direction, "source_supplies_target")

    def test_vendor_word_inside_competitor_list_stays_competitor(self) -> None:
        classification = classify_relationship_detail(
            "Our primary competitors in data center infrastructure are technology vendors, "
            "such as Dell Technologies Inc.",
            matched_alias="Dell Technologies",
        )

        self.assertEqual(classification.relationship_type, "competitor")
        self.assertEqual(classification.source_role, "competitor")
        self.assertEqual(classification.target_role, "competitor")

    def test_competition_coming_from_vendor_list_stays_competitor(self) -> None:
        classification = classify_relationship_detail(
            "The market has competition also coming from other large network equipment "
            "and system vendors, including Hewlett Packard Enterprise.",
            matched_alias="Hewlett Packard Enterprise",
        )

        self.assertEqual(classification.relationship_type, "competitor")
        self.assertEqual(classification.source_role, "competitor")
        self.assertEqual(classification.target_role, "competitor")

    def test_competitive_solution_vendor_context_stays_competitor(self) -> None:
        classification = classify_relationship_detail(
            "Microsoft, Alphabet, or those that have acquired security vendors and "
            "have the technical and financial resources to bring competitive solutions to the market.",
            matched_alias="Microsoft",
        )

        self.assertEqual(classification.relationship_type, "competitor")
        self.assertEqual(classification.source_role, "competitor")
        self.assertEqual(classification.target_role, "competitor")

    def test_executive_bio_context_is_not_operating_relationship(self) -> None:
        classification = classify_relationship_detail(
            "Wayne previously served as Chief Executive Officer of Surgery Partners, "
            "an operator of surgical facilities.",
            matched_alias="Surgery Partners",
        )

        self.assertEqual(classification.relationship_type, "generic")
        self.assertLess(classification.confidence, 0.45)

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
        nvidia = relationships.loc[relationships["target_ticker"] == "NVDA"].iloc[0]
        self.assertEqual(nvidia["supplier_ticker"], "NVDA")
        self.assertEqual(nvidia["customer_ticker"], "AAA")
        self.assertEqual(nvidia["supply_chain_direction"], "target_supplies_source")

    def test_extract_relationships_adds_customer_direction(self) -> None:
        metadata = pd.DataFrame(
            {
                "ticker": ["AAA", "MSFT"],
                "company_name": ["Acme Analytics Inc.", "Microsoft Corporation"],
            }
        )
        sections = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "accession_no": ["0001"],
                "filing_date": [pd.Timestamp("2025-01-31")],
                "business_text": ["Our customers include Microsoft Corporation."],
                "risk_factors_text": [""],
                "mda_text": [""],
            }
        )

        relationships = extract_relationships(sections, metadata)

        row = relationships.iloc[0]
        self.assertEqual(row["relationship_type"], "customer")
        self.assertEqual(row["supplier_ticker"], "AAA")
        self.assertEqual(row["customer_ticker"], "MSFT")
        self.assertEqual(row["supply_chain_direction"], "source_supplies_target")

    def test_lowercase_descriptive_matches_are_rejected(self) -> None:
        metadata = pd.DataFrame(
            {
                "ticker": ["AAA", "AMAT"],
                "company_name": ["Acme Analytics Inc.", "Applied Materials Inc."],
            }
        )
        sections = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "accession_no": ["0001"],
                "filing_date": [pd.Timestamp("2025-01-31")],
                "business_text": ["We sell into chemical and applied materials markets."],
                "risk_factors_text": [""],
                "mda_text": [""],
            }
        )

        relationships = extract_relationships(sections, metadata)

        self.assertTrue(relationships.empty)


if __name__ == "__main__":
    unittest.main()

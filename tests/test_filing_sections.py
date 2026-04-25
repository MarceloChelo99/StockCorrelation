from __future__ import annotations

import unittest

from src.filings.sections import extract_primary_document, parse_10k_sections, parse_filing_sections


SAMPLE_SUBMISSION = """
<SEC-DOCUMENT>
<DOCUMENT>
<TYPE>EX-13
<TEXT><html><body><div>Exhibit body</div></body></html></TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>10-K
<TEXT>
<html><body>
<div>PART I</div>
<div>Item 1.</div><div>Business</div><div>1</div>
<div>PART I</div>
<div>ITEM 1. BUSINESS</div>
<div>Table of Contents</div>
<div>ITEM 1. BUSINESS</div>
<div>Actual business text starts here.</div>
<div>It continues in a second paragraph.</div>
<div>ITEM 1A. RISK FACTORS</div>
<div>Table of Contents</div>
<div>ITEM 1A. RISK FACTORS</div>
<div>Actual risk factor text starts here.</div>
<div>ITEM 7. MD&amp;A</div>
<div>Table of Contents</div>
<div>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</div>
<div>Actual management discussion starts here.</div>
<div>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</div>
<div>Financial statement content.</div>
</body></html>
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""

SAMPLE_8K = """
<SEC-DOCUMENT>
<DOCUMENT>
<TYPE>8-K
<TEXT>
<html><body>
<div>ITEM 1.01 Entry into a Material Definitive Agreement.</div>
<div>Agreement text names a new strategic partnership.</div>
<div>ITEM 2.02 Results of Operations and Financial Condition.</div>
<div>Earnings narrative text starts here.</div>
<div>ITEM 9.01 Financial Statements and Exhibits.</div>
</body></html>
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""

SAMPLE_10Q = """
<SEC-DOCUMENT>
<DOCUMENT>
<TYPE>10-Q
<TEXT>
<html><body>
<div>ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</div>
<div>Quarterly operating discussion starts here.</div>
<div>ITEM 3. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK</div>
<div>Market risk text starts here.</div>
<div>ITEM 1. LEGAL PROCEEDINGS</div>
<div>Legal text starts here.</div>
<div>ITEM 1A. RISK FACTORS</div>
<div>Quarterly risk factor text starts here.</div>
</body></html>
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""


class FilingSectionTests(unittest.TestCase):
    def test_extract_primary_document_prefers_target_form(self) -> None:
        document = extract_primary_document(SAMPLE_SUBMISSION, form_type="10-K")
        self.assertIn("Actual business text starts here.", document)
        self.assertNotIn("Exhibit body", document)

    def test_parse_10k_sections_extracts_expected_sections(self) -> None:
        parsed = parse_10k_sections(SAMPLE_SUBMISSION)
        business = parsed.get("business")
        risk_factors = parsed.get("risk_factors")
        mda = parsed.get("mda")

        self.assertIsNotNone(business)
        self.assertIsNotNone(risk_factors)
        self.assertIsNotNone(mda)
        self.assertIn("Actual business text starts here.", business)
        self.assertIn("Actual risk factor text starts here.", risk_factors)
        self.assertIn("Actual management discussion starts here.", mda)
        self.assertNotIn("Exhibit body", business)
        self.assertLess(business.find("Actual business text starts here."), business.find("It continues"))

    def test_parse_filing_sections_dispatches_8k_items(self) -> None:
        parsed = parse_filing_sections(SAMPLE_8K, "8-K")

        agreement = parsed.get("eightk_item_1_01_material_agreement")
        results = parsed.get("eightk_item_2_02_results")

        self.assertIsNotNone(agreement)
        self.assertIsNotNone(results)
        self.assertIn("strategic partnership", agreement)
        self.assertIn("Earnings narrative", results)
        self.assertNotIn("ITEM 9.01", results)

    def test_parse_filing_sections_dispatches_10q_sections(self) -> None:
        parsed = parse_filing_sections(SAMPLE_10Q, "10-Q")

        self.assertIn("Quarterly operating discussion", parsed.get("q_mda") or "")
        self.assertIn("Market risk text", parsed.get("q_market_risk") or "")
        self.assertIn("Legal text", parsed.get("q_legal_proceedings") or "")
        self.assertIn("Quarterly risk factor text", parsed.get("q_risk_factors") or "")


if __name__ == "__main__":
    unittest.main()

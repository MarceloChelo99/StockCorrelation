# Stability Test

Temporal embeddings: `experiments/20260425_165422_temporal_business_view/embeddings.parquet`
Vanilla embeddings: `experiments/20260425_153806_historical_text_business/embeddings.parquet`
Stable-firm median temporal YoY displacement: `1.2139`
Stable-firm median vanilla YoY displacement: `1.3865`
Pass condition: temporal median < vanilla median.
Result: `PASS`

See `stability_yoy_displacements.csv` for firm/year details.

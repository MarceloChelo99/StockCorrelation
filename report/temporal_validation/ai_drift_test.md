# AI Drift Test

Embeddings: `experiments/20260425_165422_temporal_business_view/embeddings.parquet`
Window: `2022-12-31` to `2024-12-31`
AI-mover median displacement: `2.5210`
Control median displacement: `1.6805`
Pass condition: AI median >= 2x control median.
Result: `FAIL`

See `ai_drift_displacements.csv` and `ai_drift_histogram.svg` for details.

# Event Alignment Test

Embeddings: `experiments/20260425_165422_temporal_business_view/embeddings.parquet`
Event-window days: `45`
Event pass rate: `0.00%`
Baseline uses the firm's non-zero filing-update velocities, because monthly as-of embeddings are flat between filings.
Pass condition: at least 60% of events have event-window velocity > 1.5x positive baseline median.
Result: `FAIL`

See `event_alignment.csv` for event-level details.

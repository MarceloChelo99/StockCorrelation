# Temporal Autoencoder Results

This note summarizes the first temporal-aware business embedding run:

`experiments/20260425_165422_temporal_business_view/`

The model was trained on the historical text feature panel using the same business-view architecture as the vanilla autoencoder, plus adaptive temporal smoothing:

- `lambda_temp = 0.5`
- `alpha = 1.0`
- `max_pair_gap_days = 45`
- `embedding_dim = 32`
- `hidden_dims = [256, 128]`

## Training Diagnostics

The temporal precondition passed before training: historical text feature norms varied meaningfully within firms over time, so the model was not trained on degenerate static inputs.

Training produced:

- Embedding rows: `87,282`
- Fit rows after complete-case filtering: `84,048`
- Consecutive training pairs: `83,549`
- Dropped pair gaps: `0`
- Dropped non-finite pairs: `0`
- Validation temporal loss: `1.0040 -> 0.0790`
- Within-firm temporal variance ratio: `0.408`
- Velocity distribution: p10 `0.000`, p50 `0.000`, p90 `0.719`, max `3.157`

The zero median velocity is expected in the current panel because monthly observations are as-of carried forward between filing updates. The embedding moves when a new filing changes the historical text features, not every calendar month.

## Validation Tests

The first temporal configuration passed one validation test and failed two.

| Test | Result | Main number | Interpretation |
|---|---:|---:|---|
| AI drift | FAIL | AI median `2.521`, control median `1.681` | AI movers moved about `1.5x` controls, which is directional but below the strict `2x` pass bar. |
| Stability | PASS | temporal `1.214` vs vanilla `1.387` | Stable utilities/staples move less under the temporal AE, so smoothing is reducing noise. |
| Event alignment | FAIL | `0%` event pass rate | Named events did not reliably create velocity spikes above each firm's normal non-zero filing-update movement. |

The event test uses non-zero baseline filing-update velocities because the as-of monthly panel is flat between filings. Even with that fairer denominator, the first configuration did not pass.

## Downstream Metrics

Compared with the latest vanilla historical-text business run:

| Metric | Vanilla | Temporal | Read |
|---|---:|---:|---|
| Clustering ARI | `0.255` | `0.269` | Slightly better hard alignment with GICS. |
| Clustering NMI | `0.433` | `0.412` | Slightly worse information overlap with GICS, but within a modest degradation band. |
| 21-day peer diff vs GICS | `-0.1266` | `-0.1197` | Slightly less negative, still below GICS sub-industry peers. |
| Relationship direct rate | `0.1096` | `0.1192` | Slightly better than vanilla and far above random, still below GICS benchmark `0.1726`. |

## Assessment

This configuration is useful as a dashboard/exploration embedding because it smooths stable firms and keeps cross-sectional structure broadly intact. It should not yet be promoted as the final temporal model because it failed the AI-drift and event-alignment validation bars.

The next tuning direction is clear: reduce smoothing pressure or make it more adaptive so true filing-driven strategic shifts move farther. The obvious sweep is:

- `lambda_temp`: `0.1`, `0.25`, `0.5`, `1.0`
- `alpha`: `0.5`, `1.0`, `2.0`

The winner should pass stability without erasing AI/event movement, and should keep clustering NMI within roughly 10% of the vanilla baseline.

## Hyperparameter Sweep Update

A full 15-run sweep was completed after the first temporal run:

`report/temporal_validation/hyperparameter_sweep.csv`

Grid:

- `lambda_temp`: `0.1`, `0.25`, `0.5`, `1.0`, `2.0`
- `alpha`: `0.5`, `1.0`, `2.0`
- Epochs per run: `60`

Outcome:

- Overall pass: `0 / 15`
- Stability pass: `15 / 15`
- NMI preservation pass: `15 / 15`
- AI drift pass: `0 / 15`
- Event alignment pass: `0 / 15`

Best diagnostic candidate by the sweep score:

- `lambda_temp = 0.5`
- `alpha = 2.0`
- AI ratio: `1.565`
- Stability improvement ratio: `0.159`
- Event pass rate: `0.00%`
- NMI ratio to vanilla: `1.144`
- Clustering NMI: `0.496`

Interpretation: temporal smoothing is doing the easy part well. It reduces noise for stable firms and preserves, even improves, cross-sectional GICS structure. It is not yet solving event detection or strong strategic-shift detection. The failure is consistent across the grid, so the likely blocker is not just a bad `lambda/alpha` setting. More likely, the current as-of monthly panel moves only when filings update, and the selected 10-K/10-Q sections do not reliably encode the event dates we are testing.

The next modeling move should not be another small lambda sweep. Better next options are:

- Add event/text-change features from 8-K Item 1.01, Item 2.01, Item 2.02, Item 7.01, and Item 8.01 to the temporal input.
- Evaluate velocity around filing dates rather than calendar event dates for this business-text view.
- Add a supervised or contrastive event-sensitivity objective only after the unsupervised temporal AE baseline is fully understood.

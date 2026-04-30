# Growth Feature Value Check

This note answers whether the growth/lifecycle feature group is actually useful.

## Short Answer

Yes, but not in every form.

The raw growth/lifecycle features are useful for sector-relative prediction. The growth-view map/clusters are less compelling as a standalone visual because they are abstract, mixed, and not strongly aligned with GICS.

## What The Growth View Actually Contains

The current `growth` view is not pure growth. In `config/default.yaml`, the view pattern is:

```yaml
growth:
  feature_pattern: "^(growth_|valuation_)"
```

So the dashboard's growth view is really a **growth + valuation + lifecycle** view.

## Coverage And Data Quality

The feature file has full panel coverage:

- Rows: `92,887`
- Tickers: `503`
- Date range: `2010-01-29` to `2026-04-22`

But some individual features are sparse in the latest cross-section:

| Feature | Latest Missing Rate |
| --- | ---: |
| `growth_payout_dividend_yield` | 71.4% |
| `growth_gross_margin_trend` | 67.0% |
| `growth_gross_margin` | 63.8% |
| `growth_rd_intensity` | 62.0% |
| `growth_leverage` | 48.5% |
| `growth_capex_intensity_trend` | 40.8% |
| `growth_capex_intensity` | 37.2% |
| `growth_operating_margin` | 27.4% |

This means the feature group is not equally strong for every company. It works best as an aggregate signal, not as a perfect firm-by-firm descriptor.

## Sector Prediction Ablation

I ran a simple GICS-sector ridge walk-forward ablation with 63-day forward excess returns and date-added point-in-time membership filtering.

| Feature Set | Features | Mean Rank IC | Top-Bottom Spread | Rotation Ending Capital |
| --- | ---: | ---: | ---: | ---: |
| Price only | 5 | 0.0015 | -0.0022 | $46,986 |
| Price + growth | 21 | 0.0869 | 0.0131 | $70,163 |
| Price + valuation | 19 | 0.0275 | 0.0029 | $60,295 |
| Price + growth + valuation | 35 | 0.0922 | 0.0113 | $75,765 |

Interpretation: growth features clearly improve the sector prediction model versus price-only. Valuation helps too, but less than growth in this run. Combining growth and valuation produced the best ending capital.

## Peer / Similarity Evaluation

Growth does not work well for return-correlation peer identification.

From the multi-view peer horizon matrix:

| View | 21d Diff vs GICS | 504d Diff vs GICS |
| --- | ---: | ---: |
| Business | -0.0949 | -0.0689 |
| Behavioral | -0.1003 | -0.0789 |
| Growth | -0.1906 | -0.1324 |
| Network | -0.3031 | n/a |

The growth view is the weakest of the full-history views for pairwise return correlation. That does not mean the features are useless; it means "similar growth profile" is not the same thing as "stocks co-move like GICS peers."

## Cluster Interpretability

The growth themes have low/moderate alignment with GICS:

- NMI vs GICS sector: `0.175`
- ARI vs GICS sector: `0.034`

That is expected. Growth/lifecycle is cross-sector by construction. Examples of interpretable theme profiles:

- High gross margin / high R&D / high operating margin: software, high-quality tech, health-care equipment style names.
- Large mature value / high book-to-market / high earnings yield: financials, industrials, mature mega-cap firms.
- High asset growth / expensive valuation / lower earnings yield: expansion or high-expectation firms.
- High payout / buyback yield: shareholder-return-oriented mature firms.
- High revenue volatility: cyclical or transition-heavy firms.

These are real states, but they are harder to show as a colorful 2D map. They are better shown as feature-profile tables.

## Recommendation

Keep the growth features.

Use them for:

- Sector prediction
- Financial/lifecycle interpretation
- Explaining why a group looks mature, expensive, asset-heavy, high-growth, or shareholder-return-heavy

Do not overuse them for:

- Peer-return correlation
- Market-map visual storytelling
- Claiming that growth clusters are clean economic sectors

Best next improvement: rename the dashboard view from `growth` to **Lifecycle / Valuation** or split it into two separate views:

- `growth_lifecycle`: revenue growth, margins, investment, payout, leverage
- `valuation`: earnings yield, book-to-market, sales yield, EV multiples, shareholder yield


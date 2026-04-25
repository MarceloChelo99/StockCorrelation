# View Cluster Dynamics Summary

Canonical run: `experiments/20260424_032154_decomposed`.

Diagnostics are saved under `report/view_cluster_dynamics/`.

## Main Read

The four views behave very differently through time.

| View | Through-time behavior | Latest sector NMI | Latest active themes | Caveat |
| --- | --- | ---: | ---: | --- |
| Behavioral | Most dynamic; price-regime clustering changes often. | 0.241 | 35 | Very high year-to-year theme churn. |
| Business | Strongest latest sector/thematic separation. | 0.415 | 35 | Collapsed until 2025, so not useful as a long historical movie yet. |
| Growth | Medium-term lifecycle clustering; moderately stable. | 0.166 | 26 | Fundamentals coverage/imputation affects early years. |
| Network | Stable structural map. | 0.209 | 37 | Graph is static, so apparent time variation mostly comes from universe composition. |

## Patterns

Behavioral clustering becomes more diversified over time. In 2010, one behavioral theme covered 92.7% of stocks; by 2026, the largest theme covered only 12.0%. This view looks like a regime-sensitive trading-behavior map rather than a stable industry map.

Business clustering is not meaningful before 2025 in the current artifacts. Before then, nearly all stocks share the same dominant business theme. In 2026 it becomes the strongest sector-aligned view, with sector NMI 0.415 and clean sector pockets such as utilities, health care, real estate, energy, and communication services.

Growth clustering is useful but more lifecycle-oriented than sector-oriented. It keeps moderate sector alignment around 0.16-0.23 NMI and shows substantial theme turnover in years like 2013, 2022, and 2025. This is consistent with a view capturing capital intensity, profitability, leverage, payout, and maturity rather than business description.

Network clustering is the most stable view. It has all 37 themes active every year and almost zero year-to-year dominant-theme churn. That is expected because the current relationship graph is static and repeated across dates. It is valuable as a structural cross-section, not as a movement-through-time signal.

## Practical Dashboard Guidance

Use `behavioral` for the long moving map from 2011 onward.

Use `growth` for lifecycle shifts from roughly 2012 onward.

Use `network` for a stable relationship/supply-chain style map.

Use `business` for the latest semantic structure, but do not interpret pre-2025 movement in that view unless we rebuild historical text embeddings in a truly point-in-time way.

# View Interpretation Guide

Canonical run: `experiments/20260424_032154_decomposed`.

Per-view theme reports were generated for manual labeling:

| View | Theme report |
| --- | --- |
| Business | `report/view_business_themes.md` |
| Behavioral | `report/view_behavioral_themes.md` |
| Growth/lifecycle | `report/view_growth_themes.md` |
| Network | `report/view_network_themes.md` |

The theme reports intentionally leave labels blank. The right workflow is to inspect top firms, characteristic features, and GICS mix, then assign human labels such as “mega-cap platforms,” “regulated utilities,” “high-R&D healthcare,” or “capital-market intermediaries.”

`report/similarity_fingerprints.md` contains top-5 peers per view for META, AAPL, BRK.B, XOM, JPM, SBUX, PFE, and NEE. Use that file as the qualitative tour of what each view thinks “similar” means.

Current read: the low cross-view NMI says these views are not duplicates. Business and behavioral are the closest pair (`0.312`), while growth and network are the most distinct pair (`0.197`). That is good evidence for the decomposed framework even though the first covariance estimator did not work.

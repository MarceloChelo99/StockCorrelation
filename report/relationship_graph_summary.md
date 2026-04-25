# Relationship Graph Summary

The graph is extracted from parsed 10-K section text using conservative public-company name matching and rule-based context classification.

- Relationship rows: `676`
- Source tickers with at least one edge: `265`
- Target tickers mentioned: `210`

Relationship type counts:

| Type | Count |
| --- | ---: |
| competitor | 252 |
| agreement | 144 |
| customer | 128 |
| partner | 92 |
| supplier | 60 |

Semantic embedding graph-alignment evaluation:

| Peer Set | Direct Link Rate | Neighbor Jaccard |
| --- | ---: | ---: |
| Semantic embedding peers | 0.070 | 0.030 |
| GICS sub-industry peers | 0.173 | 0.076 |
| Random peers | 0.010 | 0.011 |

Interpretation: semantic peers are substantially more connected than random peers, but less connected than same-GICS-sub-industry peers.
That means the current text embedding contains relationship-network signal, while GICS remains stronger for direct disclosed links.

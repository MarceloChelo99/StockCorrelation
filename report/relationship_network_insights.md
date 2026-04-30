# Relationship Network Insights

This note summarizes early insights from the filing-derived relationship graph. The graph is sparse and rule-based, so the right interpretation is not "complete supply chain database." It is better viewed as a high-precision set of disclosed economic links that can support examples, diagnostics, and future graph features.

## Headline Findings

- The current graph contains `652` relationship rows across `239` competitor, `135` agreement, `109` customer, `86` partner, and `83` supplier classifications.
- `192` rows have explicit supplier/customer direction after the latest directionality pass.
- `64` of those `192` directed supply-chain rows are cross-sector, or `33.3%`. This is important because these are exactly the links GICS sector labels tend to hide.
- Competitor links are much more sector-aligned: `79.9%` of competitor edges connect firms in the same GICS sector. This is a useful sanity check because competition should mostly stay inside sector/industry boundaries.

## Most Useful Presentation Examples

These examples are clean enough to show in class because the filing language directly supports the direction:

| Supplier | Customer | Why It Matters |
| --- | --- | --- |
| Broadcom | Arista Networks | Arista discloses reliance on Broadcom merchant silicon for switching chips. This is a concrete semiconductor-to-network-equipment dependency. |
| Constellation Energy | Microsoft | Microsoft agrees to purchase energy output for data center power needs. This connects utilities to AI/cloud infrastructure demand. |
| Boeing | Southwest Airlines | Southwest discloses dependence on Boeing aircraft and parts, showing operational concentration risk. |
| Ball Corporation | Molson Coors | Packaging supplier relationship: aluminum containers and ends for beverages. |
| Coca-Cola | Domino's | Domino's discloses Coca-Cola as exclusive beverage supplier. This is a clean cross-sector restaurant/beverage link. |
| McKesson | CVS Health | McKesson discloses CVS as its largest customer, showing health-care distribution concentration. |
| Masco | Home Depot | Masco discloses Home Depot as a material customer, showing retail-channel dependence. |
| Akamai | Airbnb | Akamai lists Airbnb among major customers, showing cross-sector digital infrastructure exposure. |
| CrowdStrike | Delta Air Lines | Delta discloses operational disruption from the CrowdStrike outage, showing cyber vendor dependency. |
| Nvidia | HP Inc. | HP discloses reliance on Nvidia and other chip suppliers for PCs/workstations. |

## Network Hubs

Top disclosed suppliers by directed edges:

| Supplier | Directed Edges | Interpretation |
| --- | ---: | --- |
| Microsoft | 12 | Platform / distribution / enterprise software provider across several customer sectors. |
| Amazon | 10 | Marketplace, cloud, and distribution exposure across customers. |
| IBM | 6 | Enterprise technology vendor role. |
| Akamai | 5 | Internet infrastructure provider to cross-sector customers. |
| CDW | 5 | Technology distribution and reseller role. |

Top disclosed customers by directed edges:

| Customer | Directed Edges | Interpretation |
| --- | ---: | --- |
| Microsoft | 16 | Both major customer and major supplier, consistent with Microsoft being a platform node. |
| Amazon | 13 | Similar dual role: customer, marketplace, cloud provider, and distribution node. |
| Walmart | 12 | Classic large buyer/customer concentration node across consumer suppliers. |
| Hewlett Packard Enterprise | 6 | Hardware/platform buyer connected to technology suppliers. |
| Nvidia | 5 | Supply-chain and platform dependencies around accelerated computing. |

## Sector-Level Pattern

Supply-chain links are not just same-sector links. The most common directed sector pairs include:

| Supplier Sector | Customer Sector | Count |
| --- | --- | ---: |
| Information Technology | Information Technology | 57 |
| Consumer Staples | Consumer Staples | 17 |
| Financials | Financials | 15 |
| Industrials | Industrials | 14 |
| Information Technology | Consumer Discretionary | 10 |
| Consumer Discretionary | Information Technology | 9 |
| Industrials | Consumer Discretionary | 7 |
| Industrials | Information Technology | 5 |

The important point: the graph finds both traditional within-sector links and cross-sector economic dependencies. This supports the decomposed-similarity thesis: "similarity" is not only sector membership; it can also be customer/vendor exposure.

## Best Class Demo Angle

The cleanest class story is the contrast between competitor edges and supply-chain edges:

- Competitor graph: mostly validates GICS because competitors usually sit in the same sector.
- Supply-chain graph: often cuts across GICS because customers and vendors sit in different sectors.
- Embeddings: can then be framed as trying to learn these non-obvious cross-sector similarities from filings and structured features.

The Microsoft example is especially strong. Microsoft appears as both a major supplier and major customer. That is economically realistic: it sells platform/software/cloud infrastructure, but also buys power, content, hardware, and data-center inputs. A one-label sector classification cannot represent that dual role.

## Caveats

- This is sparse. It captures disclosed relationships, not the full supply chain.
- Directionality is rule-based. The latest pass fixed several obvious reverse-direction cases, but manual spot checks are still needed before using this as ground truth.
- Some relationships are not ongoing supplier/customer dependencies; they may be one-time agreements, acquisitions, distribution arrangements, or risk-factor mentions.
- Private counterparties are mostly invisible because we are matching against public-company metadata.
- The graph is currently strongest for presentation examples and feature engineering, not for claiming complete economic-network coverage.


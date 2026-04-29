# Relationship Graph Summary

The graph is extracted from parsed 10-K section text using conservative public-company name matching and rule-based context classification.
Customer/supplier rows are directional: `supplier_ticker` is the firm that provides the product/service, and `customer_ticker` is the firm that buys or depends on it.

- Relationship rows: `652`
- Source tickers with at least one edge: `251`
- Target tickers mentioned: `204`
- Directed supply-chain rows: `192`
- Disclosed supplier tickers: `113`
- Disclosed customer tickers: `102`

Relationship type counts:

| Type | Count |
| --- | ---: |
| competitor | 239 |
| agreement | 135 |
| customer | 109 |
| partner | 86 |
| supplier | 83 |

Top disclosed suppliers:

| Supplier | Directed Edges |
| --- | ---: |
| MSFT | 12 |
| AMZN | 10 |
| IBM | 6 |
| AKAM | 5 |
| CDW | 5 |
| NVDA | 4 |
| AVGO | 4 |
| CRWD | 4 |
| AMD | 3 |
| AXP | 3 |
| SNPS | 3 |
| MA | 3 |
| DAL | 3 |
| PANW | 3 |
| ROP | 3 |

Top disclosed customers:

| Customer | Directed Edges |
| --- | ---: |
| MSFT | 16 |
| AMZN | 13 |
| WMT | 12 |
| HPE | 6 |
| NVDA | 5 |
| PANW | 5 |
| NOW | 5 |
| MA | 4 |
| BA | 4 |
| ORCL | 4 |
| CMCSA | 3 |
| ANET | 3 |
| CDW | 3 |
| NTAP | 3 |
| VZ | 3 |

High-confidence directed examples:

| Supplier | Customer | Source Filing Company | Evidence |
| --- | --- | --- | --- |
| AVGO | ANET | ANET | In particular, we are primarily reliant upon our predominant merchant silicon vendor, Broadcom, for our switching chips. |
| MCO | ERIE | ERIE | (1) Ratings are supplied by S&P, Moody's, and Fitch . |
| NVDA | HPQ | HPQ | We also rely on Intel, AMD, and NVIDIA, or other suppliers to provide us with a sufficient supply of processors for the majority of our PCs and workstations. |
| BA | LUV | LUV | Further, if the -7 certification is not completed in a timely manner, the Company’s growth and network plans could be restricted unless and until it could pr... |
| PNR | POOL | POOL | Our largest suppliers include Pentair plc, Zodiac Pool Systems, Inc. |
| BALL | TAP | TAP | We have supply agreements with Ball Corporation and other vendors to purchase aluminum containers and ends in addition to what is supplied from RMMC. |
| NVDA | AKAM | AKAM | AIC addresses this need by leveraging Akamai’s expertise in globally distributed infrastructure and other architectures, such as those provided by NVIDIA, to... |
| DELL | CDW | CDW | A significant portion of our sales are derived from products manufactured by Apple, Cisco, Dell Technologies, HP Inc. |
| MA | CPAY | CPAY | A significant source of our revenue comes from processing transactions through the Mastercard networks. |
| MSFT | EA | EA | Digital full game units are based on sales information provided by Microsoft and Sony; packaged goods units sold through are estimated by obtaining data from... |
| BKR | HAL | HAL | Historical average rig counts shown are based on data provided by Baker Hughes, which included |
| TMO | IDXX | IDXX | We also distribute certain water testing kits manufactured by Thermo Fisher Scientific, Inc. |

Semantic embedding graph-alignment evaluation:

| Peer Set | Direct Link Rate | Neighbor Jaccard |
| --- | ---: | ---: |
| Semantic embedding peers | 0.070 | 0.030 |
| GICS sub-industry peers | 0.173 | 0.076 |
| Random peers | 0.010 | 0.011 |

Interpretation: semantic peers are substantially more connected than random peers, but less connected than same-GICS-sub-industry peers.
That means the current text embedding contains relationship-network signal, while GICS remains stronger for direct disclosed links.

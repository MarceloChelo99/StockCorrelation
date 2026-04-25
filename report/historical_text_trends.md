# Historical Text Trends

This report uses compact historical 10-K section features: keyword counts plus MiniLM embeddings. It does not depend on storing full raw SEC filings.

## Coverage

| year | section | rows | tickers | median_chars |
| --- | --- | --- | --- | --- |
| 2023 | business | 471 | 471 | 49366.000 |
| 2023 | item_1c_cybersecurity | 35 | 35 | 24968.000 |
| 2023 | mda | 486 | 486 | 70984.000 |
| 2023 | risk_factors | 482 | 482 | 70167.000 |
| 2024 | business | 474 | 474 | 49174.500 |
| 2024 | item_1c_cybersecurity | 477 | 477 | 6539.000 |
| 2024 | mda | 489 | 489 | 69222.000 |
| 2024 | risk_factors | 485 | 485 | 71771.000 |
| 2025 | business | 480 | 479 | 47477.000 |
| 2025 | item_1c_cybersecurity | 483 | 482 | 6684.000 |
| 2025 | mda | 495 | 494 | 66612.000 |
| 2025 | risk_factors | 489 | 488 | 73182.000 |
| 2026 | business | 392 | 392 | 46498.000 |
| 2026 | item_1c_cybersecurity | 396 | 396 | 7008.000 |
| 2026 | mda | 406 | 406 | 69200.500 |
| 2026 | risk_factors | 403 | 403 | 75007.000 |

## AI Trend By Section

| section | year | rows | tickers | ai_mentions | ai_score |
| --- | --- | --- | --- | --- | --- |
| business | 2018 | 448 | 446 | 151 | 1.572 |
| business | 2019 | 453 | 452 | 229 | 2.272 |
| business | 2020 | 458 | 458 | 264 | 2.458 |
| business | 2021 | 465 | 465 | 324 | 2.449 |
| business | 2022 | 471 | 471 | 355 | 2.637 |
| business | 2023 | 471 | 471 | 451 | 3.389 |
| business | 2024 | 474 | 474 | 676 | 5.693 |
| business | 2025 | 480 | 479 | 855 | 6.909 |
| business | 2026 | 392 | 392 | 686 | 7.112 |
| mda | 2018 | 457 | 454 | 50 | 0.256 |
| mda | 2019 | 464 | 463 | 43 | 0.218 |
| mda | 2020 | 469 | 469 | 48 | 0.295 |
| mda | 2021 | 476 | 476 | 73 | 0.341 |
| mda | 2022 | 482 | 482 | 86 | 0.414 |
| mda | 2023 | 486 | 486 | 80 | 0.397 |
| mda | 2024 | 489 | 489 | 166 | 1.011 |
| mda | 2025 | 495 | 494 | 252 | 1.414 |
| mda | 2026 | 406 | 406 | 249 | 1.557 |
| risk_factors | 2018 | 453 | 450 | 30 | 0.162 |
| risk_factors | 2019 | 460 | 459 | 55 | 0.353 |
| risk_factors | 2020 | 466 | 466 | 124 | 0.684 |
| risk_factors | 2021 | 474 | 474 | 136 | 0.687 |
| risk_factors | 2022 | 480 | 480 | 159 | 0.737 |
| risk_factors | 2023 | 482 | 482 | 308 | 1.576 |
| risk_factors | 2024 | 485 | 485 | 1280 | 6.803 |
| risk_factors | 2025 | 489 | 488 | 1833 | 9.988 |
| risk_factors | 2026 | 403 | 403 | 1782 | 11.220 |

## Largest Company AI Language Increases

| ticker | company_name | gics_sector | early_ai | late_ai | ai_change | late_ai_mentions | late_cloud_compute | late_cybersecurity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EPAM | EPAM Systems | Information Technology | 0.000 | 56.734 | 56.734 | 46 | 39.330 | 33.035 |
| ADBE | Adobe Inc. | Information Technology | 1.914 | 49.722 | 47.808 | 134 | 47.464 | 20.405 |
| ADP | Automatic Data Processing | Industrials | 0.000 | 46.155 | 46.155 | 64 | 18.423 | 37.080 |
| AMZN | Amazon | Consumer Discretionary | 0.000 | 39.233 | 39.233 | 57 | 13.875 | 3.036 |
| EFX | Equifax | Industrials | 0.000 | 37.626 | 37.626 | 98 | 17.652 | 31.132 |
| PANW | Palo Alto Networks | Information Technology | 0.309 | 32.818 | 32.509 | 67 | 57.557 | 24.632 |
| NVDA | Nvidia | Information Technology | 22.959 | 55.396 | 32.437 | 121 | 192.170 | 10.384 |
| CDNS | Cadence Design Systems | Information Technology | 0.936 | 31.902 | 30.967 | 71 | 60.713 | 6.204 |
| NOW | ServiceNow | Information Technology | 0.445 | 26.321 | 25.877 | 56 | 31.595 | 11.718 |
| INTU | Intuit | Information Technology | 2.832 | 27.676 | 24.844 | 59 | 4.627 | 13.889 |
| NWSA | News Corp (Class A) | Communication Services | 0.000 | 24.816 | 24.816 | 45 | 0.998 | 15.180 |
| EMR | Emerson Electric | Industrials | 0.000 | 22.647 | 22.647 | 19 | 4.497 | 54.718 |
| ORCL | Oracle Corporation | Information Technology | 0.374 | 22.778 | 22.404 | 41 | 144.637 | 8.619 |
| QCOM | Qualcomm | Information Technology | 3.319 | 25.721 | 22.402 | 47 | 40.085 | 10.367 |
| MSFT | Microsoft | Information Technology | 2.801 | 24.585 | 21.784 | 43 | 53.047 | 20.524 |
| META | Meta Platforms | Communication Services | 3.472 | 25.131 | 21.659 | 77 | 5.728 | 4.433 |
| VRSK | Verisk Analytics | Industrials | 0.450 | 22.037 | 21.588 | 42 | 14.748 | 8.796 |
| ZBH | Zimmer Biomet | Health Care | 0.000 | 21.288 | 21.288 | 44 | 0.000 | 14.538 |
| PAYX | Paychex | Industrials | 0.000 | 20.854 | 20.854 | 24 | 5.985 | 22.069 |
| SNPS | Synopsys | Information Technology | 0.000 | 20.481 | 20.481 | 38 | 68.077 | 18.954 |
| AXP | American Express | Financials | 0.000 | 20.171 | 20.171 | 82 | 1.295 | 30.110 |
| JKHY | Jack Henry & Associates | Financials | 0.000 | 19.576 | 19.576 | 24 | 21.901 | 29.725 |
| FDX | FedEx | Industrials | 0.000 | 19.340 | 19.340 | 55 | 2.468 | 15.431 |
| ZBRA | Zebra Technologies | Information Technology | 0.000 | 19.018 | 19.018 | 34 | 10.826 | 26.199 |
| FISV | Fiserv | Financials | 0.000 | 18.641 | 18.641 | 45 | 3.957 | 26.800 |

## Sector AI Language Increases

| gics_sector | tickers | early_ai | late_ai | ai_change | late_cloud_compute | late_cybersecurity |
| --- | --- | --- | --- | --- | --- | --- |
| Information Technology | 73 | 1.499 | 14.623 | 13.124 | 44.801 | 17.071 |
| Communication Services | 20 | 0.663 | 9.311 | 8.648 | 4.171 | 13.661 |
| Industrials | 79 | 0.029 | 6.524 | 6.494 | 4.691 | 18.785 |
| Financials | 76 | 0.124 | 6.384 | 6.260 | 2.617 | 20.690 |
| Consumer Staples | 36 | 0.000 | 5.430 | 5.430 | 0.939 | 14.760 |
| Consumer Discretionary | 48 | 0.013 | 5.433 | 5.420 | 1.496 | 11.916 |
| Health Care | 58 | 0.004 | 4.759 | 4.754 | 1.437 | 12.436 |
| Real Estate | 31 | 0.000 | 3.895 | 3.895 | 19.502 | 11.376 |
| Energy | 22 | 0.013 | 2.899 | 2.886 | 1.727 | 22.954 |
| Utilities | 31 | 0.000 | 2.191 | 2.191 | 3.564 | 14.031 |
| Materials | 26 | 0.000 | 1.839 | 1.839 | 1.970 | 21.676 |
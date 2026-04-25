# Decomposed Similarity Thesis

Company similarity is not one object. This expansion decomposes similarity into four views: business similarity from 10-K text, behavioral similarity from price-derived features, growth/lifecycle similarity from XBRL fundamentals, and network similarity from filing-derived company relationships.

The research question is whether those views are complementary enough to improve downstream risk estimation. The answer from the first full decomposed run is: the views are meaningfully different, but the current multi-view factor covariance estimator is not yet better than Ledoit-Wolf. That is still useful. It says the decomposition has interpretive value, while the covariance construction needs a better calibration or hybrid policy before it has portfolio value.

Canonical run: `experiments/20260424_032154_decomposed`.

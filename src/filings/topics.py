"""Keyword topic counters for filing sections before embedding.

These counts are deliberately simple and transparent. They run before text
embedding so every parsed section keeps interpretable topic evidence even if we
later change the embedding model or add more sections.
"""
from __future__ import annotations

import re


TOPICS: dict[str, dict[str, float]] = {
    "ai": {
        "artificial intelligence": 3.0,
        "generative ai": 4.0,
        "machine learning": 2.5,
        "large language model": 4.0,
        "large language models": 4.0,
        "foundation model": 3.5,
        "foundation models": 3.5,
        "deep learning": 3.0,
        "neural network": 3.0,
        "neural networks": 3.0,
        "ai-enabled": 2.0,
        "ai driven": 2.0,
        "ai-driven": 2.0,
    },
    "cloud_compute": {
        "cloud": 1.0,
        "data center": 2.0,
        "data centers": 2.0,
        "gpu": 2.5,
        "gpus": 2.5,
        "accelerated computing": 3.0,
        "semiconductor": 2.0,
        "semiconductors": 2.0,
        "chip": 1.0,
        "chips": 1.0,
    },
    "electrification": {
        "electrification": 3.0,
        "electric vehicle": 2.0,
        "electric vehicles": 2.0,
        "battery": 1.5,
        "batteries": 1.5,
        "renewable": 1.5,
        "solar": 1.5,
        "wind": 1.0,
        "transmission": 1.0,
        "grid": 1.0,
    },
    "cybersecurity": {
        "cybersecurity": 3.0,
        "cyber security": 3.0,
        "cyber attack": 2.0,
        "cyberattack": 2.0,
        "ransomware": 3.0,
        "data breach": 2.0,
        "information security": 1.5,
    },
    "supply_chain": {
        "supply chain": 2.0,
        "supplier": 1.0,
        "suppliers": 1.0,
        "customer concentration": 2.0,
        "logistics": 1.5,
        "distribution network": 1.5,
        "manufacturing partner": 2.0,
    },
}


def topic_counts_for_text(text: str) -> dict[str, int | float]:
    """Return raw mentions and weighted mentions per 10k words for one text blob."""
    normalized = str(text or "").lower()
    word_count = max(1, len(re.findall(r"[A-Za-z]+", normalized)))
    counts: dict[str, int | float] = {"word_count": word_count}
    for topic, terms in TOPICS.items():
        raw_mentions = 0
        weighted_mentions = 0.0
        for phrase, weight in terms.items():
            count = phrase_count(normalized, phrase)
            raw_mentions += count
            weighted_mentions += weight * count
        counts[f"topic_{topic}_mentions"] = raw_mentions
        counts[f"topic_{topic}_score_per_10k_words"] = 10000.0 * weighted_mentions / word_count
    return counts


def phrase_count(text: str, phrase: str) -> int:
    """Count phrase occurrences with simple word boundaries."""
    escaped = re.escape(phrase.lower()).replace(r"\ ", r"\s+")
    return len(re.findall(rf"(?<![A-Za-z]){escaped}(?![A-Za-z])", text))

"""Compatibility shim for shared networking helpers."""
from __future__ import annotations

from market_data_fetcher.networking import RequestRateLimiter, create_ssl_context, urlopen

__all__ = ["RequestRateLimiter", "create_ssl_context", "urlopen"]

"""Shared networking helpers for market-data downloads."""
from __future__ import annotations

import ssl
import threading
import time
import urllib.request
from collections import deque

import certifi


class RequestRateLimiter:
    """Simple sliding-window limiter to cap requests per period."""

    def __init__(self, max_requests: int = 10, period_seconds: float = 1.0) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be positive.")
        if period_seconds <= 0:
            raise ValueError("period_seconds must be positive.")

        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self._request_times: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until a new request is allowed within the configured window."""
        while True:
            with self._lock:
                now = time.monotonic()

                while self._request_times and now - self._request_times[0] >= self.period_seconds:
                    self._request_times.popleft()

                if len(self._request_times) < self.max_requests:
                    self._request_times.append(now)
                    return

                sleep_for = self.period_seconds - (now - self._request_times[0])

            if sleep_for > 0:
                time.sleep(sleep_for)


def create_ssl_context() -> ssl.SSLContext:
    """Create an HTTPS context backed by certifi's CA bundle."""
    return ssl.create_default_context(cafile=certifi.where())


def urlopen(request: urllib.request.Request, *, timeout: int = 30):
    """Open a URL with a stable certificate bundle."""
    return urllib.request.urlopen(request, timeout=timeout, context=create_ssl_context())

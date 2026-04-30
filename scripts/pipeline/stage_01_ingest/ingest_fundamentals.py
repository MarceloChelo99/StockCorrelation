from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import os
import sys
from pathlib import Path



from src.config import load_config
from src.db import FilingsDB
from src.ingest.fundamentals import ingest_fundamentals
from src.utils.logging import log


def main(config_name: str = "baseline") -> None:
    """Fetch SEC companyfacts fundamentals for the configured ticker universe."""
    config = load_config(config_name)
    db = FilingsDB.from_config(config)
    tickers = db.load_tickers()
    settings = config.get("fundamentals", {})
    user_agent = os.environ.get(
        "SEC_USER_AGENT",
        settings.get("user_agent", "StockCorrelation research castellanosmarcelo1@gmail.com"),
    )
    smoke_limit = int(settings["smoke_limit"]) if settings.get("smoke_test") and settings.get("smoke_limit") else None
    result = ingest_fundamentals(
        tickers,
        output_path=config["paths"]["fundamentals_path"],
        failures_path=settings.get("failures_path", "data/raw/fundamentals_failures.csv"),
        user_agent=user_agent,
        request_delay_seconds=float(settings.get("request_delay_seconds", 0.12)),
        max_retries=int(settings.get("max_retries", 4)),
        smoke_limit=smoke_limit,
        checkpoint_every=int(settings.get("checkpoint_every", 25)),
    )
    log(f"Wrote {result.rows:,} fundamentals rows to {result.output_path}.", tag="fundamentals")
    log(f"Revenue coverage: {result.revenue_coverage:.1%}.", tag="fundamentals")
    if result.revenue_coverage < 0.90 and smoke_limit is None:
        log("Revenue coverage is below the expected 90% threshold; continuing with logged failures.", tag="fundamentals")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")

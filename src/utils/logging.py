"""Tiny timestamped logging helper for research scripts."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


def log(message: str, *, tag: str | None = None, log_path: str | Path | None = None) -> None:
    """Print a timestamped log message and optionally append it to a file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{timestamp}]"
    if tag is not None:
        prefix += f" [{tag}]"
    line = f"{prefix} {message}"
    print(line, flush=True)
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

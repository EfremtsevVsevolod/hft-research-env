"""Shared formatting helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def parse_filename_ts(path: Path) -> Optional[int]:
    """Extract epoch-ms timestamp from a ``YYYYMMdd_HHMMSS`` filename."""
    try:
        dt = datetime.strptime(path.stem, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        return None


def fmt_count(n: int) -> str:
    """Format large number as 1.2M / 350K / 42."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)

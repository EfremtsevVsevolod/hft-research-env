"""Shared formatting helpers."""

from __future__ import annotations


def fmt_count(n: int) -> str:
    """Format large number as 1.2M / 350K / 42."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)

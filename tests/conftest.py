from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from src.config.config import SymbolConfig
from src.lob.orderbook import OrderBook


@pytest.fixture
def cfg() -> SymbolConfig:
    """BTCUSDT-like config: tick=0.01, step=0.01 (simplified for easy math)."""
    return SymbolConfig(
        symbol="TESTUSDT",
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.01"),
        inv_tick=100,
        inv_step=100,
    )


@pytest.fixture
def book(cfg) -> OrderBook:
    return OrderBook(cfg)


def make_snapshot(bids: list[tuple[str, str]], asks: list[tuple[str, str]]) -> dict:
    """Build snapshot dict from (price, qty) string pairs."""
    return {"bids": list(bids), "asks": list(asks)}


def make_depth_payload(u: int, bids: list[tuple[str, str]], asks: list[tuple[str, str]],
                       *, U: int | None = None) -> dict:
    """Build Binance-format depth update payload."""
    return {
        "e": "depthUpdate",
        "U": U if U is not None else u,
        "u": u,
        "b": list(bids),
        "a": list(asks),
    }


def make_trade_payload(price: str, qty: str, is_buyer_maker: bool) -> dict:
    """Build Binance-format trade payload."""
    return {
        "e": "trade",
        "T": 0,  # not used (replay uses recv_ts)
        "p": price,
        "q": qty,
        "m": is_buyer_maker,
    }


def write_test_parquet(path: Path, events: list[tuple[int, str, dict]]) -> Path:
    """Write synthetic events to a parquet file.

    Each event is (recv_ts, event_type, payload_dict).
    Returns path to the written file.
    """
    rows = []
    for recv_ts, event_type, payload in events:
        rows.append({
            "recv_ts": recv_ts,
            "exchange_ts": recv_ts,
            "event_type": event_type,
            "stream": "test",
            "payload_json": json.dumps(payload),
        })
    df = pd.DataFrame(rows)
    parquet_path = path / "test_data.parquet"
    df.to_parquet(parquet_path, index=False)
    return path

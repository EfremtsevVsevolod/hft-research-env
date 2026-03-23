from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from src.config.config import SymbolConfig
from src.data.constants import EVENT_DEPTH_UPDATE, EVENT_TRADE
from src.dataset.dataset import DatasetBuilder
from src.lob.features import FeatureExtractor
from src.lob.labels import LabelBuilder
from src.lob.orderbook import OrderBook
from src.replay.replay_engine import ReplayEngine


# --- fixtures ----------------------------------------------------------------

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


# --- event builders ----------------------------------------------------------

def make_snapshot(bids: list[tuple[str, str]], asks: list[tuple[str, str]]) -> dict:
    return {"bids": list(bids), "asks": list(asks)}


def make_depth_payload(u: int, bids: list[tuple[str, str]], asks: list[tuple[str, str]],
                       *, U: int | None = None) -> dict:
    return {"e": "depthUpdate", "U": U if U is not None else u, "u": u,
            "b": list(bids), "a": list(asks)}


def make_trade_payload(price: str, qty: str, is_buyer_maker: bool) -> dict:
    return {"e": "trade", "T": 0, "p": price, "q": qty, "m": is_buyer_maker}


# --- stubs for narrow tests --------------------------------------------------

class SnapshotSink:
    """Drop-in for LabelBuilder — records snapshots without labeling."""
    _horizon = 100  # satisfies ReplayEngine assertion

    def __init__(self):
        self.snapshots = []

    def on_snapshot(self, snap):
        self.snapshots.append(snap)
        return None

    def reset(self):
        self.snapshots.clear()


class LabelSink:
    """Drop-in for DatasetBuilder — records labelled snapshots without filtering."""

    def __init__(self):
        self.labelled = []

    def on_labelled_snapshot(self, ls):
        self.labelled.append(ls)

    def reset_timestamp(self):
        pass

    def __len__(self):
        return len(self.labelled)


# --- replay helpers (3 levels) -----------------------------------------------

def _feed_events(engine, events):
    for recv_ts, event_type, data in events:
        if event_type == EVENT_TRADE:
            engine._on_trade(recv_ts, data)
        elif event_type == EVENT_DEPTH_UPDATE:
            engine._on_depth(recv_ts, data)
        engine._event_count += 1


def replay_to_snapshots(cfg, events, *, interval=100, warmup_s=0):
    """Replay → FeatureSnapshots only. No labeling, no dataset."""
    book = OrderBook(cfg)
    fe = FeatureExtractor(sampling_interval_ms=interval, trade_window_ms=1000)
    sink = SnapshotSink()
    sink._horizon = interval
    engine = ReplayEngine(
        data_path=Path("/unused"), order_book=book, feature_extractor=fe,
        label_builder=sink, dataset_builder=LabelSink(), warmup_seconds=warmup_s,
    )
    _feed_events(engine, events)
    return engine, sink.snapshots, book


def replay_to_labels(cfg, events, *, interval=100, horizon=200, warmup_s=0):
    """Replay → LabelledSnapshots. No dataset filtering."""
    book = OrderBook(cfg)
    fe = FeatureExtractor(sampling_interval_ms=interval, trade_window_ms=1000)
    lb = LabelBuilder(horizon_ms=horizon, sampling_interval_ms=interval)
    sink = LabelSink()
    engine = ReplayEngine(
        data_path=Path("/unused"), order_book=book, feature_extractor=fe,
        label_builder=lb, dataset_builder=sink, warmup_seconds=warmup_s,
    )
    _feed_events(engine, events)
    return engine, sink.labelled, book


def replay_to_dataset(cfg, events, *, interval=100, horizon=200, warmup_s=0):
    """Replay → full pipeline → DatasetBuilder DataFrame."""
    book = OrderBook(cfg)
    fe = FeatureExtractor(sampling_interval_ms=interval, trade_window_ms=1000)
    lb = LabelBuilder(horizon_ms=horizon, sampling_interval_ms=interval)
    db = DatasetBuilder()
    engine = ReplayEngine(
        data_path=Path("/unused"), order_book=book, feature_extractor=fe,
        label_builder=lb, dataset_builder=db, warmup_seconds=warmup_s,
    )
    _feed_events(engine, events)
    return engine, db, book

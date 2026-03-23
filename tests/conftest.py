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


# --- snapshot lookup helper --------------------------------------------------

def snap_at(snapshots, ts):
    """Find snapshot with exact timestamp. Raises if not found."""
    matches = [s for s in snapshots if s.timestamp == ts]
    assert len(matches) == 1, f"expected 1 snapshot at t={ts}, got {len(matches)}"
    return matches[0]


# --- test sinks (implement expected interfaces explicitly) -------------------

class SnapshotSink:
    """Drop-in for LabelBuilder — records snapshots without labeling.

    Implements the same interface ReplayEngine expects from LabelBuilder:
      - _horizon (int): checked in __init__ assertion
      - on_snapshot(snap) -> Optional[LabelledSnapshot]
      - reset()
    """

    def __init__(self, horizon: int):
        self._horizon = horizon
        self.snapshots: list = []

    def on_snapshot(self, snap):
        self.snapshots.append(snap)
        return None

    def reset(self):
        self.snapshots.clear()


class LabelSink:
    """Drop-in for DatasetBuilder — records labelled snapshots without filtering.

    Implements the same interface ReplayEngine expects from DatasetBuilder:
      - on_labelled_snapshot(ls)
      - reset_timestamp()
      - __len__()
    """

    def __init__(self):
        self.labelled: list = []

    def on_labelled_snapshot(self, ls):
        self.labelled.append(ls)

    def reset_timestamp(self):
        pass

    def __len__(self):
        return len(self.labelled)


# --- replay helpers (3 levels) -----------------------------------------------

def _make_engine(cfg, *, interval, label_builder, dataset_builder, warmup_s):
    book = OrderBook(cfg)
    fe = FeatureExtractor(sampling_interval_ms=interval, trade_window_ms=1000)
    engine = ReplayEngine(
        data_path=Path("/unused"), order_book=book, feature_extractor=fe,
        label_builder=label_builder, dataset_builder=dataset_builder,
        warmup_seconds=warmup_s,
    )
    return engine, book


def _feed(engine, events):
    """Feed events through engine's public process_event method."""
    for recv_ts, event_type, data in events:
        engine.process_event(recv_ts, event_type, data)


def replay_to_snapshots(cfg, events, *, interval=100, warmup_s=0):
    """Replay → FeatureSnapshots only. No labeling, no dataset."""
    sink = SnapshotSink(horizon=interval)
    engine, book = _make_engine(cfg, interval=interval, label_builder=sink,
                                dataset_builder=LabelSink(), warmup_s=warmup_s)
    _feed(engine, events)
    return engine, sink.snapshots, book


def replay_to_labels(cfg, events, *, interval=100, horizon=200, warmup_s=0):
    """Replay → LabelledSnapshots. No dataset filtering."""
    lb = LabelBuilder(horizon_ms=horizon, sampling_interval_ms=interval)
    sink = LabelSink()
    engine, book = _make_engine(cfg, interval=interval, label_builder=lb,
                                dataset_builder=sink, warmup_s=warmup_s)
    _feed(engine, events)
    return engine, sink.labelled, book


def replay_to_dataset(cfg, events, *, interval=100, horizon=200, warmup_s=0):
    """Replay → full pipeline → DatasetBuilder DataFrame."""
    lb = LabelBuilder(horizon_ms=horizon, sampling_interval_ms=interval)
    db = DatasetBuilder()
    engine, book = _make_engine(cfg, interval=interval, label_builder=lb,
                                dataset_builder=db, warmup_s=warmup_s)
    _feed(engine, events)
    return engine, db, book

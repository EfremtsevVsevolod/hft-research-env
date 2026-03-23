"""Causal contract tests for ReplayEngine.

Tests are split by level:
  - Snapshot-level: replay → FeatureSnapshots (no labeling/dataset)
  - Engine-level: replay state (gaps, resets, book)
  - End-to-end: full pipeline → DataFrame
"""
from __future__ import annotations

import pytest

from src.data.constants import EVENT_DEPTH_UPDATE, EVENT_TRADE
from tests.conftest import (
    make_depth_payload, make_trade_payload,
    replay_to_snapshots, replay_to_labels, replay_to_dataset,
)


# --- helpers -----------------------------------------------------------------

def _depth(recv_ts, u, bids, asks, *, U=None):
    return (recv_ts, EVENT_DEPTH_UPDATE, make_depth_payload(u, bids, asks, U=U))


def _trade(recv_ts, price, qty, is_buyer_maker=False):
    return (recv_ts, EVENT_TRADE, make_trade_payload(price, qty, is_buyer_maker))


# --- snapshot-level: replay → features only ----------------------------------

class TestCausalBookState:
    """Grid nodes must see book state BEFORE the triggering depth update."""

    def test_spread_from_prior_book_state(self, cfg):
        events = [
            _depth(100, 1, [("100.00", "1.00")], [("100.02", "1.00")]),
            # At t=250 ask changes to 100.04. Grid 200 must see old ask=100.02.
            _depth(250, 2, [], [("100.02", "0"), ("100.04", "1.00")]),
            _depth(350, 3, [], []),
        ]
        _, snaps, _ = replay_to_snapshots(cfg, events, warmup_s=0)
        assert len(snaps) > 0
        # Grid 200: spread = 100.02 - 100.00 = 0.02 (before t=250 update)
        assert float(snaps[0].spread) == pytest.approx(0.02)


class TestCausalTrades:
    """Only trades with recv_ts <= grid_ts are visible in features."""

    def test_future_trade_not_visible(self, cfg):
        events = [
            _depth(100, 1, [("100.00", "1.00")], [("100.01", "1.00")]),
            _trade(250, "100.00", "1.00"),  # after grid 200
            _depth(350, 2, [], []),
        ]
        _, snaps, _ = replay_to_snapshots(cfg, events, warmup_s=0)
        assert len(snaps) > 0
        assert snaps[0].buy_volume == 0

    def test_past_trade_visible(self, cfg):
        events = [
            _depth(100, 1, [("100.00", "1.00")], [("100.01", "1.00")]),
            _trade(150, "100.00", "0.50"),  # before grid 200
            _depth(250, 2, [], []),
            _depth(350, 3, [], []),
        ]
        _, snaps, _ = replay_to_snapshots(cfg, events, warmup_s=0)
        # Grid 100 at first depth, grid 200 at second depth event
        grid_200 = [s for s in snaps if s.timestamp == 200]
        assert len(grid_200) == 1
        assert float(grid_200[0].buy_volume) == pytest.approx(0.50)


class TestWarmup:
    def test_warmup_suppresses_snapshots(self, cfg):
        """Warmup=500ms, interval=100ms. Depth events every 100ms from t=0.

        Warmup ends at first depth with recv_ts >= 500.
        Grid starts at recv_ts - (recv_ts % 100) = 500.
        First snapshot emitted at grid 500 (before t=600 depth applied).
        No snapshots before t=500.
        """
        events = [
            _depth(0, 1, [("100.00", "1.00")], [("100.01", "1.00")]),
            *[_depth(i * 100, 1 + i, [], []) for i in range(1, 11)],
        ]
        _, snaps, _ = replay_to_snapshots(cfg, events, warmup_s=0.5)

        timestamps = [s.timestamp for s in snaps]
        assert len(timestamps) > 0
        assert min(timestamps) == 500, f"first snapshot at {min(timestamps)}, expected 500"
        assert all(t >= 500 for t in timestamps), "snapshot emitted before warmup boundary"
        assert all(t % 100 == 0 for t in timestamps), "snapshot not aligned to grid"


# --- engine-level: replay state (no output checking) -------------------------

class TestSequenceGap:
    def test_gap_triggers_reset(self, cfg):
        events = [
            _depth(100, 10, [("100.00", "1.00")], [("100.01", "1.00")]),
            _depth(200, 25, [("100.00", "1.00")], [("100.01", "1.00")], U=20),
        ]
        engine, _, _ = replay_to_snapshots(cfg, events, warmup_s=0)
        assert engine.sequence_gaps_detected == 1

    def test_stale_event_skipped(self, cfg):
        """Stale event (u < last_update_id) must be a complete no-op."""
        events = [
            _depth(100, 10, [("100.00", "1.00")], [("100.01", "1.00")]),
            # Valid update: bid qty changes to 2.00
            _depth(200, 11, [("100.00", "2.00")], [("100.01", "1.00")]),
            # Stale: u=9 < last=11. Must not change book.
            _depth(300, 9, [("100.00", "9.00")], [("100.01", "5.00")], U=9),
            _depth(400, 12, [], []),
        ]
        engine, _, book = replay_to_snapshots(cfg, events, warmup_s=0)
        # Book must reflect valid update (qty=200), not stale (qty=900)
        assert book.bids[10000] == 200
        assert book.asks[10001] == 100  # ask unchanged by stale event
        assert engine.sequence_gaps_detected == 0


class TestCrossedBook:
    def test_crossed_book_resets(self, cfg):
        """Crossed book must reset pipeline. Old state must not leak."""
        events = [
            # Initial: bid=100.00/5, ask=100.01/1
            _depth(100, 1, [("100.00", "5.00")], [("100.01", "1.00")]),
            # Crossed: bid=100.02 > ask=100.01 → reset clears book
            _depth(200, 2, [("100.02", "1.00")], []),
            # After reset, new sequence rebuilds from scratch
            _depth(300, 10, [("99.00", "1.00")], [("99.01", "1.00")]),
        ]
        engine, snaps, book = replay_to_snapshots(cfg, events, warmup_s=0)
        assert not book.is_crossed()
        # Old bid=100.00/5 must not survive reset
        assert 10000 not in book.bids  # 100.00 gone
        assert 10002 not in book.bids  # 100.02 gone
        # Book reflects only post-reset state
        assert book.best_bid() == 9900   # 99.00
        assert book.best_ask() == 9901   # 99.01


# --- label-level: replay → labels (no dataset filtering) ---------------------

class TestLabelEmission:
    def test_label_produced_after_horizon(self, cfg):
        """Horizon=200ms, interval=100ms. Grid nodes: 100..700.

        Labels pair each grid node with the one 200ms later:
        (100→300), (200→400), (300→500), (400→600), (500→700).
        """
        events = [
            _depth(100, 1, [("100.00", "1.00")], [("100.02", "1.00")]),
            *[_depth(100 + i * 100, 1 + i, [], []) for i in range(1, 8)],
        ]
        _, labelled, _ = replay_to_labels(cfg, events, warmup_s=0, horizon=200)
        labeled_ts = [ls.snapshot.timestamp for ls in labelled]
        assert labeled_ts == [100, 200, 300, 400, 500]

    def test_label_value_is_midprice_delta(self, cfg):
        events = [
            _depth(100, 1, [("100.00", "1.00")], [("100.02", "1.00")]),
            # At t=150 ask changes → mid changes for future grid nodes
            _depth(150, 2, [], [("100.02", "0"), ("100.04", "1.00")]),
            _depth(250, 3, [], []),
            _depth(350, 4, [], []),
            _depth(450, 5, [], []),
        ]
        _, labelled, _ = replay_to_labels(cfg, events, warmup_s=0, horizon=200)
        assert len(labelled) > 0
        ls = labelled[0]
        # Grid 100: mid=100.01 (before t=150 update). Grid 300: mid=100.02 (after update).
        # label = 100.02 - 100.01 = 0.01
        assert float(ls.label) == pytest.approx(0.01)


# --- end-to-end: full pipeline → dataset -------------------------------------

class TestEndToEnd:
    def test_full_pipeline_exact_values(self, cfg):
        """Verify exact feature values and label through the full pipeline.

        t=100: book bid=100.00, ask=100.02 → mid=100.01
        t=150: trade buy 0.50
        t=250: depth no-op
        t=350: depth changes ask to 100.04 → mid=100.02

        Grid 200 (before t=250): spread=0.02, buy_vol=0.50, mid=100.01
        Grid 400 (after t=350):  mid=100.02
        Label at 200: mid(400) - mid(200) = 0.01
        """
        events = [
            _depth(100, 1, [("100.00", "1.00")], [("100.02", "1.00")]),
            _trade(150, "100.01", "0.50"),
            _depth(250, 2, [], []),
            _depth(350, 3, [], [("100.02", "0"), ("100.04", "1.00")]),
            _depth(450, 4, [], []),
            _depth(550, 5, [], []),
            _depth(650, 6, [], []),
        ]
        _, db, _ = replay_to_dataset(cfg, events, warmup_s=0, interval=100, horizon=200)
        df = db.to_dataframe()
        assert len(df) > 0

        row = df.set_index("timestamp").loc[200]
        assert row["spread"] == pytest.approx(0.02)
        assert row["buy_volume"] == pytest.approx(0.50)
        assert row["label"] == pytest.approx(0.01)

    def test_output_integrity(self, cfg):
        events = [
            _depth(100, 1, [("100.00", "1.00")], [("100.01", "1.00")]),
            *[_depth(100 + i * 100, 1 + i, [], []) for i in range(1, 6)],
        ]
        _, db, _ = replay_to_dataset(cfg, events, warmup_s=0, interval=100, horizon=200)
        df = db.to_dataframe()
        assert len(df) > 1
        diffs = df["timestamp"].diff().dropna()
        assert (diffs > 0).all()
        assert (diffs % 100 == 0).all()
        assert not df.isnull().any().any()

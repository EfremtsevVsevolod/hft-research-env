"""Causal contract tests for ReplayEngine.

Tests are split by level:
  - Bootstrap: snapshot initialization and sync validation
  - Snapshot-level: replay → FeatureSnapshots (no labeling/dataset)
  - Engine-level: replay state (gaps, resets, book)
  - Label-level: replay → LabelledSnapshots
  - End-to-end: full pipeline → DataFrame
"""
from __future__ import annotations

import pytest

from src.data.constants import EVENT_DEPTH_SNAPSHOT, EVENT_DEPTH_UPDATE, EVENT_TRADE
from tests.conftest import (
    make_depth_payload, make_snapshot_payload, make_trade_payload, snap_at,
    replay_to_snapshots, replay_to_labels, replay_to_dataset,
)


# --- helpers -----------------------------------------------------------------

def _snapshot(recv_ts, last_update_id, bids, asks):
    return (recv_ts, EVENT_DEPTH_SNAPSHOT,
            make_snapshot_payload(last_update_id, bids, asks))


def _depth(recv_ts, u, bids, asks, *, U=None):
    return (recv_ts, EVENT_DEPTH_UPDATE, make_depth_payload(u, bids, asks, U=U))


def _trade(recv_ts, price, qty, is_buyer_maker=False):
    return (recv_ts, EVENT_TRADE, make_trade_payload(price, qty, is_buyer_maker))


# Default snapshot for convenience
def _default_snapshot(recv_ts=50, last_update_id=0):
    return _snapshot(recv_ts, last_update_id,
                     [("100.00", "1.00")], [("100.02", "1.00")])


# --- bootstrap: snapshot initialization and sync ----------------------------

class TestBootstrap:
    def test_snapshot_initializes_book(self, cfg):
        """Snapshot loads bids/asks into book correctly."""
        events = [
            _snapshot(50, 10, [("100.00", "5.00"), ("99.99", "3.00")],
                      [("100.01", "2.00"), ("100.02", "4.00")]),
            _depth(100, 11, [], [], U=11),
            _depth(200, 12, [], []),
        ]
        _, snaps, book = replay_to_snapshots(cfg, events)
        assert book.best_bid() == 10000   # 100.00
        assert book.best_ask() == 10001   # 100.01
        assert book.bids[10000] == 500    # 5.00
        assert book.bids[9999] == 300     # 99.99 → 3.00
        assert book.asks[10001] == 200    # 100.01 → 2.00
        assert book.asks[10002] == 400    # 100.02 → 4.00

    def test_diffs_before_snapshot_ignored(self, cfg):
        """Depth events before snapshot must be skipped entirely."""
        events = [
            _depth(50, 1, [("50.00", "1.00")], [("50.01", "1.00")]),
            _depth(60, 2, [("50.00", "2.00")], [("50.01", "2.00")]),
            _snapshot(100, 5, [("100.00", "1.00")], [("100.01", "1.00")]),
            _depth(200, 6, [], [], U=6),
            _depth(300, 7, [], []),
        ]
        _, _, book = replay_to_snapshots(cfg, events)
        # Book should have snapshot state, not the early depth state
        assert book.best_bid() == 10000
        assert book.best_ask() == 10001
        assert 5000 not in book.bids  # 50.00 never applied

    def test_trades_before_snapshot_ignored(self, cfg):
        """Trades before snapshot must not be buffered."""
        events = [
            _trade(50, "100.00", "1.00"),
            _snapshot(100, 5, [("100.00", "1.00")], [("100.01", "1.00")]),
            _depth(200, 6, [], [], U=6),
            _depth(300, 7, [], []),
        ]
        _, snaps, _ = replay_to_snapshots(cfg, events)
        # Grid at 200: trade at t=50 should not appear (before bootstrap)
        s = snap_at(snaps, 200)
        assert s.buy_volume == 0

    def test_first_diff_must_overlap_snapshot(self, cfg):
        """First diff after snapshot must satisfy U <= lastUpdateId+1 <= u."""
        events = [
            # Snapshot with lastUpdateId=10
            _snapshot(50, 10, [("100.00", "1.00")], [("100.01", "1.00")]),
            # Valid first diff: U=10, u=12 → 10 <= 11 <= 12 ✓
            _depth(100, 12, [], [], U=10),
            _depth(200, 13, [], []),
        ]
        engine, _, book = replay_to_snapshots(cfg, events)
        assert engine._bootstrapped
        assert engine.sequence_gaps_detected == 0

    def test_first_diff_overlap_fails_resets(self, cfg):
        """If first diff doesn't overlap snapshot, reset to un-bootstrapped."""
        events = [
            _snapshot(50, 10, [("100.00", "1.00")], [("100.01", "1.00")]),
            # Invalid: U=15 > lastUpdateId+1=11 → sync fails
            _depth(100, 20, [], [], U=15),
        ]
        engine, _, _ = replay_to_snapshots(cfg, events)
        assert not engine._bootstrapped

    def test_stale_diffs_dropped_after_snapshot(self, cfg):
        """Diffs with u <= lastUpdateId are dropped (stale from before snapshot)."""
        events = [
            _snapshot(50, 10, [("100.00", "1.00")], [("100.01", "1.00")]),
            # These diffs have u <= 10, should be dropped
            _depth(60, 8, [("99.00", "1.00")], [], U=8),
            _depth(70, 10, [("98.00", "1.00")], [], U=9),
            # Valid first diff after snapshot
            _depth(100, 12, [], [], U=10),
            _depth(200, 13, [], []),
        ]
        _, _, book = replay_to_snapshots(cfg, events)
        # Stale diffs never applied
        assert 9900 not in book.bids
        assert 9800 not in book.bids
        assert book.best_bid() == 10000

    def test_gap_after_bootstrap_resets(self, cfg):
        """Gap in sequence after bootstrap → _bootstrapped=False."""
        events = [
            _snapshot(50, 5, [("100.00", "1.00")], [("100.01", "1.00")]),
            _depth(100, 6, [], [], U=6),
            _depth(200, 7, [], []),
            # Gap: expected U <= 8, got U=20
            _depth(300, 25, [], [], U=20),
        ]
        engine, _, _ = replay_to_snapshots(cfg, events)
        assert engine.sequence_gaps_detected == 1
        assert not engine._bootstrapped

    def test_re_bootstrap_from_new_snapshot(self, cfg):
        """Gap → new snapshot → clean recovery."""
        events = [
            _snapshot(50, 5, [("100.00", "1.00")], [("100.01", "1.00")]),
            _depth(100, 6, [], [], U=6),
            _depth(200, 7, [], []),
            # Gap
            _depth(300, 25, [], [], U=20),
            # New snapshot after gap
            _snapshot(350, 30, [("99.00", "1.00")], [("99.01", "1.00")]),
            _depth(400, 31, [], [], U=31),
            _depth(500, 32, [], []),
        ]
        engine, _, book = replay_to_snapshots(cfg, events)
        assert engine._bootstrapped
        assert engine._bootstrap_count == 2
        assert book.best_bid() == 9900   # 99.00
        assert book.best_ask() == 9901   # 99.01


# --- snapshot-level: replay → features only ----------------------------------

class TestCausalBookState:
    """Grid nodes must see book state BEFORE the triggering depth update."""

    def test_spread_from_prior_book_state(self, cfg):
        events = [
            _default_snapshot(50, 0),
            _depth(100, 1, [], [], U=1),
            # At t=250 ask changes to 100.04. Grid 200 must see old ask=100.02.
            _depth(250, 2, [], [("100.02", "0"), ("100.04", "1.00")]),
            _depth(350, 3, [], []),
        ]
        _, snaps, _ = replay_to_snapshots(cfg, events)
        s = snap_at(snaps, 200)
        assert float(s.spread) == pytest.approx(0.02)

    def test_multiple_grid_nodes_between_depth_updates(self, cfg):
        """Depth at t=100, next at t=350. Grid nodes 200 and 300 must both
        see the same book state (from t=100, before t=350 update).
        """
        events = [
            _default_snapshot(50, 0),
            _depth(100, 1, [], [], U=1),
            # Gap: no depth until t=350. Grid 200 and 300 emitted with old book.
            _depth(350, 2, [], [("100.02", "0"), ("100.04", "1.00")]),
            _depth(450, 3, [], []),
        ]
        _, snaps, _ = replay_to_snapshots(cfg, events)
        emitted_ts = [s.timestamp for s in snaps]
        assert 200 in emitted_ts
        assert 300 in emitted_ts
        between = [t for t in emitted_ts if 100 < t < 350]
        assert between == [200, 300]

        s200 = snap_at(snaps, 200)
        s300 = snap_at(snaps, 300)
        assert float(s200.spread) == pytest.approx(0.02)
        assert float(s300.spread) == pytest.approx(0.02)
        assert s200.midprice == s300.midprice
        assert s200.imbalance_1 == s300.imbalance_1


class TestCausalTrades:
    """Only trades with recv_ts <= grid_ts are visible in features."""

    def test_future_trade_not_visible(self, cfg):
        events = [
            _default_snapshot(50, 0),
            _depth(100, 1, [("100.00", "1.00")], [("100.01", "1.00")], U=1),
            _trade(250, "100.00", "1.00"),  # after grid 200
            _depth(350, 2, [], []),
        ]
        _, snaps, _ = replay_to_snapshots(cfg, events)
        s = snap_at(snaps, 200)
        assert s.buy_volume == 0

    def test_past_trade_visible(self, cfg):
        events = [
            _default_snapshot(50, 0),
            _depth(100, 1, [("100.00", "1.00")], [("100.01", "1.00")], U=1),
            _trade(150, "100.00", "0.50"),  # before grid 200
            _depth(250, 2, [], []),
            _depth(350, 3, [], []),
        ]
        _, snaps, _ = replay_to_snapshots(cfg, events)
        s = snap_at(snaps, 200)
        assert float(s.buy_volume) == pytest.approx(0.50)


# --- engine-level: replay state (no output checking) -------------------------

class TestSequenceGap:
    def test_gap_triggers_reset(self, cfg):
        events = [
            _default_snapshot(50, 0),
            _depth(100, 10, [("100.00", "1.00")], [("100.01", "1.00")], U=1),
            _depth(200, 25, [("100.00", "1.00")], [("100.01", "1.00")], U=20),
        ]
        engine, _, _ = replay_to_snapshots(cfg, events)
        assert engine.sequence_gaps_detected == 1

    def test_stale_event_skipped(self, cfg):
        """Stale event (u < last_update_id) must be a complete no-op:
        no book change, no reset, no extra snapshots, no gap counter change.
        """
        events_before_stale = [
            _default_snapshot(50, 0),
            _depth(100, 10, [("100.00", "1.00")], [("100.01", "1.00")], U=1),
            _depth(200, 11, [("100.00", "2.00")], [("100.01", "1.00")]),
        ]
        events_with_stale = [
            *events_before_stale,
            # Stale: u=9 < last=11. Must not change book.
            _depth(300, 9, [("100.00", "9.00")], [("100.01", "5.00")], U=9),
            _depth(400, 12, [], []),
        ]
        # Run without stale event to get baseline snapshot count
        _, snaps_baseline, _ = replay_to_snapshots(cfg, [
            *events_before_stale, _depth(400, 12, [], []),
        ])

        engine, snaps, book = replay_to_snapshots(cfg, events_with_stale)
        # Book unchanged by stale
        assert book.bids[10000] == 200
        assert book.asks[10001] == 100
        # No reset, no gaps
        assert engine.sequence_gaps_detected == 0
        # Stale event must not produce extra snapshots
        assert len(snaps) == len(snaps_baseline)


class TestCrossedBook:
    def test_crossed_book_resets(self, cfg):
        """Crossed book must reset pipeline. Old state must not leak."""
        events = [
            _default_snapshot(50, 0),
            _depth(100, 1, [("100.00", "5.00")], [("100.01", "1.00")], U=1),
            # Crossed: bid=100.02 > ask=100.01 → reset
            _depth(200, 2, [("100.02", "1.00")], []),
            # After reset, new snapshot needed
            _snapshot(250, 10, [("99.00", "1.00")], [("99.01", "1.00")]),
            _depth(300, 11, [], [], U=11),
        ]
        _, _, book = replay_to_snapshots(cfg, events)
        assert not book.is_crossed()
        assert 10000 not in book.bids   # old 100.00 gone
        assert 10002 not in book.bids   # old 100.02 gone
        assert book.best_bid() == 9900  # 99.00
        assert book.best_ask() == 9901  # 99.01


# --- label-level: replay → labels (no dataset filtering) ---------------------

class TestLabelEmission:
    def test_label_produced_after_horizon(self, cfg):
        """Horizon=200ms, interval=100ms. Grid nodes: 100..800.

        Labels pair each grid node with the one 200ms later:
        (100→300), (200→400), (300→500), (400→600), (500→700).
        """
        events = [
            _default_snapshot(50, 0),
            _depth(100, 1, [], [], U=1),
            *[_depth(100 + i * 100, 1 + i, [], []) for i in range(1, 8)],
        ]
        _, labelled, _ = replay_to_labels(cfg, events, horizon=200)
        labeled_ts = [ls.snapshot.timestamp for ls in labelled]
        assert labeled_ts == [100, 200, 300, 400, 500]

    def test_label_value_is_midprice_delta(self, cfg):
        events = [
            _default_snapshot(50, 0),
            _depth(100, 1, [], [], U=1),
            _depth(150, 2, [], [("100.02", "0"), ("100.04", "1.00")]),
            _depth(250, 3, [], []),
            _depth(350, 4, [], []),
            _depth(450, 5, [], []),
        ]
        _, labelled, _ = replay_to_labels(cfg, events, horizon=200)
        assert len(labelled) > 0
        # Grid 100: mid=100.01 (before t=150). Grid 300: mid=100.02 (after).
        assert float(labelled[0].label) == pytest.approx(0.01)


# --- end-to-end: full pipeline → dataset -------------------------------------

class TestEndToEnd:
    def test_full_pipeline_exact_values(self, cfg):
        """Verify exact feature values and label through the full pipeline.

        Snapshot at t=50: bid=100.00, ask=100.02 → mid=100.01
        t=100: first depth (no-op, sets grid)
        t=150: trade buy 0.50
        t=250: depth no-op
        t=350: depth changes ask to 100.04 → mid=100.02

        Grid 200 (before t=250): spread=0.02, buy_vol=0.50
        Label at 200: mid(400) - mid(200) = 100.02 - 100.01 = 0.01
        """
        events = [
            _default_snapshot(50, 0),
            _depth(100, 1, [], [], U=1),
            _trade(150, "100.01", "0.50"),
            _depth(250, 2, [], []),
            _depth(350, 3, [], [("100.02", "0"), ("100.04", "1.00")]),
            _depth(450, 4, [], []),
            _depth(550, 5, [], []),
            _depth(650, 6, [], []),
        ]
        _, db, _ = replay_to_dataset(cfg, events, interval=100, horizon=200)
        df = db.to_dataframe()
        assert 200 in df["timestamp"].values
        row = df.set_index("timestamp").loc[200]
        assert row["spread"] == pytest.approx(0.02)
        assert row["buy_volume"] == pytest.approx(0.50)
        assert row["label"] == pytest.approx(0.01)

    def test_output_integrity(self, cfg):
        events = [
            _default_snapshot(50, 0),
            _depth(100, 1, [("100.00", "1.00")], [("100.01", "1.00")], U=1),
            *[_depth(100 + i * 100, 1 + i, [], []) for i in range(1, 6)],
        ]
        _, db, _ = replay_to_dataset(cfg, events, interval=100, horizon=200)
        df = db.to_dataframe()
        assert len(df) > 1
        diffs = df["timestamp"].diff().dropna()
        assert (diffs > 0).all()
        assert (diffs % 100 == 0).all()
        assert not df.isnull().any().any()

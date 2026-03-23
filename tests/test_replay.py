"""Causal contract tests for ReplayEngine.

Uses temporary parquet files with synthetic events.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from src.data.constants import EVENT_DEPTH_UPDATE, EVENT_TRADE
from src.dataset.dataset import DatasetBuilder
from src.lob.features import FeatureExtractor
from src.lob.labels import LabelBuilder
from src.lob.orderbook import OrderBook
from src.replay.replay_engine import ReplayEngine
from tests.conftest import make_depth_payload, make_trade_payload, write_test_parquet


# --- helpers -----------------------------------------------------------------

def _build_pipeline(cfg, *, interval=100, horizon=200):
    book = OrderBook(cfg)
    fe = FeatureExtractor(sampling_interval_ms=interval, trade_window_ms=1000)
    lb = LabelBuilder(horizon_ms=horizon, sampling_interval_ms=interval)
    db = DatasetBuilder()
    return book, fe, lb, db


def _run_replay(cfg, data_path, *, interval=100, horizon=200, warmup_s=0):
    book, fe, lb, db = _build_pipeline(cfg, interval=interval, horizon=horizon)
    engine = ReplayEngine(
        data_path=data_path,
        order_book=book,
        feature_extractor=fe,
        label_builder=lb,
        dataset_builder=db,
        warmup_seconds=warmup_s,
    )
    engine.run()
    return engine, db, book


def _depth(recv_ts, u, bids, asks, *, U=None):
    return (recv_ts, EVENT_DEPTH_UPDATE, make_depth_payload(u, bids, asks, U=U))


def _trade(recv_ts, price, qty, is_buyer_maker=False):
    return (recv_ts, EVENT_TRADE, make_trade_payload(price, qty, is_buyer_maker))


# --- causal book state -------------------------------------------------------

class TestCausalBookState:
    def test_grid_uses_book_before_depth_update(self, cfg, tmp_path):
        """Grid node at t=200 must see book state BEFORE the t=250 depth update."""
        events = [
            _depth(100, 1, [("100.00", "1.00")], [("100.02", "1.00")]),
            # This update changes ask to 100.04, arrives at t=250
            _depth(250, 2, [], [("100.02", "0"), ("100.04", "1.00")]),
            _depth(350, 3, [], []),
            _depth(450, 4, [], []),
            _depth(550, 5, [], []),
        ]
        data_path = write_test_parquet(tmp_path, events)
        engine, db, book = _run_replay(cfg, data_path, warmup_s=0)

        df = db.to_dataframe()
        assert len(df) > 0, "no rows produced"
        first_row = df.iloc[0]
        # Grid node 200: bid=100.00 ask=100.02 → spread=0.02 (before t=250 update)
        assert first_row["spread"] == pytest.approx(0.02)


class TestCausalTrades:
    def test_only_past_trades_visible(self, cfg, tmp_path):
        """Trade at t=250 must not be visible at grid node t=200."""
        events = [
            _depth(100, 1, [("100.00", "1.00")], [("100.01", "1.00")]),
            _trade(250, "100.00", "1.00", is_buyer_maker=False),
            _depth(350, 2, [], []),
            _depth(450, 3, [], []),
            _depth(550, 4, [], []),
        ]
        data_path = write_test_parquet(tmp_path, events)
        engine, db, book = _run_replay(cfg, data_path, warmup_s=0)

        df = db.to_dataframe()
        assert len(df) > 0, "no rows produced"
        first_row = df.iloc[0]  # grid node 200
        assert first_row["buy_volume"] == 0.0

    def test_past_trade_visible(self, cfg, tmp_path):
        """Trade at t=150 must be visible at grid node t=200."""
        events = [
            _depth(100, 1, [("100.00", "1.00")], [("100.01", "1.00")]),
            _trade(150, "100.00", "0.50", is_buyer_maker=False),
            _depth(350, 2, [], []),
            _depth(450, 3, [], []),
            _depth(550, 4, [], []),
        ]
        data_path = write_test_parquet(tmp_path, events)
        engine, db, book = _run_replay(cfg, data_path, warmup_s=0)

        df = db.to_dataframe()
        assert len(df) > 0, "no rows produced"
        first_row = df.iloc[0]  # grid node 200
        assert first_row["buy_volume"] == pytest.approx(0.50)


class TestSequenceGap:
    def test_gap_triggers_reset(self, cfg, tmp_path):
        events = [
            _depth(100, 10, [("100.00", "1.00")], [("100.01", "1.00")]),
            # Gap: expected U=11, got U=20
            _depth(200, 25, [("100.00", "1.00")], [("100.01", "1.00")], U=20),
        ]
        data_path = write_test_parquet(tmp_path, events)
        engine, db, book = _run_replay(cfg, data_path, warmup_s=0)
        assert engine.sequence_gaps_detected == 1

    def test_stale_event_skipped(self, cfg, tmp_path):
        events = [
            _depth(100, 10, [("100.00", "1.00")], [("100.01", "1.00")]),
            _depth(200, 11, [("100.00", "2.00")], [("100.01", "1.00")]),
            # Stale: u=9 < last_update_id=11
            _depth(300, 9, [("100.00", "9.00")], [("100.01", "1.00")], U=9),
            _depth(400, 12, [], []),
            _depth(500, 13, [], []),
            _depth(600, 14, [], []),
        ]
        data_path = write_test_parquet(tmp_path, events)
        engine, db, book = _run_replay(cfg, data_path, warmup_s=0)
        assert book.bids.get(10000) != 900
        assert engine.sequence_gaps_detected == 0


class TestCrossedBook:
    def test_crossed_book_resets(self, cfg, tmp_path):
        events = [
            _depth(100, 1, [("100.00", "1.00")], [("100.01", "1.00")]),
            # bid=100.02 > ask=100.01 → crossed
            _depth(200, 2, [("100.02", "1.00")], []),
            # After reset, new sequence
            _depth(300, 10, [("100.00", "1.00")], [("100.01", "1.00")]),
        ]
        data_path = write_test_parquet(tmp_path, events)
        engine, db, book = _run_replay(cfg, data_path, warmup_s=0)
        assert not book.is_crossed()


class TestWarmup:
    def test_warmup_suppresses_features(self, cfg, tmp_path):
        events = [_depth(i * 100, i + 1, [], []) for i in range(11)]
        # Set initial book in first event
        events[0] = _depth(0, 1, [("100.00", "1.00")], [("100.01", "1.00")])
        data_path = write_test_parquet(tmp_path, events)

        engine_no, db_no, _ = _run_replay(cfg, data_path, warmup_s=0, horizon=200)
        engine_wu, db_wu, _ = _run_replay(cfg, data_path, warmup_s=1, horizon=200)

        assert len(db_wu) < len(db_no)


class TestOutputIntegrity:
    def test_timestamps_strictly_increasing_and_on_grid(self, cfg, tmp_path):
        events = [
            _depth(100, 1, [("100.00", "1.00")], [("100.01", "1.00")]),
            *[_depth(100 + i * 100, 1 + i, [], []) for i in range(1, 6)],
        ]
        data_path = write_test_parquet(tmp_path, events)
        engine, db, book = _run_replay(cfg, data_path, warmup_s=0, interval=100)

        df = db.to_dataframe()
        assert len(df) > 1, "need multiple rows to check ordering"
        diffs = df["timestamp"].diff().dropna()
        assert (diffs > 0).all(), "timestamps not strictly increasing"
        assert (diffs % 100 == 0).all(), "timestamps not on 100ms grid"


class TestEndToEnd:
    def test_full_pipeline_exact_values(self, cfg, tmp_path):
        """End-to-end with exact expected values.

        Timeline:
          t=100: book initialized: bid=100.00/1, ask=100.02/1
                 mid=100.01, spread=0.02
          t=150: trade buy 0.50 @ 100.01
          t=250: depth no-op
          t=350: depth changes ask 100.02→100.04
                 grid node 200 emitted BEFORE t=250 update:
                   book: bid=100.00, ask=100.02, spread=0.02
                   trade at 150 visible (150 <= 200): buy_volume=0.50
                 grid node 300 emitted BEFORE t=350 update:
                   book still: bid=100.00, ask=100.02 (t=350 not yet applied)
          t=450,550: more depth for labels

        Label at grid 200: mid(200)=100.01, mid(400)=?
          At grid 400: book has t=350 update applied → ask=100.04, mid=100.02
          label = 100.02 - 100.01 = 0.01
        """
        events = [
            _depth(100, 1, [("100.00", "1.00")], [("100.02", "1.00")]),
            _trade(150, "100.01", "0.50", is_buyer_maker=False),
            _depth(250, 2, [], []),
            _depth(350, 3, [], [("100.02", "0"), ("100.04", "1.00")]),
            _depth(450, 4, [], []),
            _depth(550, 5, [], []),
            _depth(650, 6, [], []),
        ]
        data_path = write_test_parquet(tmp_path, events)
        engine, db, book = _run_replay(cfg, data_path, warmup_s=0, interval=100, horizon=200)

        df = db.to_dataframe()
        assert len(df) > 0, "no dataset rows produced"

        # Grid alignment
        assert (df["timestamp"] % 100 == 0).all()
        assert (df["timestamp"].diff().dropna() > 0).all()
        assert not df.isnull().any().any()

        # First labeled row: grid node 200
        row = df.iloc[0]
        assert row["timestamp"] == 200

        # Spread at grid 200: bid=100.00, ask=100.02 (before t=250 update)
        assert row["spread"] == pytest.approx(0.02)

        # Trade at 150 visible at grid 200
        assert row["buy_volume"] == pytest.approx(0.50)

        # Label: mid(400) - mid(200) = 100.02 - 100.01 = 0.01
        assert row["label"] == pytest.approx(0.01)

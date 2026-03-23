from __future__ import annotations

from decimal import Decimal

import pytest

from src.lob.features import FeatureExtractor, Trade
from tests.conftest import make_snapshot


class TestGridEmission:
    def test_snapshot_on_grid_node(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "1.00")],
        ))
        fe = FeatureExtractor(sampling_interval_ms=100)
        snap = fe.on_book_update(1000, book)
        assert snap is not None
        assert snap.timestamp == 1000

    def test_no_snapshot_between_nodes(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "1.00")],
        ))
        fe = FeatureExtractor(sampling_interval_ms=100)
        fe.on_book_update(1000, book)  # first grid node
        snap = fe.on_book_update(1050, book)  # between nodes
        assert snap is None

    def test_monotonic_timestamp_violation(self, book):
        fe = FeatureExtractor(sampling_interval_ms=100)
        fe.on_book_update(1000, book)
        with pytest.raises(AssertionError, match="not monotonic"):
            fe.on_book_update(999, book)

    def test_missed_grid_node_violation(self, book):
        fe = FeatureExtractor(sampling_interval_ms=100)
        fe.on_book_update(1000, book)
        with pytest.raises(AssertionError, match="missed grid node"):
            fe.on_book_update(1200, book)  # skipped 1100


class TestTradeWindow:
    def test_trade_included_in_window(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "1.00")],
        ))
        fe = FeatureExtractor(sampling_interval_ms=100, trade_window_ms=200)
        trades = [Trade(timestamp=950, price=Decimal("100.00"),
                        quantity=Decimal("0.50"), is_buyer_maker=False)]
        snap = fe.on_book_update(1000, book, trades)
        assert snap.buy_volume == Decimal("0.50")

    def test_trade_evicted_after_window(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "1.00")],
        ))
        fe = FeatureExtractor(sampling_interval_ms=100, trade_window_ms=200)
        old_trade = [Trade(timestamp=700, price=Decimal("100.00"),
                           quantity=Decimal("1.00"), is_buyer_maker=False)]
        fe.on_book_update(1000, book, old_trade)
        # At t=1100, trade at 700 is outside window (1100-200=900 > 700)
        snap = fe.on_book_update(1100, book)
        assert snap.buy_volume == Decimal("0")

    def test_trade_at_window_boundary_included(self, book):
        """Trade at exactly ts - window is included (eviction uses strict <)."""
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "1.00")],
        ))
        fe = FeatureExtractor(sampling_interval_ms=100, trade_window_ms=200)
        # Trade at 800 = 1000 - 200 (exactly at boundary)
        boundary_trade = [Trade(timestamp=800, price=Decimal("100.00"),
                                quantity=Decimal("0.75"), is_buyer_maker=False)]
        snap = fe.on_book_update(1000, book, boundary_trade)
        assert snap.buy_volume == Decimal("0.75")

    def test_trade_just_before_boundary_evicted(self, book):
        """Trade at ts - window - 1 is evicted."""
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "1.00")],
        ))
        fe = FeatureExtractor(sampling_interval_ms=100, trade_window_ms=200)
        # Trade at 799 = 1000 - 200 - 1 (just outside boundary)
        old_trade = [Trade(timestamp=799, price=Decimal("100.00"),
                           quantity=Decimal("0.75"), is_buyer_maker=False)]
        snap = fe.on_book_update(1000, book, old_trade)
        assert snap.buy_volume == Decimal("0")

    def test_sell_volume_from_buyer_maker(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "1.00")],
        ))
        fe = FeatureExtractor(sampling_interval_ms=100, trade_window_ms=200)
        trades = [Trade(timestamp=950, price=Decimal("100.00"),
                        quantity=Decimal("0.30"), is_buyer_maker=True)]
        snap = fe.on_book_update(1000, book, trades)
        assert snap.sell_volume == Decimal("0.30")
        assert snap.buy_volume == Decimal("0")


class TestFeatureValues:
    def test_spread_and_midprice(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.02", "1.00")],
        ))
        fe = FeatureExtractor(sampling_interval_ms=100)
        snap = fe.on_book_update(1000, book)
        assert snap.spread == Decimal("0.02")
        assert snap.midprice == Decimal("100.01")

    def test_delta_midprice(self, book):
        fe = FeatureExtractor(sampling_interval_ms=100)

        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.02", "1.00")],
        ))
        snap1 = fe.on_book_update(1000, book)
        assert snap1.delta_midprice is None  # first snapshot, no previous

        book.apply_snapshot(make_snapshot(
            bids=[("100.04", "1.00")], asks=[("100.06", "1.00")],
        ))
        snap2 = fe.on_book_update(1100, book)
        assert snap2.delta_midprice == Decimal("0.04")  # 100.05 - 100.01


class TestReset:
    def test_reset_clears_state(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "1.00")],
        ))
        fe = FeatureExtractor(sampling_interval_ms=100)
        fe.on_book_update(1000, book)
        fe.reset()

        # After reset, grid restarts from next event
        snap = fe.on_book_update(5000, book)
        assert snap is not None
        assert snap.timestamp == 5000
        assert snap.delta_midprice is None  # prev_midprice cleared

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


class TestObserveDepth:
    """Per-update hook contract: OFI / queue-delta / depth-count."""

    def _fe(self, interval=100):
        return FeatureExtractor(sampling_interval_ms=interval,
                                 trade_window_ms=100)

    def test_seed_emits_no_flow(self):
        fe = self._fe()
        fe.observe_depth(1000, 100_000, 50, 100_010, 40)
        # Seed only stores prev_L1. Query at same ts should return 0 everywhere.
        # We query via a book-less path: construct a minimal book to _sample.
        # Easier: inspect rolling sums directly.
        assert fe._ofi_100.value(1000) == 0.0
        assert fe._ofi_1000.value(1000) == 0.0
        assert fe._queue_delta_1000.value(1000) == 0.0
        # depth count counts the seed call itself.
        assert fe._depth_count_1000.count(1000) == 1

    def test_ofi_same_price_add_on_bid(self):
        fe = self._fe()
        fe.observe_depth(1000, 100_000, 50, 100_010, 40)  # seed
        fe.observe_depth(1010, 100_000, 60, 100_010, 40)  # +10 bid qty
        # e_b = 60-50 = +10, e_a = 0; OFI = +10
        assert fe._ofi_100.value(1010) == 10.0
        assert fe._queue_delta_1000.value(1010) == 10.0

    def test_ofi_same_price_add_on_ask(self):
        fe = self._fe()
        fe.observe_depth(1000, 100_000, 50, 100_010, 40)
        fe.observe_depth(1010, 100_000, 50, 100_010, 55)  # +15 ask qty
        # e_a = 55-40 = +15 (ask adds), OFI contribution = -15
        assert fe._ofi_100.value(1010) == -15.0
        # queue_delta bid - ask delta = 0 - 15 = -15
        assert fe._queue_delta_1000.value(1010) == -15.0

    def test_ofi_bid_price_up(self):
        fe = self._fe()
        fe.observe_depth(1000, 100_000, 50, 100_010, 40)
        fe.observe_depth(1010, 100_001, 70, 100_010, 40)  # new better bid
        # e_b = +70 (new better bid), e_a = 0 → OFI = +70
        assert fe._ofi_100.value(1010) == 70.0
        # queue_delta_diff: bid price changed so bid delta = 0, ask same = 0 → 0
        assert fe._queue_delta_1000.value(1010) == 0.0

    def test_ofi_ask_price_down(self):
        fe = self._fe()
        fe.observe_depth(1000, 100_000, 50, 100_010, 40)
        fe.observe_depth(1010, 100_000, 50, 100_009, 80)  # tighter ask
        # e_b = 0, e_a = +80 (new tighter ask) → OFI = -80
        assert fe._ofi_100.value(1010) == -80.0

    def test_ofi_bid_retreat(self):
        fe = self._fe()
        fe.observe_depth(1000, 100_000, 50, 100_010, 40)
        fe.observe_depth(1010, 99_999, 30, 100_010, 40)  # bid dropped
        # e_b = -50 (prev bid vanished), e_a = 0 → OFI = -50
        assert fe._ofi_100.value(1010) == -50.0

    def test_ofi_window_eviction(self):
        fe = self._fe()
        fe.observe_depth(1000, 100_000, 50, 100_010, 40)
        fe.observe_depth(1010, 100_000, 60, 100_010, 40)   # +10 OFI
        fe.observe_depth(1200, 100_000, 70, 100_010, 40)   # +10 OFI again
        # At t=1200 in a 100ms window, the +10 at t=1010 is evicted (1010 < 1100).
        assert fe._ofi_100.value(1200) == 10.0
        # In a 1000ms window, both are kept.
        assert fe._ofi_1000.value(1200) == 20.0

    def test_depth_update_count_window(self):
        fe = self._fe()
        fe.observe_depth(1000, 100_000, 50, 100_010, 40)  # seed
        fe.observe_depth(1050, 100_000, 51, 100_010, 40)
        fe.observe_depth(1100, 100_000, 52, 100_010, 40)
        fe.observe_depth(2500, 100_000, 53, 100_010, 40)  # far later
        # At t=2500, 1000ms cutoff = 1500 → only the t=2500 push remains.
        assert fe._depth_count_1000.count(2500) == 1

    def test_reset_clears_observe_state(self):
        fe = self._fe()
        fe.observe_depth(1000, 100_000, 50, 100_010, 40)
        fe.observe_depth(1010, 100_000, 70, 100_010, 40)
        assert fe._ofi_100.value(1010) != 0
        fe.reset()
        fe.observe_depth(2000, 100_000, 50, 100_010, 40)
        assert fe._ofi_100.value(2000) == 0
        assert fe._prev_mid_sum == 100_000 + 100_010


class TestGridFeaturesFromObserve:
    """End-to-end: observe_depth drives on_book_update snapshot values."""

    def test_time_since_last_mid_move(self, book, cfg):
        from tests.conftest import make_snapshot

        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "1.00")],
        ))
        fe = FeatureExtractor(sampling_interval_ms=100, trade_window_ms=100)
        # Seed at t=900 with the snapshot L1.
        fe.observe_depth(900, book.best_bid(), book.volume_at_best()[0],
                          book.best_ask(), book.volume_at_best()[1])
        snap1 = fe.on_book_update(1000, book)
        # At t=1000, last mid move was at seed time 900.
        assert snap1.time_since_last_mid_move_ms == 100

        # Qty change only (no mid move).
        fe.observe_depth(1050, book.best_bid(), 200,
                          book.best_ask(), book.volume_at_best()[1])
        snap2 = fe.on_book_update(1100, book)
        assert snap2.time_since_last_mid_move_ms == 200

        # Now move the best_bid — mid moves.
        book.apply_update({"bids": [("100.01", "2.00")], "asks": []})
        fe.observe_depth(1150, book.best_bid(), book.volume_at_best()[0],
                          book.best_ask(), book.volume_at_best()[1])
        snap3 = fe.on_book_update(1200, book)
        assert snap3.time_since_last_mid_move_ms == 50

    def test_signed_trade_volume(self, book):
        from tests.conftest import make_snapshot

        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "1.00")],
        ))
        fe = FeatureExtractor(sampling_interval_ms=100, trade_window_ms=200)
        trades = [
            Trade(timestamp=950, price=Decimal("100.00"),
                  quantity=Decimal("1.00"), is_buyer_maker=False),  # buy aggr.
            Trade(timestamp=960, price=Decimal("100.00"),
                  quantity=Decimal("0.30"), is_buyer_maker=True),   # sell aggr.
        ]
        snap = fe.on_book_update(1000, book, trades)
        assert snap.signed_trade_volume_1000ms == Decimal("0.70")


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

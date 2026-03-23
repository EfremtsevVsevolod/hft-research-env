from __future__ import annotations

from tests.conftest import make_snapshot


class TestOrderBookSnapshot:
    def test_snapshot_initializes_book(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00"), ("99.99", "2.00")],
            asks=[("100.01", "3.00"), ("100.02", "4.00")],
        ))
        assert book.best_bid() == 10000  # 100.00 * inv_tick(100)
        assert book.best_ask() == 10001
        assert book.spread() == 1

    def test_snapshot_replaces_previous(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("50.00", "1.00")], asks=[("50.01", "1.00")],
        ))
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "1.00")],
        ))
        assert book.best_bid() == 10000
        assert len(book.bids) == 1


class TestOrderBookUpdate:
    def test_incremental_update(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "1.00")],
        ))
        book.apply_update({"bids": [("99.99", "5.00")], "asks": []})
        assert len(book.bids) == 2
        assert book.bids[9999] == 500  # 5.00 * inv_step(100)

    def test_zero_qty_removes_level(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00"), ("99.99", "2.00")],
            asks=[("100.01", "1.00")],
        ))
        book.apply_update({"bids": [("99.99", "0")], "asks": []})
        assert len(book.bids) == 1
        assert 9999 not in book.bids

    def test_update_changes_existing_level(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "1.00")],
        ))
        book.apply_update({"bids": [("100.00", "5.00")], "asks": []})
        assert book.bids[10000] == 500


class TestOrderBookValidation:
    def test_crossed_book_detected(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.01", "1.00")], asks=[("100.00", "1.00")],
        ))
        assert book.is_crossed()

    def test_locked_book_detected(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.00", "1.00")],
        ))
        assert book.is_crossed()

    def test_valid_book_not_crossed(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "1.00")],
        ))
        assert not book.is_crossed()

    def test_empty_book_not_crossed(self, book):
        assert not book.is_crossed()

    def test_clear_empties_book(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "1.00")],
        ))
        book.clear()
        assert len(book.bids) == 0
        assert len(book.asks) == 0

    def test_validate_catches_crossed(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.01", "1.00")], asks=[("100.00", "1.00")],
        ))
        errors = book.validate()
        assert any("crossed" in e for e in errors)


class TestOrderBookImbalance:
    def test_symmetric_imbalance(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "1.00")],
        ))
        assert book.imbalance(1) == 0.0

    def test_bid_heavy_imbalance(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "3.00")], asks=[("100.01", "1.00")],
        ))
        assert book.imbalance(1) > 0

    def test_ask_heavy_imbalance(self, book):
        book.apply_snapshot(make_snapshot(
            bids=[("100.00", "1.00")], asks=[("100.01", "3.00")],
        ))
        assert book.imbalance(1) < 0

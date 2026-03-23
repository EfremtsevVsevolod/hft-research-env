"""Tests for BinanceStream sync logic.

Tests exercise _handle_message directly with a mock recorder to verify
sequence validation, recv_ts preservation, and gap detection without
needing a live WebSocket connection.
"""
from __future__ import annotations

import json

import pytest

from src.data.binance_stream import BinanceStream
from src.data.constants import EVENT_DEPTH_UPDATE, EVENT_TRADE


# --- mock recorder -----------------------------------------------------------

class MockRecorder:
    """Records all appended events for inspection."""

    def __init__(self):
        self.events: list[dict] = []

    def append(self, *, recv_ts, exchange_ts, event_type, stream, payload_json):
        self.events.append({
            "recv_ts": recv_ts,
            "exchange_ts": exchange_ts,
            "event_type": event_type,
            "stream": stream,
            "payload_json": payload_json,
        })

    def flush(self):
        pass

    def __len__(self):
        return len(self.events)


# --- helpers ------------------------------------------------------------------

def _wrap(data: dict) -> str:
    """Wrap a payload in the combined stream format that Binance WS uses."""
    return json.dumps({"stream": "test@depth@100ms", "data": data})


def _depth_msg(U: int, u: int, E: int = 1000) -> str:
    return _wrap({
        "e": "depthUpdate", "E": E, "U": U, "u": u,
        "b": [["100.00", "1.00"]], "a": [["100.01", "1.00"]],
    })


def _trade_msg(price: str = "100.00", qty: str = "0.50", E: int = 1000) -> str:
    return _wrap({
        "e": "trade", "T": E, "p": price, "q": qty, "m": False,
    })


def _make_stream() -> tuple[BinanceStream, MockRecorder]:
    rec = MockRecorder()
    stream = BinanceStream("TESTUSDT", rec)
    return stream, rec


# --- recv_ts preservation -----------------------------------------------------

class TestRecvTsPreservation:
    def test_explicit_recv_ts_used_for_buffered_message(self):
        """Buffered messages must be recorded with their original recv_ts,
        not the time they are later processed."""
        stream, rec = _make_stream()
        # Simulate synced state
        stream._synced = True
        stream._last_update_id = 10
        stream._snapshot_update_id = None

        # Process a message with explicit recv_ts (as if buffered earlier)
        stream._handle_message(_depth_msg(U=11, u=11), recv_ts=42000)

        assert len(rec.events) == 1
        assert rec.events[0]["recv_ts"] == 42000

    def test_live_message_gets_current_ts(self):
        """Live (non-buffered) messages generate recv_ts internally."""
        stream, rec = _make_stream()
        stream._synced = True
        stream._last_update_id = 10
        stream._snapshot_update_id = None

        stream._handle_message(_depth_msg(U=11, u=11))

        assert len(rec.events) == 1
        # Must be a recent wall-clock timestamp (not 0 or None)
        assert rec.events[0]["recv_ts"] > 0

    def test_buffered_event_ordering_preserved(self):
        """Multiple buffered events preserve their original recv_ts order."""
        stream, rec = _make_stream()
        stream._synced = True
        stream._last_update_id = 10
        stream._snapshot_update_id = None

        stream._handle_message(_depth_msg(U=11, u=11), recv_ts=1000)
        stream._handle_message(_trade_msg(), recv_ts=1001)
        stream._handle_message(_depth_msg(U=12, u=12), recv_ts=1002)

        assert len(rec.events) == 3
        ts = [e["recv_ts"] for e in rec.events]
        assert ts == [1000, 1001, 1002]


# --- stale diff filtering -----------------------------------------------------

class TestStaleDiffFiltering:
    def test_stale_diffs_before_snapshot_dropped(self):
        """Diffs with u <= snapshot_update_id must not be recorded."""
        stream, rec = _make_stream()
        stream._snapshot_update_id = 10
        stream._last_update_id = 10
        stream._synced = False

        # Stale: u=8 <= 10
        stream._handle_message(_depth_msg(U=7, u=8), recv_ts=100)
        # Stale: u=10 <= 10
        stream._handle_message(_depth_msg(U=9, u=10), recv_ts=200)

        assert len(rec.events) == 0

    def test_stale_diffs_after_sync_dropped(self):
        """Diffs with u < last_update_id after sync must not be recorded."""
        stream, rec = _make_stream()
        stream._synced = True
        stream._last_update_id = 20
        stream._snapshot_update_id = None

        # Stale: u=15 < 20
        stream._handle_message(_depth_msg(U=14, u=15), recv_ts=100)

        assert len(rec.events) == 0


# --- first diff overlap validation --------------------------------------------

class TestFirstDiffOverlap:
    def test_valid_first_diff_syncs(self):
        """First diff where U <= lastUpdateId+1 <= u must sync and be recorded."""
        stream, rec = _make_stream()
        stream._snapshot_update_id = 10
        stream._last_update_id = 10
        stream._synced = False

        # U=10 <= 11 <= u=12 ✓
        stream._handle_message(_depth_msg(U=10, u=12), recv_ts=100)

        assert stream._synced
        assert not stream._needs_resync
        assert stream._last_update_id == 12
        assert len(rec.events) == 1

    def test_invalid_first_diff_triggers_resync(self):
        """First diff where overlap check fails must trigger resync."""
        stream, rec = _make_stream()
        stream._snapshot_update_id = 10
        stream._last_update_id = 10
        stream._synced = False

        # U=15 > 11 → overlap fails
        stream._handle_message(_depth_msg(U=15, u=20), recv_ts=100)

        assert not stream._synced
        assert stream._needs_resync
        assert len(rec.events) == 0

    def test_exact_boundary_first_diff(self):
        """First diff where U=lastUpdateId+1 and u=lastUpdateId+1 (single update)."""
        stream, rec = _make_stream()
        stream._snapshot_update_id = 10
        stream._last_update_id = 10
        stream._synced = False

        # U=11, u=11: 11 <= 11 <= 11 ✓
        stream._handle_message(_depth_msg(U=11, u=11), recv_ts=100)

        assert stream._synced
        assert len(rec.events) == 1


# --- gap detection ------------------------------------------------------------

class TestGapDetection:
    def test_gap_triggers_resync(self):
        """U > last_update_id + 1 must set needs_resync."""
        stream, rec = _make_stream()
        stream._synced = True
        stream._last_update_id = 10
        stream._snapshot_update_id = None

        # Gap: U=15 > 11
        stream._handle_message(_depth_msg(U=15, u=20), recv_ts=100)

        assert stream._needs_resync
        assert len(rec.events) == 0

    def test_contiguous_diff_accepted(self):
        """U <= last_update_id + 1 must be accepted."""
        stream, rec = _make_stream()
        stream._synced = True
        stream._last_update_id = 10
        stream._snapshot_update_id = None

        stream._handle_message(_depth_msg(U=11, u=11), recv_ts=100)

        assert not stream._needs_resync
        assert stream._last_update_id == 11
        assert len(rec.events) == 1

    def test_events_suppressed_during_resync(self):
        """After needs_resync is set, all subsequent events are suppressed."""
        stream, rec = _make_stream()
        stream._synced = True
        stream._last_update_id = 10
        stream._snapshot_update_id = None

        # Gap triggers resync
        stream._handle_message(_depth_msg(U=15, u=20), recv_ts=100)
        assert stream._needs_resync

        # Subsequent events suppressed
        stream._handle_message(_depth_msg(U=21, u=21), recv_ts=200)
        stream._handle_message(_trade_msg(), recv_ts=300)

        assert len(rec.events) == 0


# --- trade passthrough --------------------------------------------------------

class TestTradePassthrough:
    def test_trades_recorded_when_synced(self):
        """Trades pass through without sequence validation when synced."""
        stream, rec = _make_stream()
        stream._synced = True
        stream._last_update_id = 10
        stream._snapshot_update_id = None

        stream._handle_message(_trade_msg(), recv_ts=500)

        assert len(rec.events) == 1
        assert rec.events[0]["event_type"] == EVENT_TRADE

    def test_trades_recorded_before_first_diff_sync(self):
        """Trades pass through even before the first diff is sync-validated."""
        stream, rec = _make_stream()
        stream._snapshot_update_id = 10
        stream._last_update_id = 10
        stream._synced = False

        stream._handle_message(_trade_msg(), recv_ts=500)

        assert len(rec.events) == 1
        assert rec.events[0]["event_type"] == EVENT_TRADE

"""Tests for BinanceStream sync logic.

Tests exercise _handle_message directly with a mock recorder to verify
sequence validation, recv_ts preservation, and gap detection without
needing a live WebSocket connection.

Integration tests for _do_sync use a mock WebSocket and patched REST fetch.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from src.data.binance_stream import BinanceStream
from src.data.constants import EVENT_DEPTH_SNAPSHOT, EVENT_DEPTH_UPDATE, EVENT_TRADE


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


# --- mock WebSocket for _do_sync integration tests ---------------------------

class MockWebSocket:
    """Fake WebSocket that delivers messages from a queue.

    After the queue is drained, recv() blocks until timeout (simulating
    no more messages available — like the real WS library buffer being empty).
    """

    def __init__(self, messages: list[str]):
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        for m in messages:
            self._queue.put_nowait(m)

    async def recv(self) -> str:
        return await self._queue.get()

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        try:
            return await asyncio.wait_for(self.recv(), timeout=0.05)
        except asyncio.TimeoutError:
            raise StopAsyncIteration


def _snapshot_response(last_update_id: int = 10) -> dict:
    return {
        "lastUpdateId": last_update_id,
        "bids": [["100.00", "1.00"]],
        "asks": [["100.01", "1.00"]],
    }


# --- _do_sync integration tests -----------------------------------------------

@pytest.mark.asyncio
class TestDoSync:
    async def test_buffered_messages_preserve_recv_ts(self):
        """Messages buffered before snapshot fetch keep their recv_ts."""
        ws = MockWebSocket([
            _trade_msg(),                    # buffered before first depth
            _depth_msg(U=5, u=5),            # first depth → triggers fetch
        ])
        rec = MockRecorder()
        stream = BinanceStream("TESTUSDT", rec)

        with patch("src.data.binance_stream._fetch_snapshot_sync",
                    return_value=_snapshot_response(10)):
            await stream._do_sync(ws)

        # Recorder should have: snapshot, then buffered trade + depth
        types = [e["event_type"] for e in rec.events]
        assert types[0] == EVENT_DEPTH_SNAPSHOT
        # Buffered events come after snapshot in recording
        assert EVENT_TRADE in types[1:]

        # All buffered events have recv_ts > 0 (real timestamps, not restamped)
        for e in rec.events:
            assert e["recv_ts"] > 0

    async def test_snapshot_recorded_before_buffered_diffs(self):
        """Snapshot event must appear before buffered depth diffs in recorder."""
        ws = MockWebSocket([
            _depth_msg(U=5, u=5),
            _depth_msg(U=6, u=6),
        ])
        rec = MockRecorder()
        stream = BinanceStream("TESTUSDT", rec)

        with patch("src.data.binance_stream._fetch_snapshot_sync",
                    return_value=_snapshot_response(10)):
            await stream._do_sync(ws)

        types = [e["event_type"] for e in rec.events]
        snap_idx = types.index(EVENT_DEPTH_SNAPSHOT)
        # All depth updates are after the snapshot
        depth_indices = [i for i, t in enumerate(types) if t == EVENT_DEPTH_UPDATE]
        for idx in depth_indices:
            assert idx > snap_idx

    async def test_stale_buffered_diffs_not_recorded(self):
        """Buffered diffs with u <= lastUpdateId are dropped during processing."""
        ws = MockWebSocket([
            _depth_msg(U=3, u=3),   # stale: u=3 <= lastUpdateId=10
            _depth_msg(U=8, u=9),   # stale: u=9 <= 10
            _depth_msg(U=10, u=12), # valid: overlaps snapshot
        ])
        rec = MockRecorder()
        stream = BinanceStream("TESTUSDT", rec)

        with patch("src.data.binance_stream._fetch_snapshot_sync",
                    return_value=_snapshot_response(10)):
            await stream._do_sync(ws)

        types = [e["event_type"] for e in rec.events]
        # Snapshot + 1 valid depth (stale ones dropped)
        assert types.count(EVENT_DEPTH_SNAPSHOT) == 1
        assert types.count(EVENT_DEPTH_UPDATE) == 1

    async def test_snapshot_retry_when_stale(self):
        """Snapshot is retried when lastUpdateId < first_U."""
        ws = MockWebSocket([
            _depth_msg(U=20, u=20),
        ])
        rec = MockRecorder()
        stream = BinanceStream("TESTUSDT", rec)

        call_count = 0
        def _mock_fetch(symbol):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _snapshot_response(10)  # too old: 10 < 20
            return _snapshot_response(25)      # fresh enough: 25 >= 20

        with patch("src.data.binance_stream._fetch_snapshot_sync", side_effect=_mock_fetch):
            await stream._do_sync(ws)

        assert call_count == 2
        assert stream._snapshot_update_id == 25

    async def test_trade_before_depth_preserved_in_order(self):
        """A trade arriving before the first depth is buffered and recorded."""
        ws = MockWebSocket([
            _trade_msg(price="99.50"),
            _depth_msg(U=5, u=5),
        ])
        rec = MockRecorder()
        stream = BinanceStream("TESTUSDT", rec)

        with patch("src.data.binance_stream._fetch_snapshot_sync",
                    return_value=_snapshot_response(10)):
            await stream._do_sync(ws)

        types = [e["event_type"] for e in rec.events]
        # snapshot first, then trade, then depth (stale depth dropped)
        assert types[0] == EVENT_DEPTH_SNAPSHOT
        assert EVENT_TRADE in types

    async def test_sync_state_after_do_sync(self):
        """After _do_sync, stream is ready for normal message processing."""
        ws = MockWebSocket([
            _depth_msg(U=10, u=12),  # valid first diff: 10 <= 11 <= 12
        ])
        rec = MockRecorder()
        stream = BinanceStream("TESTUSDT", rec)

        with patch("src.data.binance_stream._fetch_snapshot_sync",
                    return_value=_snapshot_response(10)):
            await stream._do_sync(ws)

        assert stream._synced
        assert stream._last_update_id == 12
        assert not stream._needs_resync

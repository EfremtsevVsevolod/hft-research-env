"""Tests for BinanceStream sync logic.

Unit tests exercise _handle_message directly with a mock recorder.
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


def _make_synced_stream(last_update_id: int = 10) -> tuple[BinanceStream, MockRecorder]:
    rec = MockRecorder()
    stream = BinanceStream("TESTUSDT", rec)
    stream._synced = True
    stream._last_update_id = last_update_id
    stream._snapshot_update_id = None
    return stream, rec


def _make_pre_sync_stream(snapshot_update_id: int = 10) -> tuple[BinanceStream, MockRecorder]:
    rec = MockRecorder()
    stream = BinanceStream("TESTUSDT", rec)
    stream._snapshot_update_id = snapshot_update_id
    stream._last_update_id = snapshot_update_id
    stream._synced = False
    return stream, rec


# --- recv_ts preservation -----------------------------------------------------

class TestRecvTsPreservation:
    def test_explicit_recv_ts_used_for_buffered_message(self):
        """Buffered messages keep their original recv_ts."""
        stream, rec = _make_synced_stream()
        stream._handle_message(_depth_msg(U=11, u=11), recv_ts=42000)
        assert rec.events[0]["recv_ts"] == 42000

    def test_live_message_gets_current_ts(self):
        """Live messages generate recv_ts internally."""
        stream, rec = _make_synced_stream()
        stream._handle_message(_depth_msg(U=11, u=11))
        assert rec.events[0]["recv_ts"] > 0

    def test_buffered_event_ordering_preserved(self):
        """Multiple buffered events preserve their original recv_ts order."""
        stream, rec = _make_synced_stream()
        stream._handle_message(_depth_msg(U=11, u=11), recv_ts=1000)
        stream._handle_message(_trade_msg(), recv_ts=1001)
        stream._handle_message(_depth_msg(U=12, u=12), recv_ts=1002)
        assert [e["recv_ts"] for e in rec.events] == [1000, 1001, 1002]


# --- stale diff filtering (parametrized) --------------------------------------

@pytest.mark.parametrize("desc,setup,U,u", [
    ("before_snapshot", "pre_sync", 7, 8),     # u=8 <= snapshot_id=10
    ("before_snapshot_boundary", "pre_sync", 9, 10),  # u=10 <= snapshot_id=10
    ("after_sync", "synced", 14, 15),           # u=15 < last_update_id=20
])
def test_stale_diff_dropped(desc, setup, U, u):
    """Stale diffs must not be recorded regardless of sync phase."""
    if setup == "pre_sync":
        stream, rec = _make_pre_sync_stream(10)
    else:
        stream, rec = _make_synced_stream(20)
    stream._handle_message(_depth_msg(U=U, u=u), recv_ts=100)
    assert len(rec.events) == 0


# --- first diff overlap validation (parametrized) ----------------------------

@pytest.mark.parametrize("desc,U,u,expect_synced", [
    ("overlap_wide", 10, 12, True),     # U=10 <= 11 <= u=12 ✓
    ("exact_boundary", 11, 11, True),   # U=11 <= 11 <= u=11 ✓
    ("gap_after_snapshot", 15, 20, False),  # U=15 > 11 → resync
])
def test_first_diff_overlap(desc, U, u, expect_synced):
    """First diff after snapshot must satisfy U <= lastUpdateId+1 <= u."""
    stream, rec = _make_pre_sync_stream(10)
    stream._handle_message(_depth_msg(U=U, u=u), recv_ts=100)
    assert stream._synced == expect_synced
    assert stream._needs_resync == (not expect_synced)
    assert len(rec.events) == (1 if expect_synced else 0)


# --- gap detection (parametrized) --------------------------------------------

@pytest.mark.parametrize("desc,U,u,expect_gap", [
    ("contiguous", 11, 11, False),   # U=11 <= 11 → OK
    ("gap", 15, 20, True),           # U=15 > 11 → gap
])
def test_sequence_continuity(desc, U, u, expect_gap):
    """Contiguous diffs accepted, gaps trigger resync."""
    stream, rec = _make_synced_stream(10)
    stream._handle_message(_depth_msg(U=U, u=u), recv_ts=100)
    assert stream._needs_resync == expect_gap
    assert len(rec.events) == (0 if expect_gap else 1)


def test_events_suppressed_during_resync():
    """After needs_resync is set, all subsequent events are suppressed."""
    stream, rec = _make_synced_stream(10)
    stream._handle_message(_depth_msg(U=15, u=20), recv_ts=100)  # gap
    stream._handle_message(_depth_msg(U=21, u=21), recv_ts=200)
    stream._handle_message(_trade_msg(), recv_ts=300)
    assert len(rec.events) == 0


# --- trade passthrough (parametrized) ----------------------------------------

@pytest.mark.parametrize("desc,setup", [
    ("synced", "synced"),
    ("pre_sync", "pre_sync"),
])
def test_trades_recorded(desc, setup):
    """Trades pass through without sequence validation in any non-resync state."""
    if setup == "synced":
        stream, rec = _make_synced_stream()
    else:
        stream, rec = _make_pre_sync_stream()
    stream._handle_message(_trade_msg(), recv_ts=500)
    assert len(rec.events) == 1
    assert rec.events[0]["event_type"] == EVENT_TRADE


# --- mock WebSocket for _do_sync integration tests ---------------------------

class MockWebSocket:
    """Fake WebSocket that delivers messages from a queue."""

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
    async def test_snapshot_before_buffered_events(self):
        """Snapshot is recorded first. Buffered trade + depth follow with recv_ts > 0."""
        ws = MockWebSocket([
            _trade_msg(),
            _depth_msg(U=5, u=5),
        ])
        rec = MockRecorder()
        stream = BinanceStream("TESTUSDT", rec)

        with patch("src.data.binance_stream._fetch_snapshot_sync",
                    return_value=_snapshot_response(10)):
            await stream._do_sync(ws)

        types = [e["event_type"] for e in rec.events]
        assert types[0] == EVENT_DEPTH_SNAPSHOT
        assert EVENT_TRADE in types[1:]
        for e in rec.events:
            assert e["recv_ts"] > 0

    async def test_stale_buffered_diffs_not_recorded(self):
        """Buffered diffs with u <= lastUpdateId are dropped."""
        ws = MockWebSocket([
            _depth_msg(U=3, u=3),   # stale
            _depth_msg(U=8, u=9),   # stale
            _depth_msg(U=10, u=12), # valid overlap
        ])
        rec = MockRecorder()
        stream = BinanceStream("TESTUSDT", rec)

        with patch("src.data.binance_stream._fetch_snapshot_sync",
                    return_value=_snapshot_response(10)):
            await stream._do_sync(ws)

        types = [e["event_type"] for e in rec.events]
        assert types.count(EVENT_DEPTH_SNAPSHOT) == 1
        assert types.count(EVENT_DEPTH_UPDATE) == 1

    async def test_snapshot_retry_when_stale(self):
        """Snapshot is retried when lastUpdateId < first_U."""
        ws = MockWebSocket([_depth_msg(U=20, u=20)])
        rec = MockRecorder()
        stream = BinanceStream("TESTUSDT", rec)

        call_count = 0
        def _mock_fetch(symbol):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _snapshot_response(10)  # too old
            return _snapshot_response(25)

        with patch("src.data.binance_stream._fetch_snapshot_sync", side_effect=_mock_fetch):
            await stream._do_sync(ws)

        assert call_count == 2
        assert stream._snapshot_update_id == 25

    async def test_sync_state_after_valid_diff(self):
        """After _do_sync with a valid first diff, stream is synced."""
        ws = MockWebSocket([_depth_msg(U=10, u=12)])
        rec = MockRecorder()
        stream = BinanceStream("TESTUSDT", rec)

        with patch("src.data.binance_stream._fetch_snapshot_sync",
                    return_value=_snapshot_response(10)):
            await stream._do_sync(ws)

        assert stream._synced
        assert stream._last_update_id == 12
        assert not stream._needs_resync

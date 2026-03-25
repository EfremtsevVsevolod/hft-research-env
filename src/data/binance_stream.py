"""Binance WebSocket stream connector (async, using ``websockets``).

Connects to combined depth + trade streams, attaches recv_ts,
and forwards raw messages to a Recorder.

Implements the Binance local order book management procedure:
1. Open WS, buffer events, note U of first depth event
2. Fetch REST snapshot; if lastUpdateId < first U, retry
3. Discard buffered events where u <= lastUpdateId
4. First remaining event must have lastUpdateId in [U, u]
5. Apply buffered events, then all subsequent events
6. On gap (U > local_update_id + 1) → restart from step 1
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.request

import websockets

from .constants import EVENT_DEPTH_SNAPSHOT, EVENT_DEPTH_UPDATE, EVENT_TRADE
from .recorder import Recorder
from src.utils import fmt_count

logger = logging.getLogger(__name__)

_BASE_URL = "wss://stream.binance.com:9443/stream"
_REST_BASE = "https://api.binance.com"


def _extract_exchange_ts(event_type: str, data: dict) -> int:
    """Pull the exchange timestamp from the raw payload."""
    if event_type == EVENT_DEPTH_UPDATE:
        return data["E"]
    if event_type == EVENT_TRADE:
        return data["T"]
    return data.get("E", 0)


def _fetch_snapshot_sync(symbol: str) -> dict:
    """Fetch order book snapshot from Binance REST API (blocking).

    Returns dict with keys: lastUpdateId, bids, asks.
    """
    url = f"{_REST_BASE}/api/v3/depth?symbol={symbol.upper()}&limit=5000"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read())
    logger.info(
        "Fetched snapshot: lastUpdateId=%d, %d bids, %d asks",
        data["lastUpdateId"], len(data["bids"]), len(data["asks"]),
    )
    return data


class BinanceStream:
    """Connect to Binance combined WebSocket streams and record raw messages.

    Implements the full Binance order book management procedure:
    snapshot at startup, sequence tracking, gap detection with re-snapshot.

    Usage::

        recorder = Recorder(Path("data/raw/binance/BTCUSDT"))
        stream = BinanceStream("btcusdt", recorder)
        asyncio.run(stream.run())
    """

    def __init__(
        self,
        symbol: str,
        recorder: Recorder,
        reconnect_delay_s: float = 5.0,
    ) -> None:
        self._symbol = symbol.lower()
        self._symbol_upper = symbol.upper()
        self._recorder = recorder
        self._reconnect_delay = reconnect_delay_s
        self._running = True
        self._stats_interval_s = 30
        self._msg_count = 0
        self._depth_count = 0
        self._trade_count = 0
        self._period_msgs = 0
        self._latency_sum = 0
        self._latency_count = 0
        self._start_ts = time.monotonic()
        self._last_stats_ts = time.monotonic()

        # Sequence tracking
        self._last_update_id: int | None = None
        self._snapshot_update_id: int | None = None
        self._synced = False
        self._needs_resync = False

    @property
    def _url(self) -> str:
        streams = f"{self._symbol}@depth@100ms/{self._symbol}@trade"
        return f"{_BASE_URL}?streams={streams}"

    async def run(self) -> None:
        """Connect and block.  Reconnects automatically on drop."""
        while self._running:
            try:
                logger.info("connecting to %s", self._url)
                async with websockets.connect(
                    self._url,
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    logger.info("connected")
                    await self._sync_and_listen(ws)
            except (
                websockets.ConnectionClosed,
                websockets.InvalidURI,
                OSError,
            ) as e:
                logger.warning("disconnected: %s", e)

            if self._running:
                self._recorder.flush()
                logger.info(
                    "reconnecting in %.1fs", self._reconnect_delay
                )
                await asyncio.sleep(self._reconnect_delay)

    def stop(self) -> None:
        """Signal the run loop to exit after the current connection closes."""
        self._running = False

    async def _do_sync(self, ws: websockets.ClientConnection) -> None:
        """Binance sync procedure: buffer → snapshot (retry if stale) → apply.

        Follows the documented procedure exactly:
        1. Buffer events, note U of first depth event
        2. Fetch snapshot; retry if lastUpdateId < first U
        3. Record snapshot, process buffered events
        """
        self._synced = False
        self._snapshot_update_id = None
        self._last_update_id = None
        self._needs_resync = False

        # Step 1: Buffer until we have at least one depth event.
        # Capture recv_ts at actual receipt time to preserve timing fidelity.
        # Note the U of the first depth event.
        buffer: list[tuple[int, str]] = []
        first_U: int | None = None
        while first_U is None:
            raw = await ws.recv()
            recv_ts = time.time_ns() // 1_000_000
            if not self._running:
                return
            buffer.append((recv_ts, raw))
            msg = json.loads(raw)
            data = msg.get("data", msg)
            if data.get("e") == EVENT_DEPTH_UPDATE:
                first_U = int(data["U"])

        # Step 2: Fetch snapshot. Retry if lastUpdateId < first_U.
        # During fetch, WS messages queue in the websockets library buffer.
        while True:
            snapshot = await asyncio.to_thread(
                _fetch_snapshot_sync, self._symbol_upper,
            )
            if snapshot["lastUpdateId"] >= first_U:
                break
            logger.info(
                "Snapshot too old (lastUpdateId=%d < first_U=%d), retrying",
                snapshot["lastUpdateId"], first_U,
            )

        # Step 3: Drain WS messages that queued during the REST fetch.
        # They get recv_ts = now (ms-level error vs hundreds of ms without drain).
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.01)
                recv_ts = time.time_ns() // 1_000_000
                buffer.append((recv_ts, raw))
            except asyncio.TimeoutError:
                break

        # Step 4: Record snapshot, set state.
        self._record_snapshot(snapshot)
        self._snapshot_update_id = snapshot["lastUpdateId"]
        self._last_update_id = snapshot["lastUpdateId"]

        # Step 5: Process all buffered messages with their original recv_ts.
        for recv_ts, raw in buffer:
            if not self._running:
                return
            self._handle_message(raw, recv_ts=recv_ts)

    async def _sync_and_listen(self, ws: websockets.ClientConnection) -> None:
        """Initial sync, then listen. Re-syncs on gap detection."""
        await self._do_sync(ws)

        async for raw in ws:
            if not self._running:
                break
            self._handle_message(raw)
            if self._needs_resync:
                await self._do_sync(ws)

    def _record_snapshot(self, snapshot_data: dict) -> None:
        """Record a depth snapshot event to the recorder."""
        recv_ts = time.time_ns() // 1_000_000
        payload_json = json.dumps(snapshot_data, separators=(",", ":"))
        self._recorder.append(
            recv_ts=recv_ts,
            exchange_ts=0,
            event_type=EVENT_DEPTH_SNAPSHOT,
            stream=f"{self._symbol}@depth_snapshot",
            payload_json=payload_json,
        )

    def _handle_message(self, raw: str, *, recv_ts: int | None = None) -> None:
        if recv_ts is None:
            recv_ts = time.time_ns() // 1_000_000

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("invalid JSON: %s", raw[:200])
            return

        # Combined stream wrapper: {"stream": "...", "data": {...}}
        stream_name = msg.get("stream", "")
        data = msg.get("data", msg)

        event_type = data.get("e", "")

        # Suppress all events while waiting for resync
        if self._needs_resync:
            return

        # Sequence validation for depth updates
        if event_type == EVENT_DEPTH_UPDATE:
            u = int(data["u"])
            U = int(data["U"])

            # Drop diffs where u <= snapshot lastUpdateId (step 5 of docs)
            if self._snapshot_update_id is not None and u <= self._snapshot_update_id:
                return

            # First diff after snapshot: lastUpdateId must be in [U, u]
            if not self._synced and self._snapshot_update_id is not None:
                if not (U <= self._snapshot_update_id + 1 <= u):
                    logger.warning(
                        "First diff sync failed: U=%d, u=%d, lastUpdateId=%d. Re-syncing.",
                        U, u, self._snapshot_update_id,
                    )
                    self._needs_resync = True
                    return
                self._synced = True
                self._last_update_id = u
            elif self._synced:
                # Stale: u < local update ID → ignore
                if u < self._last_update_id:
                    return
                # Gap: U > local update ID + 1 → missed events
                if U > self._last_update_id + 1:
                    logger.warning(
                        "Sequence gap: expected <= %d, got U=%d. Re-syncing.",
                        self._last_update_id + 1, U,
                    )
                    self._needs_resync = True
                    return
                self._last_update_id = u

        exchange_ts = _extract_exchange_ts(event_type, data)

        # Store the inner payload, not the combined wrapper.
        payload_json = json.dumps(data, separators=(",", ":"))

        self._recorder.append(
            recv_ts=recv_ts,
            exchange_ts=exchange_ts,
            event_type=event_type,
            stream=stream_name,
            payload_json=payload_json,
        )

        self._msg_count += 1
        self._period_msgs += 1
        if event_type == EVENT_DEPTH_UPDATE:
            self._depth_count += 1
        elif event_type == EVENT_TRADE:
            self._trade_count += 1
        if exchange_ts > 0:
            self._latency_sum += recv_ts - exchange_ts
            self._latency_count += 1

        now = time.monotonic()
        if now - self._last_stats_ts >= self._stats_interval_s:
            dt = now - self._last_stats_ts
            elapsed = now - self._start_ts
            h, rem = divmod(int(elapsed), 3600)
            m, s = divmod(rem, 60)
            rate = self._period_msgs / dt if dt > 0 else 0
            avg_lat = (self._latency_sum / self._latency_count
                       if self._latency_count > 0 else 0)
            logger.info(
                "%02d:%02d:%02d | %s msgs (%s depth, %s trade)"
                " | %.0f msg/s | latency %dms",
                h, m, s,
                fmt_count(self._msg_count),
                fmt_count(self._depth_count),
                fmt_count(self._trade_count),
                rate, avg_lat,
            )
            self._last_stats_ts = now
            self._period_msgs = 0
            self._latency_sum = 0
            self._latency_count = 0

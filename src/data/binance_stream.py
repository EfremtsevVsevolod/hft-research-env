"""Binance WebSocket stream connector (async, using ``websockets``).

Connects to combined depth + trade streams, attaches recv_ts,
and forwards raw messages to a Recorder.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import websockets

from .recorder import Recorder

logger = logging.getLogger(__name__)

_BASE_URL = "wss://stream.binance.com:9443/stream"


def _extract_exchange_ts(event_type: str, data: dict) -> int:
    """Pull the exchange timestamp from the raw payload."""
    if event_type == "depthUpdate":
        return data["E"]
    if event_type == "trade":
        return data["T"]
    return data.get("E", 0)


class BinanceStream:
    """Connect to Binance combined WebSocket streams and record raw messages.

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
        self._recorder = recorder
        self._reconnect_delay = reconnect_delay_s
        self._running = True
        self._stats_interval_s = 30
        self._msg_count = 0
        self._depth_count = 0
        self._trade_count = 0
        self._last_stats_ts = time.monotonic()

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
                    await self._listen(ws)
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

    async def _listen(self, ws: websockets.ClientConnection) -> None:
        async for raw in ws:
            if not self._running:
                break
            self._handle_message(raw)

    def _handle_message(self, raw: str) -> None:
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
        if event_type == "depthUpdate":
            self._depth_count += 1
        elif event_type == "trade":
            self._trade_count += 1

        now = time.monotonic()
        if now - self._last_stats_ts >= self._stats_interval_s:
            logger.info(
                "received %d msgs (depth: %d, trade: %d), buffer: %d",
                self._msg_count,
                self._depth_count,
                self._trade_count,
                len(self._recorder._buffer),
            )
            self._last_stats_ts = now

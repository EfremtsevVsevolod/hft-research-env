#!/usr/bin/env python
"""Record raw Binance WebSocket messages to Parquet files.

Usage:
    python scripts/record_binance.py BTCUSDT
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from pathlib import Path

from src.data.binance_stream import BinanceStream
from src.data.recorder import Recorder


async def run(args: argparse.Namespace) -> None:
    symbol = args.symbol.upper()
    output_dir = Path(args.output_dir or f"data/raw/binance/{symbol}")

    recorder = Recorder(
        base_dir=output_dir,
        flush_every_events=args.flush_every_events,
        flush_every_seconds=args.flush_every_seconds,
    )
    stream = BinanceStream(symbol, recorder)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: _shutdown(stream, recorder))

    await stream.run()


def _shutdown(stream: BinanceStream, recorder: Recorder) -> None:
    logging.info("shutting down...")
    stream.stop()
    recorder.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Binance market data")
    parser.add_argument("symbol", help="Trading pair, e.g. BTCUSDT")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: data/raw/binance/<SYMBOL>)",
    )
    parser.add_argument("--flush-every-events", type=int, default=20000)
    parser.add_argument("--flush-every-seconds", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(run(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build an ML dataset by replaying recorded Binance events.

Usage::

    python scripts/replay_dataset.py data/raw/binance/BTCUSDT \\
        --symbol BTCUSDT --output dataset.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config.config import load_symbols
from src.dataset.dataset import DatasetBuilder
from src.lob.features import FeatureExtractor
from src.lob.labels import LabelBuilder
from src.lob.orderbook import OrderBook
from src.replay.replay_engine import ReplayEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay raw Binance data to build ML dataset")
    parser.add_argument("data_path", type=Path, help="Directory with recorded Parquet files")
    parser.add_argument("--symbol", required=True, help="Symbol name (e.g. BTCUSDT)")
    parser.add_argument("--output", type=Path, default=Path("dataset.parquet"), help="Output Parquet path")
    parser.add_argument("--warmup", type=int, default=600, help="Warmup seconds (default: 600)")
    parser.add_argument("--interval", type=int, default=50, help="Sampling interval ms (default: 50)")
    parser.add_argument("--horizon", type=int, default=200, help="Label horizon ms (default: 200)")
    parser.add_argument("--trade-window", type=int, default=100, help="Trade window ms (default: 100)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    symbols = load_symbols()
    if args.symbol not in symbols:
        raise SystemExit(f"Symbol {args.symbol!r} not found in config/symbols.yaml")
    cfg = symbols[args.symbol]

    book = OrderBook(cfg)
    fe = FeatureExtractor(sampling_interval_ms=args.interval, trade_window_ms=args.trade_window)
    lb = LabelBuilder(horizon_ms=args.horizon, sampling_interval_ms=args.interval)
    db = DatasetBuilder()

    engine = ReplayEngine(
        data_path=args.data_path,
        order_book=book,
        feature_extractor=fe,
        label_builder=lb,
        dataset_builder=db,
        warmup_seconds=args.warmup,
    )
    engine.run()
    db.save_parquet(args.output)
    logging.getLogger(__name__).info("Saved %d rows to %s", len(db), args.output)


if __name__ == "__main__":
    main()

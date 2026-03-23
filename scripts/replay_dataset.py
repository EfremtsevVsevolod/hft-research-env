#!/usr/bin/env python3
"""Build an ML dataset by replaying recorded Binance events.

Usage:
    python scripts/replay_dataset.py data/raw/binance/BTCUSDT --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config.config import fetch_symbol_config
from src.dataset.dataset import DatasetBuilder
from src.lob.features import FeatureExtractor
from src.lob.labels import LabelBuilder
from src.lob.orderbook import OrderBook
from src.replay.replay_engine import ReplayEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay raw Binance data to build ML dataset")
    parser.add_argument("data_path", type=Path, help="Directory with recorded Parquet files")
    parser.add_argument("--symbol", required=True, help="Symbol name (e.g. BTCUSDT)")
    parser.add_argument("--output", type=Path, default=Path("data/datasets/dataset.parquet"), help="Output Parquet path")
    parser.add_argument("--warmup", type=int, default=600, help="Warmup seconds (default: 600)")
    parser.add_argument("--interval", type=int, default=100, help="Sampling interval ms (default: 100)")
    parser.add_argument("--horizon", type=int, default=200, help="Label horizon ms (default: 200)")
    parser.add_argument("--trade-window", type=int, default=1000, help="Trade window ms (default: 1000)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output file already exists: {args.output}\nUse --overwrite to replace.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = fetch_symbol_config(args.symbol)

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

    metadata = {
        "symbol": args.symbol.upper(),
        "data_path": str(args.data_path),
        "interval_ms": args.interval,
        "horizon_ms": args.horizon,
        "trade_window_ms": args.trade_window,
        "warmup_s": args.warmup,
        "tick_size": str(cfg.tick_size),
        "step_size": str(cfg.step_size),
        "rows": len(db),
        "sequence_gaps": engine.sequence_gaps_detected,
    }
    db.save_parquet(args.output, metadata=metadata)
    logging.getLogger(__name__).info("Saved %d rows to %s", len(db), args.output)


if __name__ == "__main__":
    main()

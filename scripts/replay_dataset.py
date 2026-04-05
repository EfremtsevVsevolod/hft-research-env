#!/usr/bin/env python3
"""Build an ML dataset by replaying recorded Binance events.

Usage:
    python scripts/replay_dataset.py data/raw/binance/BTCUSDT_v2 --symbol BTCUSDT --duration 4D --horizon 200 --output_folder data/datasets/binance/BTCUSDT_v2 --output_base_name dataset
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

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
    parser.add_argument("--output_folder", type=Path, required=True, help="Output folder for Parquet path")
    parser.add_argument("--output_base_name", type=str, required=True, help="Output base name for Parquet path")
    parser.add_argument("--warmup", type=int, default=600, help="Feature warmup seconds after bootstrap (default: 600)")
    parser.add_argument("--interval", type=int, default=100, help="Sampling interval ms (default: 100)")
    parser.add_argument("--horizon", type=int, default=500, help="Label horizon ms (default: 500)")
    parser.add_argument("--trade_window", type=int, default=1000, help="Trade window ms (default: 1000)")
    parser.add_argument("--duration", type=str, default=None, help="Use only this much data from the start (e.g. '4h', '1D')")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    duration_tag = f"_d{args.duration}" if args.duration else ""
    output_path = args.output_folder / (
        f'{args.output_base_name}{duration_tag}'
        f'_i{args.interval}_tw{args.trade_window}_w{args.warmup}_h{args.horizon}.parquet'
    )

    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output file already exists: {output_path}\nUse --overwrite to replace.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = fetch_symbol_config(args.symbol)

    book = OrderBook(cfg)
    fe = FeatureExtractor(sampling_interval_ms=args.interval, trade_window_ms=args.trade_window)
    lb = LabelBuilder(horizon_ms=args.horizon, sampling_interval_ms=args.interval)
    db = DatasetBuilder()

    duration_ms = int(pd.Timedelta(args.duration).total_seconds() * 1000) if args.duration else None

    engine = ReplayEngine(
        data_path=args.data_path,
        order_book=book,
        feature_extractor=fe,
        label_builder=lb,
        dataset_builder=db,
        warmup_seconds=args.warmup,
        duration_ms=duration_ms,
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
        "rows_seen": db.rows_seen,
        "rows_kept": len(db),
        "rows_dropped_missing_required": db.rows_dropped_missing_required,
        "sequence_gaps": engine.sequence_gaps_detected,
        "bootstrap_count": engine.bootstrap_count,
    }
    db.save_parquet(output_path, metadata=metadata)
    logging.getLogger(__name__).info("Saved %d rows to %s", len(db), output_path)


if __name__ == "__main__":
    main()

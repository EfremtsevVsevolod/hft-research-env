#!/usr/bin/env python3
"""Show summary of recorded raw data in a directory.

Usage:
    python scripts/raw_info.py data/raw/binance/BTCUSDT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import parse_filename_ts


def main() -> None:
    parser = argparse.ArgumentParser(description="Show summary of recorded raw data")
    parser.add_argument("data_path", type=Path, help="Directory with recorded Parquet files")
    args = parser.parse_args()

    files = sorted(args.data_path.glob("*.parquet"))
    if not files:
        print(f"No parquet files in {args.data_path}")
        return

    valid = [(f, ts) for f in files if (ts := parse_filename_ts(f)) is not None]
    if not valid:
        print(f"{len(files)} files, but none match YYYYMMdd_HHMMSS naming")
        return

    first_ts = valid[0][1]
    last_ts = valid[-1][1]
    duration = pd.Timedelta(milliseconds=last_ts - first_ts)

    print(f"Path:     {args.data_path}")
    print(f"Files:    {len(valid)}")
    print(f"First:    {valid[0][0].name}")
    print(f"Last:     {valid[-1][0].name}")
    print(f"Duration: {duration}")


if __name__ == "__main__":
    main()

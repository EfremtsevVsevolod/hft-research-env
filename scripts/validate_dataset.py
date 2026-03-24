#!/usr/bin/env python3
"""Validate a built dataset parquet file.

Usage:
    python scripts/validate_dataset.py data/datasets/dataset.parquet

Exit codes:
    0 — OK or WARNING
    1 — FAIL (dataset contract violated)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _load(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    table = pq.read_table(path)
    raw = table.schema.metadata or {}
    meta = {k.decode(): v.decode() for k, v in raw.items() if k != b"pandas"}
    df = table.to_pandas()
    return df, meta


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _meta_int(meta: dict[str, str], key: str) -> int | None:
    v = meta.get(key)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset parquet")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    if not args.path.exists():
        print(f"FAIL: file not found: {args.path}")
        sys.exit(1)

    df, meta = _load(args.path)
    fails: list[str] = []
    warns: list[str] = []

    # --- basic info ---
    _section("Dataset info")
    print(f"  Rows:    {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    if "timestamp" in df.columns and len(df) > 0:
        ts_min, ts_max = df["timestamp"].min(), df["timestamp"].max()
        print(f"  Timestamp range: {ts_min} — {ts_max}")
        print(f"  Time range:      {pd.Timestamp(ts_min, unit='ms')} — "
              f"{pd.Timestamp(ts_max, unit='ms')}")
        duration_s = (ts_max - ts_min) / 1000
        print(f"  Duration: {duration_s:.1f}s")

    # --- schema ---
    if "timestamp" not in df.columns:
        fails.append("no timestamp column")
    if "label" not in df.columns:
        fails.append("no label column")

    if len(df) == 0:
        fails.append("dataset is empty")
    elif len(df) < 36000:
        warns.append(f"dataset very small ({len(df)} rows)")

    # --- timestamp integrity ---
    _section("Timestamp integrity")
    if "timestamp" in df.columns and len(df) > 1:
        diffs = df["timestamp"].diff().dropna()
        dupes = int((diffs == 0).sum())
        non_increasing = int((diffs < 0).sum())

        if non_increasing > 0:
            fails.append(f"{non_increasing} non-increasing timestamps")
            print(f"  FAIL: {non_increasing} non-increasing timestamps")
        elif dupes > 0:
            fails.append(f"{dupes} duplicate timestamps")
            print(f"  FAIL: {dupes} duplicate timestamps")
        else:
            print(f"  Strictly increasing: OK")

        interval = _meta_int(meta, "interval_ms")
        if interval:
            misaligned = int((df["timestamp"].astype(int) % interval != 0).sum())
            if misaligned > 0:
                fails.append(f"{misaligned} timestamps not aligned to {interval}ms grid")
                print(f"  FAIL: {misaligned} not aligned to {interval}ms grid")
            else:
                print(f"  Grid alignment ({interval}ms): OK")

            gaps = diffs[diffs > interval]
            if len(gaps) > 0:
                warns.append(f"{len(gaps)} timestamp gaps (max {int(gaps.max())}ms)")
                print(f"  WARNING: {len(gaps)} gaps > {interval}ms "
                      f"(max {int(gaps.max())}ms)")
    elif "timestamp" in df.columns:
        print(f"  Skipped (< 2 rows)")

    # --- NaN / inf ---
    _section("Missing / invalid values")
    nan_counts = df.isnull().sum()
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = pd.Series(
        {c: int(np.isinf(df[c]).sum()) for c in num_cols}
    )

    if nan_counts.sum() > 0:
        fails.append("NaN values found")
        for col in nan_counts[nan_counts > 0].index:
            print(f"  FAIL NaN: {col} = {nan_counts[col]}")
    else:
        print(f"  NaN: none")

    if inf_counts.sum() > 0:
        fails.append("inf values found")
        for col in inf_counts[inf_counts > 0].index:
            print(f"  FAIL inf: {col} = {inf_counts[col]}")
    else:
        print(f"  inf: none")

    # --- feature invariants ---
    if len(df) > 0:
        _section("Feature invariants")
        all_ok = True

        for col in ("spread", "buy_volume", "sell_volume"):
            if col in df.columns and (df[col] < 0).any():
                n = int((df[col] < 0).sum())
                fails.append(f"{col} < 0 ({n} rows)")
                print(f"  FAIL: {col} < 0 in {n} rows")
                all_ok = False

        for col in ("imbalance_1", "imbalance_5", "imbalance_10"):
            if col in df.columns:
                oob = ((df[col] < -1) | (df[col] > 1)).sum()
                if oob > 0:
                    fails.append(f"{col} outside [-1, 1] ({int(oob)} rows)")
                    print(f"  FAIL: {col} outside [-1, 1] in {int(oob)} rows")
                    all_ok = False

        if all_ok:
            print(f"  OK")

    # --- suspicious patterns ---
    if len(df) > 0:
        _section("Suspicious patterns")
        any_warn = False
        feature_cols = [c for c in num_cols if c not in {"timestamp", "label"}]
        for col in feature_cols:
            nunique = df[col].nunique()
            if nunique == 1:
                warns.append(f"{col} is constant")
                print(f"  WARNING: {col} is constant (value={df[col].iloc[0]})")
                any_warn = True
            elif len(df) > 10 and (df[col] == 0).sum() / len(df) > 0.95:
                pct = (df[col] == 0).sum() / len(df)
                warns.append(f"{col} >95% zeros")
                print(f"  WARNING: {col} is {pct:.0%} zeros")
                any_warn = True
        if not any_warn:
            print(f"  OK")

    # --- label summary ---
    if "label" in df.columns and len(df) > 0:
        _section("Label distribution")
        lbl = df["label"]
        n = len(lbl)
        std = lbl.std()
        mean = lbl.mean()
        print(f"  mean:     {mean:.6f}")
        print(f"  std:      {std:.6f}")
        print(f"  min/max:  {lbl.min():.6f} / {lbl.max():.6f}")
        for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
            print(f"  q{q:.0%}:   {lbl.quantile(q):.6f}".replace("%", ""))
        n_zero = int((lbl == 0).sum())
        n_pos = int((lbl > 0).sum())
        n_neg = int((lbl < 0).sum())
        print(f"  zero:     {n_zero} ({n_zero / n:.1%})")
        print(f"  positive: {n_pos} ({n_pos / n:.1%})")
        print(f"  negative: {n_neg} ({n_neg / n:.1%})")

        if std == 0:
            warns.append("label std == 0")
        if n_zero / n > 0.95:
            warns.append(f"label >95% zeros ({n_zero / n:.0%})")
        if std > 0 and abs(mean) > 5 * std:
            warns.append(f"label very skewed (|mean|/std = {abs(mean)/std:.1f})")

    # --- metadata ---
    if meta:
        _section("Parquet metadata")
        for k, v in sorted(meta.items()):
            print(f"  {k}: {v}")

    # --- metadata consistency ---
    if meta and len(df) > 0:
        _section("Metadata consistency")
        meta_ok = True

        rows_kept = _meta_int(meta, "rows_kept")
        if rows_kept is not None and rows_kept != len(df):
            fails.append(f"rows_kept ({rows_kept}) != len(df) ({len(df)})")
            print(f"  FAIL: rows_kept={rows_kept} != actual rows={len(df)}")
            meta_ok = False

        interval = _meta_int(meta, "interval_ms")
        if interval is not None and interval <= 0:
            fails.append(f"interval_ms <= 0 ({interval})")
            print(f"  FAIL: interval_ms={interval}")
            meta_ok = False

        horizon = _meta_int(meta, "horizon_ms")
        if horizon is not None and horizon <= 0:
            fails.append(f"horizon_ms <= 0 ({horizon})")
            print(f"  FAIL: horizon_ms={horizon}")
            meta_ok = False

        if interval and interval > 0 and horizon and horizon % interval != 0:
            fails.append(f"horizon_ms ({horizon}) not multiple of interval_ms ({interval})")
            print(f"  FAIL: horizon_ms % interval_ms != 0")
            meta_ok = False

        rows_seen = _meta_int(meta, "rows_seen")
        dropped = _meta_int(meta, "rows_dropped_missing_required")
        if rows_seen and rows_seen > 0 and dropped is not None:
            drop_rate = dropped / rows_seen
            if drop_rate > 0.5:
                warns.append(f"high drop rate ({drop_rate:.0%})")
                print(f"  WARNING: drop rate = {drop_rate:.0%} "
                      f"({dropped}/{rows_seen})")
                meta_ok = False

        gaps = _meta_int(meta, "sequence_gaps")
        if gaps and gaps > 0:
            warns.append(f"{gaps} sequence gaps")
            print(f"  WARNING: {gaps} sequence gap(s)")
            meta_ok = False

        bootstraps = _meta_int(meta, "bootstrap_count")
        if bootstraps and bootstraps > 1:
            warns.append(f"{bootstraps} bootstraps (re-syncs occurred)")
            print(f"  WARNING: {bootstraps} bootstraps")
            meta_ok = False

        if meta_ok:
            print(f"  OK")

    # --- verdict ---
    _section("Verdict")
    if fails:
        print("  FAIL:")
        for f in fails:
            print(f"    {f}")
        sys.exit(1)
    elif warns:
        print("  WARNING:")
        for w in warns:
            print(f"    {w}")
    else:
        print("  OK")


if __name__ == "__main__":
    main()

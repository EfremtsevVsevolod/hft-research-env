#!/usr/bin/env python3
"""Compare two built dataset parquet files and report drift.

Usage:
    python scripts/compare_datasets.py old.parquet new.parquet
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


def _safe_corr(a: pd.Series, b: pd.Series) -> str:
    if a.std() == 0 or b.std() == 0:
        return "N/A (constant)"
    return f"{a.corr(b):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two dataset parquets")
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top changed timestamps to print (default: 10)")
    args = parser.parse_args()

    for path in (args.old, args.new):
        if not path.exists():
            print(f"FAIL: file not found: {path}")
            sys.exit(1)

    df_old, meta_old = _load(args.old)
    df_new, meta_new = _load(args.new)

    # --- dataset info ---
    for label, df, path in [("Old", df_old, args.old), ("New", df_new, args.new)]:
        _section(f"{label} dataset: {path.name}")
        print(f"  Rows:    {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        if "timestamp" in df.columns and len(df) > 0:
            ts_min, ts_max = df["timestamp"].min(), df["timestamp"].max()
            print(f"  Time range: {pd.Timestamp(ts_min, unit='ms')} — "
                  f"{pd.Timestamp(ts_max, unit='ms')}")
            print(f"  Duration:   {(ts_max - ts_min) / 1000:.1f}s")

    # --- metadata diff ---
    _section("Metadata diff")
    meta_keys = [
        "interval_ms", "horizon_ms", "warmup_s", "trade_window_ms",
        "rows_seen", "rows_kept", "rows_dropped_missing_required",
        "sequence_gaps", "bootstrap_count",
    ]
    any_diff = False
    print(f"  {'key':<35s} {'old':>12s}  {'new':>12s}")
    print(f"  {'-'*35} {'-'*12}  {'-'*12}")
    for key in meta_keys:
        v_old = meta_old.get(key, "—")
        v_new = meta_new.get(key, "—")
        marker = " *" if v_old != v_new else ""
        if marker:
            any_diff = True
        print(f"  {key:<35s} {v_old:>12s}  {v_new:>12s}{marker}")
    if not any_diff:
        print(f"  (no differences)")

    # --- timestamp overlap ---
    _section("Timestamp overlap")
    if "timestamp" not in df_old.columns or "timestamp" not in df_new.columns:
        print(f"  Cannot compare: timestamp column missing")
        return

    ts_old = set(df_old["timestamp"])
    ts_new = set(df_new["timestamp"])
    only_old = ts_old - ts_new
    only_new = ts_new - ts_old
    shared = ts_old & ts_new
    print(f"  Only in old: {len(only_old)}")
    print(f"  Only in new: {len(only_new)}")
    print(f"  Shared:      {len(shared)}")

    if not shared:
        print(f"  No shared timestamps — cannot compare values.")
        return

    # inner join
    merged = df_old.merge(df_new, on="timestamp", suffixes=("_old", "_new"))
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    num_cols_old = set(df_old.select_dtypes(include=[np.number]).columns) - {"timestamp"}
    num_cols_new = set(df_new.select_dtypes(include=[np.number]).columns) - {"timestamp"}
    shared_num = sorted(num_cols_old & num_cols_new)

    # --- column sets ---
    only_in_old = sorted(num_cols_old - num_cols_new)
    only_in_new = sorted(num_cols_new - num_cols_old)
    _section("Column sets")
    print(f"  Numeric only in old: {only_in_old if only_in_old else '(none)'}")
    print(f"  Numeric only in new: {only_in_new if only_in_new else '(none)'}")
    print(f"  Numeric shared:      {shared_num}")

    # --- column drift ---
    _section("Column drift (shared timestamps)")
    print(f"  {'column':<25s} {'eq_rate':>8s} {'allclose':>8s} {'mean_abs':>10s} "
          f"{'max_abs':>10s} {'corr':>16s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*16}")
    for col in shared_num:
        old = merged[f"{col}_old"]
        new = merged[f"{col}_new"]
        diff = (old - new).abs()
        eq_rate = (old == new).mean()
        close = (np.isclose(old, new, atol=1e-9, rtol=1e-6)
                 | (old.isna() & new.isna())).mean()
        corr = _safe_corr(old, new)
        print(f"  {col:<25s} {eq_rate:>8.1%} {close:>8.1%} {diff.mean():>10.6f} "
              f"{diff.max():>10.6f} {corr:>16s}")

    # --- label drift ---
    if "label" in shared_num:
        _section("Label drift")
        old_lbl = merged["label_old"]
        new_lbl = merged["label_new"]
        diff_lbl = (old_lbl - new_lbl).abs()
        print(f"  mean  old={old_lbl.mean():.6f}  new={new_lbl.mean():.6f}")
        print(f"  std   old={old_lbl.std():.6f}  new={new_lbl.std():.6f}")
        print(f"  mean_abs_diff: {diff_lbl.mean():.6f}")
        print(f"  max_abs_diff:  {diff_lbl.max():.6f}")
        print(f"  correlation:   {_safe_corr(old_lbl, new_lbl)}")

    # --- feature-label relationship drift ---
    if "label" in shared_num:
        feature_cols = [c for c in shared_num if c != "label"]
        if feature_cols:
            _section("Feature-label correlation drift")
            print(f"  {'feature':<25s} {'corr_old':>10s} {'corr_new':>10s} "
                  f"{'delta':>10s}")
            print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
            for col in feature_cols:
                old_col = merged[f"{col}_old"]
                new_col = merged[f"{col}_new"]
                old_lbl = merged["label_old"]
                new_lbl = merged["label_new"]

                c_old = old_col.corr(old_lbl) if old_col.std() > 0 and old_lbl.std() > 0 else float("nan")
                c_new = new_col.corr(new_lbl) if new_col.std() > 0 and new_lbl.std() > 0 else float("nan")
                delta = c_new - c_old if np.isfinite(c_old) and np.isfinite(c_new) else float("nan")

                def _fmt(v: float) -> str:
                    return f"{v:>10.4f}" if np.isfinite(v) else f"{'N/A':>10s}"

                print(f"  {col:<25s} {_fmt(c_old)} {_fmt(c_new)} {_fmt(delta)}")

    # --- top changed timestamps ---
    important_cols = [
        "label", "spread", "imbalance_1", "delta_midprice",
        "microprice_minus_mid", "buy_volume", "sell_volume",
    ]
    cols_to_show = [c for c in important_cols if c in shared_num]
    if cols_to_show:
        _section(f"Top {args.top_k} changed timestamps")
        for col in cols_to_show:
            diff = (merged[f"{col}_old"] - merged[f"{col}_new"]).abs()
            if diff.max() == 0:
                continue
            top = diff.nlargest(args.top_k)
            print(f"\n  {col}:")
            print(f"    {'timestamp':>18s}  {'old':>12s}  {'new':>12s}  {'abs_diff':>12s}")
            for idx in top.index:
                ts = int(merged.loc[idx, "timestamp"])
                v_old = merged.loc[idx, f"{col}_old"]
                v_new = merged.loc[idx, f"{col}_new"]
                print(f"    {ts:>18d}  {v_old:>12.6f}  {v_new:>12.6f}  "
                      f"{abs(v_old - v_new):>12.6f}")


if __name__ == "__main__":
    main()

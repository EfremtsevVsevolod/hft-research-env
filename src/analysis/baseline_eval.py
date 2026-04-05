"""Shared evaluation helpers for baseline notebooks.

Keeps notebook code compact by centralising split logic, purge, and
ranking-style evaluation tables.  Analysis-only — does not touch core
pipeline semantics.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def full_utc_days(df: pd.DataFrame) -> list[_dt.date]:
    """Return UTC calendar dates that span a full 00:00 → 23:59 range.

    Requires a ``datetime`` column (UTC-aware) already present.
    """
    df = df.copy()
    df["_date"] = df["datetime"].dt.date
    grouped = df.groupby("_date")["datetime"]
    full = []
    for d, grp in grouped:
        if grp.min().hour == 0 and grp.max().hour == 23:
            full.append(d)
    return sorted(full)


def split_by_utc_days(
    df: pd.DataFrame,
    train_dates: list[_dt.date],
    val_dates: list[_dt.date],
    test_dates: list[_dt.date],
    horizon_ms: int,
) -> dict[str, pd.DataFrame]:
    """Split *df* by UTC calendar dates and purge at each boundary.

    Purge removes rows from the **tail** of each preceding split whose
    label window (``horizon_ms`` forward) would reach into the next split.

    A row at timestamp *t* has label target at *t + horizon_ms*.
    To avoid leakage the row must satisfy ``t + horizon_ms < next_split_start``,
    i.e. ``t < next_split_start - horizon_ms``.
    """
    date_col = df["datetime"].dt.date
    parts: dict[str, pd.DataFrame] = {
        "train": df[date_col.isin(train_dates)].copy(),
        "val":   df[date_col.isin(val_dates)].copy(),
        "test":  df[date_col.isin(test_dates)].copy(),
    }

    ordered = ["train", "val", "test"]
    for i in range(len(ordered) - 1):
        left_name, right_name = ordered[i], ordered[i + 1]
        left, right = parts[left_name], parts[right_name]
        if len(left) == 0 or len(right) == 0:
            continue
        # Strict inequality: row at t must have t + horizon < right_start
        purge_boundary = right["timestamp"].min() - horizon_ms
        n_before = len(left)
        parts[left_name] = left[left["timestamp"] < purge_boundary].copy()
        n_purged = n_before - len(parts[left_name])
        print(f"  Purged {n_purged} rows from {left_name} tail")

    for name in ordered:
        p = parts[name]
        if len(p) == 0:
            print(f"{name:>5s}: {'empty':>10s}")
        else:
            t_min = pd.Timestamp(p["timestamp"].min(), unit="ms", tz="UTC")
            t_max = pd.Timestamp(p["timestamp"].max(), unit="ms", tz="UTC")
            print(f"{name:>5s}: {len(p):>10,} rows  [{t_min} → {t_max}]")

    return parts


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def bucket_analysis(
    score: np.ndarray,
    label: np.ndarray,
    ref_price: float,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Decile bucket stats."""
    tmp = pd.DataFrame({"score": score, "label": label})
    tmp["bucket"] = pd.qcut(tmp["score"], n_buckets, labels=False, duplicates="drop")
    grp = tmp.groupby("bucket")
    return pd.DataFrame({
        "count": grp["label"].count(),
        "mean_label": grp["label"].mean(),
        "mean_label_bps": grp["label"].mean() / ref_price * 1e4,
        "p_up": grp["label"].apply(lambda x: (x > 0).mean()),
        "p_down": grp["label"].apply(lambda x: (x < 0).mean()),
    })


def tail_stats(
    score: np.ndarray,
    label: np.ndarray,
    ref_price: float,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Stats for top / bottom tails at given quantile thresholds."""
    if quantiles is None:
        quantiles = [0.05, 0.10]
    rows = []
    for q in quantiles:
        lo = np.quantile(score, q)
        hi = np.quantile(score, 1 - q)
        for side, mask in [
            (f"bottom_{int(q * 100)}%", score <= lo),
            (f"top_{int(q * 100)}%", score >= hi),
        ]:
            sl = label[mask]
            rows.append({
                "tail": side,
                "count": len(sl),
                "mean_label": sl.mean(),
                "mean_label_bps": sl.mean() / ref_price * 1e4,
                "mean_abs_label_bps": np.abs(sl).mean() / ref_price * 1e4,
                "p_up": (sl > 0).mean(),
                "p_down": (sl < 0).mean(),
            })
    return pd.DataFrame(rows).set_index("tail")


def coverage_edge_sweep(
    score: np.ndarray,
    label: np.ndarray,
    ref_price: float,
    n_thresholds: int = 20,
) -> pd.DataFrame:
    """Threshold sweep: long if score > thr, short if score < -thr."""
    abs_score = np.abs(score)
    thresholds = np.quantile(abs_score, np.linspace(0, 0.95, n_thresholds))
    rows = []
    n = len(score)
    for thr in thresholds:
        long_mask = score > thr
        short_mask = score < -thr
        trade_mask = long_mask | short_mask
        if trade_mask.sum() == 0:
            continue
        signed_edge = np.where(long_mask, label, np.where(short_mask, -label, 0.0))
        traded = signed_edge[trade_mask]
        long_edge = label[long_mask].mean() if long_mask.sum() > 0 else np.nan
        short_edge = -label[short_mask].mean() if short_mask.sum() > 0 else np.nan
        rows.append({
            "threshold": thr,
            "coverage": trade_mask.sum() / n,
            "n_trades": trade_mask.sum(),
            "mean_edge_bps": traded.mean() / ref_price * 1e4,
            "long_n": long_mask.sum(),
            "long_edge_bps": long_edge / ref_price * 1e4,
            "short_n": short_mask.sum(),
            "short_edge_bps": short_edge / ref_price * 1e4,
        })
    return pd.DataFrame(rows)


def eval_summary(
    scores: dict[str, np.ndarray],
    label: np.ndarray,
    ref_price: float,
    horizon_ms: int,
) -> pd.DataFrame:
    """One-row-per-baseline summary: rank IC, tail stats, best edge."""
    rows = []
    for bl_name, sc in scores.items():
        ts = tail_stats(sc, label, ref_price, [0.05, 0.10])
        ce = coverage_edge_sweep(sc, label, ref_price)
        ic, _ = spearmanr(sc, label)
        ce_5 = ce[ce["coverage"] >= 0.05]
        best_edge = ce_5["mean_edge_bps"].max() if len(ce_5) > 0 else np.nan
        rows.append({
            "baseline": bl_name,
            "rank_IC": ic,
            "top10_bps": ts.loc["top_10%", "mean_label_bps"],
            "bot10_bps": ts.loc["bottom_10%", "mean_label_bps"],
            "top5_bps": ts.loc["top_5%", "mean_label_bps"],
            "bot5_bps": ts.loc["bottom_5%", "mean_label_bps"],
            "best_edge_bps_5cov": best_edge,
        })
    summary = pd.DataFrame(rows).set_index("baseline")
    print(f"Summary — h={horizon_ms}ms")
    return summary

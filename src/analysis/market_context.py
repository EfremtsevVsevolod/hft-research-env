"""Reusable helpers for market context / regime analysis.

Analysis-side utilities — they do NOT modify core replay, feature, or label
semantics.  They read raw recorded data and built datasets to produce summary
statistics and sampled book state for reporting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from src.config.config import SymbolConfig
from src.data.constants import EVENT_DEPTH_SNAPSHOT, EVENT_DEPTH_UPDATE, EVENT_TRADE
from src.lob.orderbook import OrderBook

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def load_raw_events(
    data_path: Path | str,
    sample_every: int = 1,
) -> pd.DataFrame:
    """Load raw parquet files into a single DataFrame.

    Parameters
    ----------
    data_path : directory containing *.parquet files
    sample_every : load every Nth file (1 = all)

    Returns DataFrame with columns:
        recv_ts, exchange_ts, event_type, stream, payload_json, datetime
    """
    data_path = Path(data_path)
    files = sorted(data_path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_path}")
    sample = files[::sample_every]
    df = pd.concat(
        [pd.read_parquet(f) for f in tqdm(sample, desc="Loading raw files")],
        ignore_index=True,
    )
    df["datetime"] = pd.to_datetime(df["recv_ts"], unit="ms", utc=True)
    logger.info(
        "Loaded %d/%d files (every %d), %d rows",
        len(sample), len(files), sample_every, len(df),
    )
    return df


def parse_trades(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Extract structured trade data from raw events.

    Returns DataFrame with columns:
        recv_ts, datetime, price, qty, is_buyer_maker, side, notional
    Units: price in quote (USDT), qty in base asset, notional in quote.
    """
    trades = raw_df[raw_df["event_type"] == EVENT_TRADE].copy()
    if trades.empty:
        return pd.DataFrame(columns=[
            "recv_ts", "datetime", "price", "qty", "is_buyer_maker",
            "side", "notional",
        ])
    parsed = trades["payload_json"].apply(json.loads)
    trades["price"] = parsed.apply(lambda x: float(x["p"]))
    trades["qty"] = parsed.apply(lambda x: float(x["q"]))
    trades["is_buyer_maker"] = parsed.apply(lambda x: x["m"])
    trades["side"] = np.where(trades["is_buyer_maker"], "sell", "buy")
    trades["notional"] = trades["price"] * trades["qty"]
    return trades[["recv_ts", "datetime", "price", "qty", "is_buyer_maker",
                    "side", "notional"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Lightweight book replay
# ---------------------------------------------------------------------------

def replay_book_samples(
    data_path: Path | str,
    cfg: SymbolConfig,
    sample_ms: int = 1000,
    max_files: int | None = None,
) -> pd.DataFrame:
    """Replay raw events through OrderBook and sample book state.

    Lightweight replay — only maintains the order book (no features, labels,
    or dataset).  Handles snapshot bootstrapping and sequence gap detection
    the same way ReplayEngine does.

    All book-derived metrics (midprice, spread, depth) come from the same
    replay pass so they are internally consistent.

    Parameters
    ----------
    data_path : directory with raw parquet files
    cfg : SymbolConfig for the symbol
    sample_ms : how often to sample book state (ms)
    max_files : limit number of files processed (None = all)

    Returns DataFrame with columns:
        timestamp (int, epoch ms), datetime (UTC),
        midprice (float, USDT), spread_ticks (int),
        bid_depth_lots (int), ask_depth_lots (int),
        segment_id (int, increments on each bootstrap)
    """
    data_path = Path(data_path)
    files = sorted(data_path.glob("*.parquet"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_path}")

    book = OrderBook(cfg)
    rows: list[dict] = []

    state = "WAIT_SNAPSHOT"
    last_uid: Optional[int] = None
    snap_uid: Optional[int] = None
    next_sample: Optional[int] = None
    segment_id: int = -1  # incremented to 0 on first snapshot

    def _sample(ts: int) -> None:
        mid = book.midprice()
        if mid is None:
            return
        sp = book.spread()
        bid_q, ask_q = book.volume_at_best()
        rows.append({
            "timestamp": ts,
            "midprice": float(mid),
            "spread_ticks": sp if sp is not None else 0,
            "bid_depth_lots": bid_q,
            "ask_depth_lots": ask_q,
            "segment_id": segment_id,
        })

    for fpath in tqdm(files, desc="Book replay"):
        df = pd.read_parquet(fpath)
        for row in df.itertuples(index=False):
            recv_ts = int(row.recv_ts)
            etype = row.event_type
            data = json.loads(row.payload_json)

            if etype == EVENT_DEPTH_SNAPSHOT:
                book.apply_snapshot({"bids": data["bids"], "asks": data["asks"]})
                snap_uid = int(data["lastUpdateId"])
                last_uid = snap_uid
                next_sample = None
                segment_id += 1
                state = "LIVE"
                continue

            if state == "WAIT_SNAPSHOT":
                continue

            if etype == EVENT_TRADE:
                continue  # trades don't affect book state

            if etype != EVENT_DEPTH_UPDATE:
                continue

            U = int(data["U"])
            u = int(data["u"])

            # Sequence validation (mirrors ReplayEngine)
            if snap_uid is not None:
                if u <= snap_uid:
                    continue
                if not (U <= snap_uid + 1 <= u):
                    state = "WAIT_SNAPSHOT"
                    book.clear()
                    last_uid = None
                    snap_uid = None
                    next_sample = None
                    continue
                snap_uid = None
                last_uid = u
            elif u < last_uid:
                continue
            elif U > last_uid + 1:
                state = "WAIT_SNAPSHOT"
                book.clear()
                last_uid = None
                next_sample = None
                continue
            else:
                last_uid = u

            # Sample on aligned grid, strictly before current depth update (causal).
            # Grid is globally aligned: floor(recv_ts / sample_ms) * sample_ms.
            if next_sample is not None:
                while next_sample < recv_ts:
                    _sample(next_sample)
                    next_sample += sample_ms
            else:
                next_sample = recv_ts - (recv_ts % sample_ms)

            book.apply_update({"bids": data["b"], "asks": data["a"]})

            if book.is_crossed():
                state = "WAIT_SNAPSHOT"
                book.clear()
                last_uid = None
                next_sample = None

    result = pd.DataFrame(rows)
    if not result.empty:
        result["datetime"] = pd.to_datetime(result["timestamp"], unit="ms", utc=True)
    n_segments = segment_id + 1
    logger.info(
        "Book replay: %d samples, %d segments from %d files",
        len(result), n_segments, len(files),
    )
    return result


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_with_meta(path: Path | str) -> tuple[pd.DataFrame, dict]:
    """Load a built dataset parquet with its metadata.

    Returns (DataFrame, metadata_dict).  Metadata keys/values are decoded
    strings.  The DataFrame gets an added 'datetime' column.
    """
    path = Path(path)
    table = pq.read_table(path)
    df = table.to_pandas()
    meta = {}
    if table.schema.metadata:
        meta = {
            k.decode(): v.decode()
            for k, v in table.schema.metadata.items()
            if not k.startswith(b"pandas") and not k.startswith(b"ARROW")
        }
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df, meta

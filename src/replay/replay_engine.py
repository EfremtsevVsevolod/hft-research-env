"""Offline replay of recorded Binance events through the feature pipeline.

Reads raw Parquet files produced by Recorder, reconstructs the order book
from depth snapshots, and drives FeatureExtractor → LabelBuilder → DatasetBuilder.

The book is initialized from a depthSnapshot event (REST snapshot stored in the
recording).  After each bootstrap a warmup period lets feature state stabilize
before snapshots are emitted to labels/dataset.  During warmup the book
updates normally and features are computed — only emission is suppressed.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.constants import EVENT_DEPTH_SNAPSHOT, EVENT_DEPTH_UPDATE, EVENT_TRADE
from src.dataset.dataset import DatasetBuilder
from src.lob.features import FeatureExtractor, FeatureSnapshot, Trade
from src.lob.labels import LabelBuilder
from src.lob.orderbook import OrderBook

logger = logging.getLogger(__name__)


class ReplayEngine:
    """Replay recorded Binance events through the full feature pipeline.

    All timing uses ``recv_ts`` (local receive time), not ``exchange_ts``.
    This models what a local system knew at the time it knew it — in a
    live system you act on data when you receive it, not when the exchange
    generated it.  ``exchange_ts`` is preserved in raw data for analysis.

    The book must be initialized from a ``depthSnapshot`` event before any
    depth updates are applied.  Recordings without snapshots will produce
    no output.

    Parameters
    ----------
    data_path
        Directory containing Parquet files produced by :class:`Recorder`.
    order_book
        Pre-configured OrderBook instance.
    feature_extractor
        FeatureExtractor (determines grid interval).
    label_builder
        LabelBuilder (determines horizon).
    dataset_builder
        DatasetBuilder to accumulate labelled rows.
    warmup_seconds
        Seconds after each bootstrap before feature snapshots are emitted.
        During warmup the book updates and features are computed normally,
        but snapshots are not forwarded to labels/dataset.
    """

    def __init__(
        self,
        data_path: Path,
        order_book: OrderBook,
        feature_extractor: FeatureExtractor,
        label_builder: LabelBuilder,
        dataset_builder: DatasetBuilder,
        warmup_seconds: int = 0,
    ) -> None:
        self._data_path = Path(data_path)
        self._book = order_book
        self._fe = feature_extractor
        self._lb = label_builder
        self._db = dataset_builder
        self._interval = feature_extractor.interval
        self._warmup_ms = warmup_seconds * 1000
        assert label_builder._horizon % feature_extractor.interval == 0, (
            f"label horizon ({label_builder._horizon}ms) must be a multiple of "
            f"sampling interval ({feature_extractor.interval}ms)"
        )

        self._last_update_id: Optional[int] = None
        self._snapshot_update_id: Optional[int] = None
        self._bootstrapped = False
        self._bootstrap_count: int = 0
        self._next_grid: Optional[int] = None
        self._warmup_done = False
        self._warmup_end_ts: Optional[int] = None
        self._trade_buffer: deque[Trade] = deque()
        self._event_count: int = 0
        self.sequence_gaps_detected: int = 0

    def _reset(self, reason: str) -> None:
        """Clear book and pipeline state, wait for next snapshot."""
        logger.warning("%s. Resetting book, waiting for next snapshot.", reason)
        self._book.clear()
        self._fe.reset()
        self._lb.reset()
        self._db.reset_timestamp()
        self._last_update_id = None
        self._bootstrapped = False
        self._next_grid = None
        self._warmup_done = False
        self._warmup_end_ts = None
        self._trade_buffer = deque()

    def run(self) -> None:
        """Replay all Parquet files and populate the dataset builder."""
        files = sorted(self._data_path.glob("*.parquet"))
        if not files:
            logger.warning("No parquet files found in %s", self._data_path)
            return

        logger.info("Replaying %d parquet files from %s", len(files), self._data_path)

        for fpath in files:
            df = pd.read_parquet(fpath)
            for row in df.itertuples(index=False):
                self.process_event(int(row.recv_ts), row.event_type, json.loads(row.payload_json))

        logger.info(
            "Replay finished: %d events, %d dataset rows, %d sequence gaps, %d bootstraps",
            self._event_count, len(self._db), self.sequence_gaps_detected,
            self._bootstrap_count,
        )

    def process_event(self, recv_ts: int, event_type: str, data: dict) -> None:
        """Process a single event. Used by run() and available for testing."""
        if event_type == EVENT_DEPTH_SNAPSHOT:
            self._on_snapshot(recv_ts, data)
        elif event_type == EVENT_TRADE:
            self._on_trade(recv_ts, data)
        elif event_type == EVENT_DEPTH_UPDATE:
            self._on_depth(recv_ts, data)
        else:
            raise ValueError(f"Unknown event type: {event_type}")
        self._event_count += 1
        if self._event_count % 100_000 == 0:
            logger.info(
                "Processed %d events, recv_ts=%d, dataset rows=%d",
                self._event_count, recv_ts, len(self._db),
            )

    def _on_snapshot(self, recv_ts: int, data: dict) -> None:
        """Initialize book from a depth snapshot."""
        self._book.apply_snapshot({"bids": data["bids"], "asks": data["asks"]})
        self._snapshot_update_id = int(data["lastUpdateId"])
        self._last_update_id = int(data["lastUpdateId"])
        self._bootstrapped = True
        self._bootstrap_count += 1
        self._next_grid = None
        self._warmup_done = False
        self._warmup_end_ts = None
        self._fe.reset()
        self._lb.reset()
        self._db.reset_timestamp()
        self._trade_buffer.clear()
        logger.info(
            "Bootstrap #%d from snapshot lastUpdateId=%d",
            self._bootstrap_count, self._snapshot_update_id,
        )

    def _on_trade(self, recv_ts: int, data: dict) -> None:
        """Buffer a trade event (only after bootstrap)."""
        if not self._bootstrapped:
            return
        trade = Trade(
            timestamp=recv_ts,
            price=Decimal(data["p"]),
            quantity=Decimal(data["q"]),
            is_buyer_maker=data["m"],
        )
        self._trade_buffer.append(trade)

    def _on_depth(self, recv_ts: int, data: dict) -> None:
        """Validate sequence, emit grid snapshots, then apply update.

        Order of operations ensures causality: grid nodes see only the
        book state from BEFORE this depth event, matching live behavior.
        """
        if not self._bootstrapped:
            return

        U = int(data["U"])
        u = int(data["u"])

        # 1. Sequence check
        if self._snapshot_update_id is not None:
            # Drop stale diffs from before the snapshot
            if u <= self._snapshot_update_id:
                return
            # First non-stale diff after snapshot: validate overlap
            if not (U <= self._snapshot_update_id + 1 <= u):
                logger.warning(
                    "First diff after snapshot failed sync: U=%d, u=%d, lastUpdateId=%d",
                    U, u, self._snapshot_update_id,
                )
                self._reset("Snapshot sync validation failed")
                return
            self._snapshot_update_id = None
            self._last_update_id = u
        elif u < self._last_update_id:
            return  # stale event
        elif U > self._last_update_id + 1:
            self.sequence_gaps_detected += 1
            self._reset(
                f"Sequence gap #{self.sequence_gaps_detected}: "
                f"expected {self._last_update_id + 1}, got U={U}, "
                f"{self._event_count} events processed"
            )
            return
        else:
            self._last_update_id = u

        # 2. Compute features BEFORE applying update (causal correctness).
        #    Grid nodes use the book state prior to this depth event.
        #    During warmup features are computed (state evolves) but not emitted.
        if self._next_grid is not None:
            while self._next_grid < recv_ts:
                causal_trades: list[Trade] = []
                while self._trade_buffer and self._trade_buffer[0].timestamp <= self._next_grid:
                    causal_trades.append(self._trade_buffer.popleft())
                snap = self._fe.on_book_update(
                    self._next_grid, self._book, causal_trades,
                )
                if snap is not None:
                    if not self._warmup_done and self._next_grid >= self._warmup_end_ts:
                        self._warmup_done = True
                        logger.info(
                            "Warmup complete at grid_ts=%d (%d events processed)",
                            self._next_grid, self._event_count,
                        )
                    if self._warmup_done:
                        self._emit(snap)
                self._next_grid += self._interval

        # 3. Apply update, then validate.
        self._book.apply_update({"bids": data["b"], "asks": data["a"]})

        if self._book.is_crossed():
            self._reset("Crossed book detected")
            return

        # 4. Initialize grid and warmup deadline on first depth after snapshot.
        if self._next_grid is None:
            self._next_grid = recv_ts - (recv_ts % self._interval)
            self._warmup_end_ts = self._next_grid + self._warmup_ms
            self._warmup_done = (self._warmup_ms == 0)

    def _emit(self, snap: FeatureSnapshot) -> None:
        """Push a snapshot through label builder and dataset builder."""
        labelled = self._lb.on_snapshot(snap)
        if labelled is not None:
            self._db.on_labelled_snapshot(labelled)

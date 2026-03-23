"""Offline replay of recorded Binance events through the feature pipeline.

Reads raw Parquet files produced by Recorder, reconstructs the order book
from depth snapshots, and drives FeatureExtractor → LabelBuilder → DatasetBuilder.

State machine
-------------
WAIT_SNAPSHOT  →  depthSnapshot  →  WARMING  →  warmup elapsed  →  LIVE
     ↑                                                                |
     └──── gap / invalid continuity / crossed book ───────────────────┘

WAIT_SNAPSHOT: ignore depthUpdate and trade, wait for depthSnapshot.
WARMING:       book updates normally, features evolve, but no rows emitted.
LIVE:          full pipeline — features emitted to labels and dataset.

Snapshot initializes the book immediately.  Warmup only stabilizes feature
state (delta_midprice, trade windows).  Snapshot-boundary timing is treated
pragmatically because the long warmup makes sub-ms precision irrelevant.
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

# Replay states
WAIT_SNAPSHOT = "WAIT_SNAPSHOT"
WARMING = "WARMING"
LIVE = "LIVE"


class ReplayEngine:
    """Replay recorded Binance events through the full feature pipeline.

    All timing uses ``recv_ts`` (local receive time), not ``exchange_ts``.
    This models what a local system knew at the time it knew it — in a
    live system you act on data when you receive it, not when the exchange
    generated it.  ``exchange_ts`` is preserved in raw data for analysis.

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
        assert label_builder.horizon % feature_extractor.interval == 0, (
            f"label horizon ({label_builder.horizon}ms) must be a multiple of "
            f"sampling interval ({feature_extractor.interval}ms)"
        )
        if self._warmup_ms < feature_extractor.trade_window:
            raise ValueError(
                f"warmup ({self._warmup_ms}ms) must be >= "
                f"trade_window ({feature_extractor.trade_window}ms) "
                f"for feature stabilization"
            )

        self.state: str = WAIT_SNAPSHOT
        self._last_update_id: Optional[int] = None
        self._snapshot_update_id: Optional[int] = None
        self.bootstrap_count: int = 0
        self._next_grid: Optional[int] = None
        self._warmup_end_ts: Optional[int] = None
        self._trade_buffer: deque[Trade] = deque()
        self._event_count: int = 0
        self.sequence_gaps_detected: int = 0

    # --- state transitions ---------------------------------------------------

    def _bootstrap_from_snapshot(self, data: dict) -> None:
        """WAIT_SNAPSHOT → WARMING (or LIVE if warmup_ms == 0)."""
        self._book.apply_snapshot({"bids": data["bids"], "asks": data["asks"]})
        self._snapshot_update_id = int(data["lastUpdateId"])
        self._last_update_id = int(data["lastUpdateId"])
        self.bootstrap_count += 1
        self._next_grid = None
        self._warmup_end_ts = None
        self._fe.reset()
        self._lb.reset()
        self._db.reset_timestamp()
        self._trade_buffer.clear()
        self.state = LIVE if self._warmup_ms == 0 else WARMING
        logger.info(
            "Bootstrap #%d from snapshot lastUpdateId=%d → %s",
            self.bootstrap_count, self._snapshot_update_id, self.state,
        )

    def _transition_to_wait_snapshot(self, reason: str) -> None:
        """Any state → WAIT_SNAPSHOT on invalid continuity."""
        logger.warning("%s. Resetting, waiting for next snapshot.", reason)
        self._book.clear()
        self._fe.reset()
        self._lb.reset()
        self._db.reset_timestamp()
        self._last_update_id = None
        self._snapshot_update_id = None
        self._next_grid = None
        self._warmup_end_ts = None
        self._trade_buffer = deque()
        self.state = WAIT_SNAPSHOT

    def _maybe_transition_warming_to_live(self, grid_ts: int) -> None:
        """WARMING → LIVE when grid timestamp reaches warmup deadline."""
        if self.state == WARMING and grid_ts >= self._warmup_end_ts:
            self.state = LIVE
            logger.info(
                "Warmup complete at grid_ts=%d (%d events processed)",
                grid_ts, self._event_count,
            )

    # --- replay loop ---------------------------------------------------------

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
            self.bootstrap_count,
        )

    def process_event(self, recv_ts: int, event_type: str, data: dict) -> None:
        """Process a single event. Used by run() and available for testing."""
        if event_type == EVENT_DEPTH_SNAPSHOT:
            self._bootstrap_from_snapshot(data)
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

    # --- event handlers ------------------------------------------------------

    def _on_trade(self, recv_ts: int, data: dict) -> None:
        """Buffer a trade event (ignored in WAIT_SNAPSHOT)."""
        if self.state == WAIT_SNAPSHOT:
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
        if self.state == WAIT_SNAPSHOT:
            return

        U = int(data["U"])
        u = int(data["u"])

        # 1. Sequence check
        if self._snapshot_update_id is not None:
            if u <= self._snapshot_update_id:
                return  # stale diff from before snapshot
            if not (U <= self._snapshot_update_id + 1 <= u):
                logger.warning(
                    "First diff after snapshot failed sync: U=%d, u=%d, lastUpdateId=%d",
                    U, u, self._snapshot_update_id,
                )
                self._transition_to_wait_snapshot("Snapshot sync validation failed")
                return
            self._snapshot_update_id = None
            self._last_update_id = u
        elif u < self._last_update_id:
            return  # stale event
        elif U > self._last_update_id + 1:
            self.sequence_gaps_detected += 1
            self._transition_to_wait_snapshot(
                f"Sequence gap #{self.sequence_gaps_detected}: "
                f"expected {self._last_update_id + 1}, got U={U}, "
                f"{self._event_count} events processed"
            )
            return
        else:
            self._last_update_id = u

        # 2. Compute features BEFORE applying update (causal correctness).
        #    During WARMING features are computed (state evolves) but not emitted.
        if self._next_grid is not None:
            while self._next_grid < recv_ts:
                causal_trades: list[Trade] = []
                while self._trade_buffer and self._trade_buffer[0].timestamp <= self._next_grid:
                    causal_trades.append(self._trade_buffer.popleft())
                snap = self._fe.on_book_update(
                    self._next_grid, self._book, causal_trades,
                )
                if snap is not None:
                    self._maybe_transition_warming_to_live(self._next_grid)
                    if self.state == LIVE:
                        self._emit(snap)
                self._next_grid += self._interval

        # 3. Apply update, then validate.
        self._book.apply_update({"bids": data["b"], "asks": data["a"]})

        if self._book.is_crossed():
            self._transition_to_wait_snapshot("Crossed book detected")
            return

        # 4. Initialize grid and warmup deadline on first depth after snapshot.
        if self._next_grid is None:
            self._next_grid = recv_ts - (recv_ts % self._interval)
            self._warmup_end_ts = self._next_grid + self._warmup_ms

    def _emit(self, snap: FeatureSnapshot) -> None:
        """Push a snapshot through label builder and dataset builder."""
        labelled = self._lb.on_snapshot(snap)
        if labelled is not None:
            self._db.on_labelled_snapshot(labelled)

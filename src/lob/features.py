"""Event-driven causal microstructure feature extraction.

Every event is processed (trade buffers updated).  Snapshots are emitted
only at regular grid nodes.  The caller guarantees that an event arrives
at every grid node (real or synthetic).  No backfilling, no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import NamedTuple, Optional

from sortedcontainers import SortedList

from .orderbook import OrderBook


class Trade(NamedTuple):
    """A single trade event."""
    timestamp: int      # recv_ts in ms
    price: Decimal
    quantity: Decimal
    is_buyer_maker: bool  # True → sell aggressor (taker sold)


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """One row of sampled features at a grid node."""
    timestamp: int
    midprice: Optional[Decimal]
    spread: Optional[Decimal]
    imbalance_1: float
    imbalance_5: float
    imbalance_10: float
    microprice: Optional[Decimal]
    microprice_minus_mid: Optional[Decimal]
    delta_midprice: Optional[Decimal]
    buy_volume: Decimal
    sell_volume: Decimal


class FeatureExtractor:
    """Causal feature sampler on a regular time grid.

    The caller drives the grid by calling ``on_book_update`` at each grid
    node with the **current** book state and any trades up to that time.

    The extractor:

    1. Buffers incoming trades on **every** call.
    2. Emits a ``FeatureSnapshot`` only when the event timestamp lands
       on a grid node.

    The grid starts at the first event timestamp and advances by
    ``sampling_interval_ms``.  The caller must ensure that an event
    (real or synthetic) arrives at every grid node — no gaps.

    In ReplayEngine the book state passed to ``on_book_update`` is the
    state *before* the depth update that triggered grid emission (emit
    before apply).  This ensures grid nodes see only causally available
    data.
    """

    def __init__(
        self,
        sampling_interval_ms: int = 50,
        trade_window_ms: int = 100,
    ) -> None:
        self.interval = sampling_interval_ms
        self.trade_window = trade_window_ms

        self._next_emit: Optional[int] = None
        self._prev_ts: Optional[int] = None
        self._prev_midprice: Optional[Decimal] = None

        self._trades: SortedList[Trade] = SortedList()
        self._buy_vol = Decimal(0)
        self._sell_vol = Decimal(0)

    def reset(self) -> None:
        """Reset all internal state. Call after a book reset / sequence gap."""
        self._next_emit = None
        self._prev_ts = None
        self._prev_midprice = None
        self._trades.clear()
        self._buy_vol = Decimal(0)
        self._sell_vol = Decimal(0)

    def on_book_update(
        self,
        timestamp: int,
        orderbook: OrderBook,
        trades: Optional[list[Trade]] = None,
    ) -> Optional[FeatureSnapshot]:
        """Process an event.  Return a snapshot if *timestamp* is a grid node.

        Trade buffers are updated on every call regardless of emission.
        """
        # 1. Validate monotonic timestamps.
        assert self._prev_ts is None or timestamp >= self._prev_ts, (
            f"timestamps not monotonic: {timestamp} < {self._prev_ts}"
        )
        self._prev_ts = timestamp

        # 2. Initialise grid on first event.
        if self._next_emit is None:
            self._next_emit = timestamp

        # 3. Buffer incoming trades and update running sums.
        if trades:
            for t in trades:
                self._trades.add(t)
                if t.is_buyer_maker:
                    self._sell_vol += t.quantity
                else:
                    self._buy_vol += t.quantity

        # 4. Emit only on grid nodes.
        assert timestamp <= self._next_emit, (
            f"missed grid node: expected event at {self._next_emit}, "
            f"got {timestamp}"
        )
        if timestamp != self._next_emit:
            return None

        snap = self._sample(timestamp, orderbook)
        self._next_emit += self.interval
        return snap

    # ------------------------------------------------------------------

    def _sample(self, ts: int, book: OrderBook) -> FeatureSnapshot:
        midprice = book.midprice()
        microprice = book.microprice()
        spread_ticks = book.spread()

        spread = (
            book.price_from_ticks(spread_ticks)
            if spread_ticks is not None else None
        )

        microprice_minus_mid: Optional[Decimal] = None
        if microprice is not None and midprice is not None:
            microprice_minus_mid = microprice - midprice

        delta_midprice: Optional[Decimal] = None
        if midprice is not None and self._prev_midprice is not None:
            delta_midprice = midprice - self._prev_midprice
        self._prev_midprice = midprice

        buy_vol, sell_vol = self._evict_trades(ts)

        return FeatureSnapshot(
            timestamp=ts,
            midprice=midprice,
            spread=spread,
            imbalance_1=book.imbalance(1),
            imbalance_5=book.imbalance(5),
            imbalance_10=book.imbalance(10),
            microprice=microprice,
            microprice_minus_mid=microprice_minus_mid,
            delta_midprice=delta_midprice,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
        )

    def _evict_trades(self, ts: int) -> tuple[Decimal, Decimal]:
        cutoff = ts - self.trade_window
        while self._trades and self._trades[0].timestamp < cutoff:
            t = self._trades.pop(0)
            if t.is_buyer_maker:
                self._sell_vol -= t.quantity
            else:
                self._buy_vol -= t.quantity
        return self._buy_vol, self._sell_vol

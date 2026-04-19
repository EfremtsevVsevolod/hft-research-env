"""Event-driven causal microstructure feature extraction.

The caller drives two hooks:

* ``observe_depth`` on **every** depth update (after ``apply_update``),
  feeding post-update L1 state.  Used for per-update features: OFI, queue
  delta, depth-update count, time-since-mid-move.  Does not emit rows.
* ``on_book_update`` on **every** grid node, where a :class:`FeatureSnapshot`
  is emitted.  Grid nodes see the book state *before* the depth update that
  triggered emission (emit-before-apply contract, maintained by caller).

Both hooks buffer trades on every call.  No backfilling, no look-ahead.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from math import log, sqrt
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
    # New compact package — per-update aggregates read at grid node.
    ofi_100ms: float
    ofi_500ms: float
    ofi_1000ms: float
    queue_delta_diff_1000ms: float
    depth_update_count_1000ms: int
    time_since_last_mid_move_ms: int
    signed_trade_volume_1000ms: Decimal
    realized_vol_microprice_1000ms: Optional[float]


class _RollingSum:
    """O(1) amortised rolling sum over a time window.

    Values are pushed with a timestamp and evicted on query when older than
    ``window_ms``.  Eviction boundary is strict ``<``: an entry at
    ``query_ts - window_ms`` is still included.
    """

    __slots__ = ("window_ms", "_buf", "_sum")

    def __init__(self, window_ms: int) -> None:
        self.window_ms = window_ms
        self._buf: deque[tuple[int, float]] = deque()
        self._sum: float = 0.0

    def push(self, ts: int, value: float) -> None:
        self._buf.append((ts, value))
        self._sum += value

    def value(self, query_ts: int) -> float:
        cutoff = query_ts - self.window_ms
        while self._buf and self._buf[0][0] < cutoff:
            _, v = self._buf.popleft()
            self._sum -= v
        return self._sum

    def count(self, query_ts: int) -> int:
        cutoff = query_ts - self.window_ms
        while self._buf and self._buf[0][0] < cutoff:
            _, v = self._buf.popleft()
            self._sum -= v
        return len(self._buf)

    def reset(self) -> None:
        self._buf.clear()
        self._sum = 0.0


class FeatureExtractor:
    """Causal feature sampler on a regular time grid.

    Two driver hooks:

    * :meth:`observe_depth` is called on every depth update (post
      ``apply_update``).  It buffers L1 deltas for OFI / queue-delta /
      depth-count / time-since-mid-move.
    * :meth:`on_book_update` is called at every grid node.  It buffers
      trades, then emits a :class:`FeatureSnapshot` when the event
      timestamp matches the next grid node.

    Caller responsibilities:

    * Seed ``observe_depth`` once right after bootstrap with the L1 read
      from the snapshot, so the first real update has a valid previous
      state.
    * Call both hooks in causal order: grid emission uses the pre-apply
      book state; ``observe_depth`` is called post-apply with the new L1.
    """

    _RV_MICRO_WINDOW_MS = 1000  # realized-vol window, grid-sampled

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

        # observe_depth state
        self._prev_L1: Optional[tuple[int, int, int, int]] = None  # (bb, bq, ba, aq)
        self._prev_mid_sum: Optional[int] = None  # bb + ba, for mid-move detection
        self._last_mid_move_ts: Optional[int] = None

        self._ofi_100 = _RollingSum(100)
        self._ofi_500 = _RollingSum(500)
        self._ofi_1000 = _RollingSum(1000)
        self._queue_delta_1000 = _RollingSum(1000)
        self._depth_count_1000 = _RollingSum(1000)

        # realized-vol state — grid-cadence
        rv_n = max(1, self._RV_MICRO_WINDOW_MS // sampling_interval_ms)
        self._micro_log_returns: deque[float] = deque(maxlen=rv_n)
        self._prev_micro: Optional[Decimal] = None

    def reset(self) -> None:
        """Reset all internal state. Call after a book reset / sequence gap."""
        self._next_emit = None
        self._prev_ts = None
        self._prev_midprice = None
        self._trades.clear()
        self._buy_vol = Decimal(0)
        self._sell_vol = Decimal(0)

        self._prev_L1 = None
        self._prev_mid_sum = None
        self._last_mid_move_ts = None
        self._ofi_100.reset()
        self._ofi_500.reset()
        self._ofi_1000.reset()
        self._queue_delta_1000.reset()
        self._depth_count_1000.reset()

        self._micro_log_returns.clear()
        self._prev_micro = None

    # ------------------------------------------------------------------
    # Per-update hook
    # ------------------------------------------------------------------

    def observe_depth(
        self,
        ts: int,
        best_bid: Optional[int],
        best_bid_qty: int,
        best_ask: Optional[int],
        best_ask_qty: int,
    ) -> None:
        """Record a depth-update observation (post-apply L1 state).

        Called on every depth event and once at bootstrap to seed the
        previous L1.  When one side is empty (best_bid or best_ask is
        None), treats that side as (price=0, qty=0) for delta purposes.
        """
        # Empty-side sentinel: 0 is always strictly below any real tick price.
        bb = -1 if best_bid is None else best_bid
        ba = -1 if best_ask is None else best_ask
        bq = best_bid_qty
        aq = best_ask_qty

        if self._prev_L1 is not None:
            pbb, pbq, pba, paq = self._prev_L1
            e_b = self._level_flow(pbb, pbq, bb, bq)     # bid-side pressure (+)
            e_a = -self._level_flow(pba, paq, ba, aq, ask=True)  # ask pressure as neg of ask flow
            # Queue delta at L1, price-preserving only.
            q_delta_bid = (bq - pbq) if (bb == pbb and bb != -1) else 0
            q_delta_ask = (aq - paq) if (ba == pba and ba != -1) else 0
            self._ofi_100.push(ts, float(e_b + e_a))
            self._ofi_500.push(ts, float(e_b + e_a))
            self._ofi_1000.push(ts, float(e_b + e_a))
            self._queue_delta_1000.push(ts, float(q_delta_bid - q_delta_ask))
            self._depth_count_1000.push(ts, 1.0)

            # Mid-move detection — compare sum bb+ba (proxy for 2*mid).
            if bb != -1 and ba != -1:
                mid_sum = bb + ba
                if self._prev_mid_sum is None or mid_sum != self._prev_mid_sum:
                    self._last_mid_move_ts = ts
                self._prev_mid_sum = mid_sum

        else:
            # Seed call — record state, no flow emitted.
            self._depth_count_1000.push(ts, 1.0)
            if bb != -1 and ba != -1:
                self._prev_mid_sum = bb + ba
                self._last_mid_move_ts = ts

        self._prev_L1 = (bb, bq, ba, aq)

    @staticmethod
    def _level_flow(p0: int, q0: int, p1: int, q1: int, *, ask: bool = False) -> int:
        """Cont-Kukanov-Stoikov L1 flow contribution for one side.

        Empty side is encoded as price ``-1``.  Transitions:

        * both empty           → 0
        * empty → present      → arrived at q1
        * present → empty      → retreated (-q0)

        Otherwise, for bid (``ask=False``) improvement is a higher price,
        for ask (``ask=True``) improvement is a lower price; same-price
        returns the qty delta.
        """
        if p0 == -1 and p1 == -1:
            return 0
        if p0 == -1:
            return q1
        if p1 == -1:
            return -q0
        if not ask:
            if p1 > p0:
                return q1
            if p1 == p0:
                return q1 - q0
            return -q0
        # ask side: lower price is an improvement
        if p1 < p0:
            return q1
        if p1 == p0:
            return q1 - q0
        return -q0

    # ------------------------------------------------------------------
    # Grid-node hook
    # ------------------------------------------------------------------

    def on_book_update(
        self,
        timestamp: int,
        orderbook: OrderBook,
        trades: Optional[list[Trade]] = None,
    ) -> Optional[FeatureSnapshot]:
        """Process a grid-node event. Return a snapshot if *timestamp* is a grid node.

        Trade buffers are updated on every call regardless of emission.
        """
        assert self._prev_ts is None or timestamp >= self._prev_ts, (
            f"timestamps not monotonic: {timestamp} < {self._prev_ts}"
        )
        self._prev_ts = timestamp

        if self._next_emit is None:
            self._next_emit = timestamp

        if trades:
            for t in trades:
                self._trades.add(t)
                if t.is_buyer_maker:
                    self._sell_vol += t.quantity
                else:
                    self._buy_vol += t.quantity

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
        signed_trade_volume = buy_vol - sell_vol

        # Realized vol on microprice, grid-cadence.
        rv: Optional[float] = None
        if microprice is not None and self._prev_micro is not None:
            try:
                ret = log(float(microprice) / float(self._prev_micro))
            except (ValueError, ZeroDivisionError):
                ret = 0.0
            self._micro_log_returns.append(ret)
        if microprice is not None:
            self._prev_micro = microprice
        if self._micro_log_returns:
            rv = sqrt(sum(r * r for r in self._micro_log_returns))

        if self._last_mid_move_ts is None:
            time_since_move = 0
        else:
            time_since_move = max(0, ts - self._last_mid_move_ts)

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
            ofi_100ms=self._ofi_100.value(ts),
            ofi_500ms=self._ofi_500.value(ts),
            ofi_1000ms=self._ofi_1000.value(ts),
            queue_delta_diff_1000ms=self._queue_delta_1000.value(ts),
            depth_update_count_1000ms=int(self._depth_count_1000.count(ts)),
            time_since_last_mid_move_ms=time_since_move,
            signed_trade_volume_1000ms=signed_trade_volume,
            realized_vol_microprice_1000ms=rv,
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

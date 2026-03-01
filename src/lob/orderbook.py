from __future__ import annotations

from sortedcontainers import SortedDict

Price = float
Qty = float


class OrderBook:
    """Limit order book maintained from depth stream snapshots/updates."""

    __slots__ = ("bids", "asks")

    def __init__(self) -> None:
        self.bids: SortedDict[Price, Qty] = SortedDict()
        self.asks: SortedDict[Price, Qty] = SortedDict()

    # ---- mutation --------------------------------------------------------

    def apply_snapshot(self, snapshot: dict) -> None:
        """Reset the book and load a full snapshot."""
        self.bids.clear()
        self.asks.clear()
        for price, qty in snapshot.get("bids", []):
            self.bids[price] = qty
        for price, qty in snapshot.get("asks", []):
            self.asks[price] = qty

    def apply_update(self, update: dict) -> None:
        """Apply an incremental update. qty == 0.0 removes the level."""
        for price, qty in update.get("bids", []):
            if qty == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty
        for price, qty in update.get("asks", []):
            if qty == 0.0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

    # ---- queries ---------------------------------------------------------

    def best_bid(self) -> Price | None:
        if not self.bids:
            return None
        return self.bids.peekitem(-1)[0]

    def best_ask(self) -> Price | None:
        if not self.asks:
            return None
        return self.asks.peekitem(0)[0]

    def midprice(self) -> float | None:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def spread(self) -> float | None:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def volume_at_best(self) -> tuple[Qty, Qty]:
        """Return (best_bid_qty, best_ask_qty). 0.0 if side is empty."""
        bid_qty = self.bids.peekitem(-1)[1] if self.bids else 0.0
        ask_qty = self.asks.peekitem(0)[1] if self.asks else 0.0
        return bid_qty, ask_qty

    def imbalance(self, levels: int = 5) -> float | None:
        """Top-N volume imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)."""
        if not self.bids or not self.asks:
            return None
        bid_vol = sum(v for _, v in self.bids.items()[:-levels - 1:-1]) if len(self.bids) >= 1 else 0.0
        ask_vol = sum(v for _, v in self.asks.items()[:levels]) if len(self.asks) >= 1 else 0.0
        total = bid_vol + ask_vol
        if total == 0.0:
            return 0.0
        return (bid_vol - ask_vol) / total

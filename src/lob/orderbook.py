from __future__ import annotations

from decimal import Decimal

from sortedcontainers import SortedDict

from src.config.config import SymbolConfig


class OrderBook:
    """Limit order book maintained from depth stream snapshots/updates.

    Prices and quantities are stored as integer tick/lot counts.
    Conversion from exchange strings happens at the boundary only.
    """

    __slots__ = ("bids", "asks", "cfg")

    def __init__(self, cfg: SymbolConfig) -> None:
        self.cfg = cfg
        self.bids: SortedDict[int, int] = SortedDict()
        self.asks: SortedDict[int, int] = SortedDict()

    # ---- conversion (boundary only) --------------------------------------

    def _to_ticks(self, price: str) -> int:
        return int(Decimal(price) * self.cfg.inv_tick)

    def _to_lots(self, qty: str) -> int:
        return int(Decimal(qty) * self.cfg.inv_step)
    
    def price_from_ticks(self, ticks: int) -> Decimal:
        return Decimal(ticks) / self.cfg.inv_tick

    def qty_from_lots(self, lots: int) -> Decimal:
        return Decimal(lots) / self.cfg.inv_step

    # ---- mutation --------------------------------------------------------

    def apply_snapshot(self, snapshot: dict) -> None:
        """Reset the book and load a full snapshot."""
        self.bids.clear()
        self.asks.clear()
        for price, qty in snapshot.get("bids", []):
            self.bids[self._to_ticks(price)] = self._to_lots(qty)
        for price, qty in snapshot.get("asks", []):
            self.asks[self._to_ticks(price)] = self._to_lots(qty)

    def apply_update(self, update: dict) -> None:
        """Apply an incremental update. qty == '0' removes the level."""
        for price, qty in update.get("bids", []):
            p = self._to_ticks(price)
            q = self._to_lots(qty)
            if q == 0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = q
        for price, qty in update.get("asks", []):
            p = self._to_ticks(price)
            q = self._to_lots(qty)
            if q == 0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = q

    # ---- queries ---------------------------------------------------------

    def best_bid(self) -> int | None:
        if not self.bids:
            return None
        return self.bids.peekitem(-1)[0]

    def best_ask(self) -> int | None:
        if not self.asks:
            return None
        return self.asks.peekitem(0)[0]
    
    def volume_at_best(self) -> tuple[int, int]:
        """Return (best_bid_qty, best_ask_qty). 0 if side is empty."""
        bid_qty = self.bids.peekitem(-1)[1] if self.bids else 0
        ask_qty = self.asks.peekitem(0)[1] if self.asks else 0
        return bid_qty, ask_qty

    def midprice(self) -> Decimal | None:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return self.price_from_ticks(bb + ba) / 2

    def microprice(self) -> Decimal | None:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        bid_qty, ask_qty = self.volume_at_best()
        total = bid_qty + ask_qty
        if total == 0:
            return None
        # weighted average in tick space, then convert once
        return self.price_from_ticks(bb * ask_qty + ba * bid_qty) / total

    def spread(self) -> int | None:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def validate(self) -> list[str]:
        """Check structural integrity. Returns list of error strings (empty == ok)."""
        errors = []
        for price, qty in self.bids.items():
            if qty <= 0:
                errors.append(f"non-positive bid qty: {qty} for price {price}")
        for price, qty in self.asks.items():
            if qty <= 0:
                errors.append(f"non-positive ask qty: {qty} for price {price}")
        if self.bids and self.asks:
            bb = self.bids.peekitem(-1)[0]
            ba = self.asks.peekitem(0)[0]
            if bb >= ba:
                errors.append(f"crossed book: best_bid={bb} >= best_ask={ba}")
        return errors

    def imbalance(self, levels: int = 5) -> float:
        """Top-N volume imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)."""
        # top N bids: last `levels` items in ascending-sorted dict (highest prices)
        top_bids = self.bids.values()[-levels:]
        bid_vol = sum(top_bids)

        # top N asks: first `levels` items (lowest prices)
        top_asks = self.asks.values()[:levels]
        ask_vol = sum(top_asks)

        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

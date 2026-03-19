from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from urllib.request import urlopen

logger = logging.getLogger(__name__)

_EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo?symbol={}"


@dataclass(frozen=True, slots=True)
class SymbolConfig:
    symbol: str
    tick_size: Decimal
    step_size: Decimal
    inv_tick: int
    inv_step: int


def fetch_symbol_config(symbol: str) -> SymbolConfig:
    """Fetch tick_size and step_size from Binance REST API."""
    url = _EXCHANGE_INFO_URL.format(symbol.upper())
    with urlopen(url) as resp:
        data = json.loads(resp.read())

    filters = {f["filterType"]: f for f in data["symbols"][0]["filters"]}
    tick = Decimal(filters["PRICE_FILTER"]["tickSize"]).normalize()
    step = Decimal(filters["LOT_SIZE"]["stepSize"]).normalize()
    logger.info("Fetched %s config: tick_size=%s, step_size=%s", symbol, tick, step)

    return SymbolConfig(
        symbol=symbol.upper(),
        tick_size=tick,
        step_size=step,
        inv_tick=int(Decimal("1") / tick),
        inv_step=int(Decimal("1") / step),
    )

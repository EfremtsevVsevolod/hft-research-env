from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class SymbolConfig:
    symbol: str
    tick_size: Decimal
    step_size: Decimal
    inv_tick: int
    inv_step: int


def load_symbols(
    path: str | Path = "config/symbols.yaml",
) -> dict[str, SymbolConfig]:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    configs: dict[str, SymbolConfig] = {}
    for symbol, vals in raw.items():
        tick = Decimal(vals["tick_size"])
        step = Decimal(vals["step_size"])
        configs[symbol] = SymbolConfig(
            symbol=symbol,
            tick_size=tick,
            step_size=step,
            inv_tick=int(Decimal("1") / tick),
            inv_step=int(Decimal("1") / step),
        )
    return configs

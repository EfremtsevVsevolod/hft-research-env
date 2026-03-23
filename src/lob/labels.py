"""Forward-looking label construction for supervised learning.

Pairs each FeatureSnapshot with a midprice-change label computed from
a future snapshot after a fixed time horizon.  Designed to work with
the regular grid produced by FeatureExtractor.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from .features import FeatureSnapshot


@dataclass(frozen=True, slots=True)
class LabelledSnapshot:
    """A snapshot paired with its forward-looking label."""
    snapshot: FeatureSnapshot
    label: Decimal  # future_midprice - snapshot.midprice


class LabelBuilder:
    """Compute forward midprice-change labels with a fixed horizon.

    Buffers every grid-node snapshot to maintain correct alignment.
    Labels are dropped (not the snapshots) when midprice is None.

    The horizon must be a multiple of the sampling interval so that
    it aligns exactly with a future grid node.

    Usage::

        lb = LabelBuilder(horizon_ms=200, sampling_interval_ms=50)

        for snap in snapshots:
            labelled = lb.on_snapshot(snap)
            if labelled is not None:
                store(labelled)
    """

    def __init__(self, horizon_ms: int = 200, sampling_interval_ms: int = 50) -> None:
        assert horizon_ms % sampling_interval_ms == 0, (
            f"horizon_ms ({horizon_ms}) must be a multiple of "
            f"sampling_interval_ms ({sampling_interval_ms})"
        )
        self.horizon = horizon_ms
        self._buffer: deque[FeatureSnapshot] = deque()

    def reset(self) -> None:
        """Clear buffered snapshots. Call after a book reset / sequence gap."""
        self._buffer.clear()

    def on_snapshot(self, snap: FeatureSnapshot) -> Optional[LabelledSnapshot]:
        """Process a snapshot and return a labelled pair if horizon elapsed."""
        self._buffer.append(snap)

        past = self._buffer[0]
        if snap.timestamp - past.timestamp < self.horizon:
            return None

        # past has no midprice — can never be labeled, discard
        if past.midprice is None:
            self._buffer.popleft()
            return None

        # future midprice missing — keep past, retry next grid node
        if snap.midprice is None:
            return None

        self._buffer.popleft()
        return LabelledSnapshot(
            snapshot=past,
            label=snap.midprice - past.midprice,
        )

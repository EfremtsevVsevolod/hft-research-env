from __future__ import annotations

from decimal import Decimal
from typing import Optional

import pytest

from src.lob.features import FeatureSnapshot
from src.lob.labels import LabelBuilder


def _snap(ts: int, midprice: Optional[Decimal] = Decimal("100.00")) -> FeatureSnapshot:
    """Minimal FeatureSnapshot for label testing."""
    return FeatureSnapshot(
        timestamp=ts,
        midprice=midprice,
        spread=Decimal("0.01") if midprice else None,
        imbalance_1=0.0,
        imbalance_5=0.0,
        imbalance_10=0.0,
        microprice=midprice,
        microprice_minus_mid=Decimal("0") if midprice else None,
        delta_midprice=None if midprice is None else Decimal("0"),
        buy_volume=Decimal("0"),
        sell_volume=Decimal("0"),
    )


class TestLabelEmission:
    def test_no_label_before_horizon(self):
        lb = LabelBuilder(horizon_ms=200, sampling_interval_ms=100)
        assert lb.on_snapshot(_snap(0)) is None
        assert lb.on_snapshot(_snap(100)) is None  # only 100ms elapsed

    def test_label_emitted_at_horizon(self):
        lb = LabelBuilder(horizon_ms=200, sampling_interval_ms=100)
        lb.on_snapshot(_snap(0, Decimal("100.00")))
        lb.on_snapshot(_snap(100))
        result = lb.on_snapshot(_snap(200, Decimal("100.05")))
        assert result is not None
        assert result.snapshot.timestamp == 0
        assert result.label == Decimal("0.05")

    def test_label_value_exact(self):
        lb = LabelBuilder(horizon_ms=100, sampling_interval_ms=100)
        lb.on_snapshot(_snap(0, Decimal("50.00")))
        result = lb.on_snapshot(_snap(100, Decimal("50.03")))
        assert result.label == Decimal("0.03")


class TestNoneMidprice:
    def test_none_past_midprice_discarded(self):
        lb = LabelBuilder(horizon_ms=100, sampling_interval_ms=100)
        lb.on_snapshot(_snap(0, midprice=None))
        result = lb.on_snapshot(_snap(100, Decimal("100.00")))
        assert result is None  # past had None midprice

    def test_none_future_midprice_retried(self):
        lb = LabelBuilder(horizon_ms=100, sampling_interval_ms=100)
        lb.on_snapshot(_snap(0, Decimal("100.00")))
        # Future midprice is None — past should be kept
        result = lb.on_snapshot(_snap(100, midprice=None))
        assert result is None

        # Next grid node has valid midprice — label should now be emitted
        result = lb.on_snapshot(_snap(200, Decimal("100.10")))
        assert result is not None
        assert result.snapshot.timestamp == 0
        assert result.label == Decimal("0.10")


class TestInvalidConfig:
    def test_horizon_not_multiple_of_interval(self):
        with pytest.raises(AssertionError, match="must be a multiple"):
            LabelBuilder(horizon_ms=150, sampling_interval_ms=100)


class TestReset:
    def test_reset_clears_buffer(self):
        lb = LabelBuilder(horizon_ms=100, sampling_interval_ms=100)
        lb.on_snapshot(_snap(0))
        lb.reset()
        # After reset, buffer is empty — need new horizon period
        assert lb.on_snapshot(_snap(1000)) is None
        result = lb.on_snapshot(_snap(1100, Decimal("100.05")))
        assert result is not None

"""Convert a stream of LabelledSnapshots into a tabular ML dataset."""

from __future__ import annotations

from math import isfinite
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.lob.labels import LabelledSnapshot

# Columns included in the dataset (order matters).
_COLUMNS = [
    "timestamp",
    "spread",
    "imbalance_1",
    "imbalance_5",
    "imbalance_10",
    "microprice_minus_mid",
    "delta_midprice",
    "buy_volume",
    "sell_volume",
    "label",
]

# FeatureSnapshot fields that are Optional — row is dropped if any is None.
_REQUIRED_FEATURES = ("spread", "microprice_minus_mid", "delta_midprice")


class DatasetBuilder:
    """Accumulate LabelledSnapshots and produce a pandas DataFrame.

    Usage::

        db = DatasetBuilder()

        for labelled in labelled_stream:
            db.on_labelled_snapshot(labelled)

        df = db.to_dataframe()
        db.save_parquet(Path("dataset.parquet"))
    """

    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._prev_ts: int | None = None

    def __len__(self) -> int:
        return len(self._rows)

    def reset_timestamp(self) -> None:
        """Reset timestamp tracking. Call after a book reset / sequence gap."""
        self._prev_ts = None

    def on_labelled_snapshot(self, labelled: LabelledSnapshot) -> None:
        """Extract features, validate, and append a row."""
        snap = labelled.snapshot

        # Validate strictly increasing timestamps (snapshots are on distinct grid nodes).
        assert self._prev_ts is None or snap.timestamp > self._prev_ts, (
            f"timestamps not strictly increasing: {snap.timestamp} <= {self._prev_ts}"
        )
        self._prev_ts = snap.timestamp

        # Drop rows where any required feature is None.
        if any(getattr(snap, f) is None for f in _REQUIRED_FEATURES):
            return

        row = {
            "timestamp": snap.timestamp,
            "spread": float(snap.spread),
            "imbalance_1": snap.imbalance_1,
            "imbalance_5": snap.imbalance_5,
            "imbalance_10": snap.imbalance_10,
            "microprice_minus_mid": float(snap.microprice_minus_mid),
            "delta_midprice": float(snap.delta_midprice),
            "buy_volume": float(snap.buy_volume),
            "sell_volume": float(snap.sell_volume),
            "label": float(labelled.label),
        }

        assert all(isfinite(v) for v in row.values() if isinstance(v, float)), (
            f"non-finite float in row at timestamp {snap.timestamp}"
        )

        self._rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        """Return accumulated rows as a DataFrame."""
        return pd.DataFrame(self._rows, columns=_COLUMNS)

    def save_parquet(self, path: Path, metadata: dict | None = None) -> None:
        """Save the dataset to a Parquet file with optional metadata."""
        table = pa.Table.from_pandas(self.to_dataframe())
        if metadata:
            existing = table.schema.metadata or {}
            existing.update({k.encode(): str(v).encode() for k, v in metadata.items()})
            table = table.replace_schema_metadata(existing)
        pq.write_table(table, path)

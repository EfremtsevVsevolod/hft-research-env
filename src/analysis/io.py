"""Shared IO helpers for analysis scripts and notebooks.

Centralises dataset parquet loading and metadata decoding so that
scripts/validate_dataset.py, scripts/compare_datasets.py, and
src/analysis/market_context.py use the same logic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def decode_parquet_metadata(table: pq.lib.Table) -> dict[str, str]:
    """Decode user-defined key-value metadata from a PyArrow table.

    Filters out internal ``pandas`` and ``ARROW`` keys.
    """
    raw = table.schema.metadata or {}
    return {
        k.decode(): v.decode()
        for k, v in raw.items()
        if not k.startswith(b"pandas") and not k.startswith(b"ARROW")
    }


def load_dataset_with_meta(
    path: Path | str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load a built dataset parquet with its metadata.

    Returns ``(DataFrame, metadata_dict)``.  Metadata keys/values are decoded
    strings.  A UTC ``datetime`` column is always added.
    """
    path = Path(path)
    table = pq.read_table(path)
    df = table.to_pandas()
    meta = decode_parquet_metadata(table)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df, meta

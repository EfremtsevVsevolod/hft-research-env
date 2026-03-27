"""Dataset column schema constants.

Single source of truth for the column layout of built ML datasets.
"""

TIMESTAMP_COL = "timestamp"
LABEL_COL = "label"

FEATURE_COLUMNS = [
    "spread",
    "imbalance_1",
    "imbalance_5",
    "imbalance_10",
    "microprice_minus_mid",
    "delta_midprice",
    "buy_volume",
    "sell_volume",
]

DATASET_COLUMNS = [TIMESTAMP_COL] + FEATURE_COLUMNS + [LABEL_COL]

# FeatureSnapshot fields that are Optional — row is dropped if any is None.
REQUIRED_FEATURE_COLUMNS = ("spread", "microprice_minus_mid", "delta_midprice")

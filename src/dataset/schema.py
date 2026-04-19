"""Dataset column schema constants.

Single source of truth for the column layout of built ML datasets.
"""

TIMESTAMP_COL = "timestamp"
LABEL_COL = "label"

OLD_FEATURE_COLUMNS = [
    "spread",
    "imbalance_1",
    "imbalance_5",
    "imbalance_10",
    "microprice_minus_mid",
    "delta_midprice",
    "buy_volume",
    "sell_volume",
]

NEW_FEATURE_COLUMNS = [
    "ofi_100ms",
    "ofi_500ms",
    "ofi_1000ms",
    "queue_delta_diff_1000ms",
    "depth_update_count_1000ms",
    "time_since_last_mid_move_ms",
    "signed_trade_volume_1000ms",
    "realized_vol_microprice_1000ms",
]

FEATURE_COLUMNS = OLD_FEATURE_COLUMNS + NEW_FEATURE_COLUMNS

DATASET_COLUMNS = [TIMESTAMP_COL] + FEATURE_COLUMNS + [LABEL_COL]

# FeatureSnapshot fields that are Optional — row is dropped if any is None.
REQUIRED_FEATURE_COLUMNS = (
    "spread",
    "microprice_minus_mid",
    "delta_midprice",
    "realized_vol_microprice_1000ms",
)

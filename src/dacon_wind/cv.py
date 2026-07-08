from __future__ import annotations

import pandas as pd


def split_train_valid_2024(dt_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return time-based masks for training before 2024 and validating on 2024."""
    years = pd.to_datetime(dt_series).dt.year
    train_mask = years < 2024
    valid_mask = years == 2024

    if not train_mask.any():
        raise ValueError("No training rows found for year < 2024.")
    if not valid_mask.any():
        raise ValueError("No validation rows found for year == 2024.")

    return train_mask, valid_mask

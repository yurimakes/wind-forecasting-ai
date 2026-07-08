from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TARGET_COLS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")

CAPACITY_KWH = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0,
}


def calculate_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, Any]:
    """Calculate the official DACON wind competition metric.

    Only rows where actual generation is at least 10% of the group's capacity
    are evaluated.
    """
    if not isinstance(y_true, pd.DataFrame) or not isinstance(y_pred, pd.DataFrame):
        raise TypeError("y_true and y_pred must both be pandas DataFrames.")

    missing_true = [col for col in TARGET_COLS if col not in y_true.columns]
    missing_pred = [col for col in TARGET_COLS if col not in y_pred.columns]
    if missing_true or missing_pred:
        raise ValueError(
            f"Missing target columns. y_true missing={missing_true}, "
            f"y_pred missing={missing_pred}"
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same number of rows: "
            f"{len(y_true)} != {len(y_pred)}"
        )

    group_nmae: dict[str, float] = {}
    group_ficr: dict[str, float] = {}

    for col in TARGET_COLS:
        actual = y_true[col].to_numpy(dtype=float)
        forecast = y_pred[col].to_numpy(dtype=float)
        capacity = CAPACITY_KWH[col]

        valid = actual >= capacity * 0.10
        if not np.any(valid):
            raise ValueError(
                f"No valid rows for {col}: actual generation must be at least "
                f"10% of capacity ({capacity * 0.10})."
            )

        actual = actual[valid]
        forecast = forecast[valid]

        normalized_error = np.abs(forecast - actual) / capacity
        group_nmae[col] = float(np.mean(normalized_error))

        settlement_score = np.select(
            [normalized_error <= 0.06, normalized_error <= 0.08],
            [4.0, 3.0],
            default=0.0,
        )
        earned_settlement = float(np.sum(actual * settlement_score))
        max_settlement = float(np.sum(actual * 4.0))
        group_ficr[col] = earned_settlement / max_settlement

    average_group_nmae = float(np.mean(list(group_nmae.values())))
    one_minus_nmae = 1.0 - average_group_nmae
    ficr = float(np.mean(list(group_ficr.values())))
    total_score = 0.5 * one_minus_nmae + 0.5 * ficr

    return {
        "total_score": float(total_score),
        "one_minus_nmae": float(one_minus_nmae),
        "ficr": float(ficr),
        "group_nmae": group_nmae,
        "group_ficr": group_ficr,
    }


def metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Return the official total score."""
    return calculate_metric(y_true, y_pred)["total_score"]


if __name__ == "__main__":
    y_true = pd.DataFrame(
        {
            "kpx_group_1": [2160.0, 4320.0, 1000.0],
            "kpx_group_2": [2160.0, 4320.0, 1000.0],
            "kpx_group_3": [2100.0, 4200.0, 1000.0],
        }
    )
    y_pred = y_true.copy()

    scores = calculate_metric(y_true, y_pred)
    assert scores["total_score"] == 1.0
    assert scores["one_minus_nmae"] == 1.0
    assert scores["ficr"] == 1.0
    print(scores)

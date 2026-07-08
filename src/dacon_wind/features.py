from __future__ import annotations

import numpy as np
import pandas as pd

from dacon_wind.metric import TARGET_COLS


def aggregate_weather(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Mean-aggregate gridded weather rows by forecast timestamp."""
    drop_cols = {"data_available_kst_dtm", "grid_id", "latitude", "longitude"}
    value_cols = [c for c in df.columns if c not in {"forecast_kst_dtm", *drop_cols}]

    agg = df.groupby("forecast_kst_dtm", sort=True)[value_cols].mean()
    agg.columns = [f"{prefix}_{c}_mean" for c in agg.columns]
    return agg.reset_index()


def calendar_features(dt_series: pd.Series) -> pd.DataFrame:
    """Calendar features from the baseline notebook."""
    dt = pd.to_datetime(dt_series)
    out = pd.DataFrame(index=dt.index)
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def build_baseline_train_frame(
    labels: pd.DataFrame,
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
) -> pd.DataFrame:
    """Join labels with baseline aggregated LDAPS and GFS features."""
    weather = aggregate_weather(ldaps, "ldaps").merge(
        aggregate_weather(gfs, "gfs"),
        on="forecast_kst_dtm",
        how="inner",
    )
    base = labels.rename(columns={"kst_dtm": "forecast_kst_dtm"})
    return base.merge(weather, on="forecast_kst_dtm", how="left")


def build_baseline_test_frame(
    sample_submission: pd.DataFrame,
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
) -> pd.DataFrame:
    """Join sample submission rows with baseline aggregated LDAPS and GFS features."""
    weather = aggregate_weather(ldaps, "ldaps").merge(
        aggregate_weather(gfs, "gfs"),
        on="forecast_kst_dtm",
        how="inner",
    )
    return sample_submission.merge(weather, on="forecast_kst_dtm", how="left")


def build_feature_matrix(train_frame: pd.DataFrame) -> pd.DataFrame:
    """Build the model matrix used by the baseline notebook."""
    drop_cols = ["forecast_kst_dtm", "forecast_id", *TARGET_COLS]
    weather_features = train_frame.drop(
        columns=[col for col in drop_cols if col in train_frame.columns]
    )

    return pd.concat(
        [calendar_features(train_frame["forecast_kst_dtm"]), weather_features],
        axis=1,
    )

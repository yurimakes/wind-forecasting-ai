from __future__ import annotations

import numpy as np
import pandas as pd

from dacon_wind.metric import TARGET_COLS


WIND_VECTOR_PAIRS = (
    (
        "ldaps_heightAboveGround_10_10u_mean",
        "ldaps_heightAboveGround_10_10v_mean",
        "ldaps_h10",
    ),
    (
        "gfs_heightAboveGround_10_10u_mean",
        "gfs_heightAboveGround_10_10v_mean",
        "gfs_h10",
    ),
    (
        "gfs_heightAboveGround_80_u_mean",
        "gfs_heightAboveGround_80_v_mean",
        "gfs_h80",
    ),
    (
        "gfs_heightAboveGround_100_100u_mean",
        "gfs_heightAboveGround_100_100v_mean",
        "gfs_h100",
    ),
    (
        "gfs_isobaricInhPa_850_u_mean",
        "gfs_isobaricInhPa_850_v_mean",
        "gfs_850hpa",
    ),
    (
        "gfs_isobaricInhPa_700_u_mean",
        "gfs_isobaricInhPa_700_v_mean",
        "gfs_700hpa",
    ),
    (
        "gfs_isobaricInhPa_500_u_mean",
        "gfs_isobaricInhPa_500_v_mean",
        "gfs_500hpa",
    ),
)

WEATHER_AGG_STATS = ("min", "max", "std", "range", "q25", "q75")

WEATHER_AGG_COLUMN_GROUPS = (
    ("ldaps_all", "ldaps_"),
    ("gfs_all", "gfs_"),
    ("ldaps_h10", "ldaps_heightAboveGround_10"),
    ("gfs_h10", "gfs_heightAboveGround_10"),
    ("gfs_h80", "gfs_heightAboveGround_80"),
    ("gfs_h100", "gfs_heightAboveGround_100"),
    ("gfs_850hpa", "gfs_isobaricInhPa_850"),
    ("gfs_700hpa", "gfs_isobaricInhPa_700"),
    ("gfs_500hpa", "gfs_isobaricInhPa_500"),
    ("ldaps_surface", "ldaps_surface"),
    ("gfs_surface", "gfs_surface"),
)


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


def build_weather_agg_feature_matrix(train_frame: pd.DataFrame) -> pd.DataFrame:
    """Build baseline features plus row-wise LDAPS/GFS aggregate statistics.

    This starts from the same calendar and mean-aggregated weather features as
    the tuned LightGBM baseline, then summarizes already-available LDAPS/GFS
    columns within each forecast timestamp row. It does not add wind vector
    derivatives or use information outside the baseline train/test frame.
    """
    baseline = build_feature_matrix(train_frame)
    weather_agg = _build_row_weather_aggregates(train_frame)
    return pd.concat([baseline, weather_agg], axis=1)


def _build_row_weather_aggregates(train_frame: pd.DataFrame) -> pd.DataFrame:
    """Create safe row-wise numeric aggregates from baseline weather columns."""
    out = pd.DataFrame(index=train_frame.index)

    numeric_weather_cols = [
        col
        for col in train_frame.select_dtypes(include=[np.number]).columns
        if col.startswith(("ldaps_", "gfs_"))
    ]

    for group_name, prefix in WEATHER_AGG_COLUMN_GROUPS:
        cols = [col for col in numeric_weather_cols if col.startswith(prefix)]
        if len(cols) < 2:
            continue

        values = train_frame.loc[:, cols].astype(float)
        out[f"{group_name}_row_min"] = values.min(axis=1)
        out[f"{group_name}_row_max"] = values.max(axis=1)
        out[f"{group_name}_row_std"] = values.std(axis=1, ddof=0)
        out[f"{group_name}_row_range"] = (
            out[f"{group_name}_row_max"] - out[f"{group_name}_row_min"]
        )
        out[f"{group_name}_row_q25"] = values.quantile(0.25, axis=1)
        out[f"{group_name}_row_q75"] = values.quantile(0.75, axis=1)

    return out


def add_wind_derived_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add wind vector features from already-aggregated u/v columns."""
    out = features.copy()
    missing = [
        col
        for u_col, v_col, _ in WIND_VECTOR_PAIRS
        for col in (u_col, v_col)
        if col not in out.columns
    ]
    if missing:
        raise ValueError(f"Missing wind vector columns: {missing}")

    eps = np.finfo(float).eps
    for u_col, v_col, name in WIND_VECTOR_PAIRS:
        u = out[u_col].astype(float)
        v = out[v_col].astype(float)
        speed = np.sqrt((u * u) + (v * v))
        denom = speed.mask(speed <= eps, np.nan)

        out[f"{name}_wind_speed"] = speed
        out[f"{name}_wind_dir_u"] = (u / denom).fillna(0.0)
        out[f"{name}_wind_dir_v"] = (v / denom).fillna(0.0)
        out[f"{name}_wind_speed_sq"] = speed * speed
        out[f"{name}_wind_speed_cubed"] = speed * speed * speed

    return out


def build_wind_feature_matrix(train_frame: pd.DataFrame) -> pd.DataFrame:
    """Build baseline features plus wind vector derivatives for LightGBM v2."""
    return add_wind_derived_features(build_feature_matrix(train_frame))

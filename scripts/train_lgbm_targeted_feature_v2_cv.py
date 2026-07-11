from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dacon_wind.cv import split_train_valid_2024
from dacon_wind.data import load_train_data
from dacon_wind.features import (
    TARGETED_COMPONENT_DIFFS,
    TARGETED_WIND_VECTOR_PAIRS,
    build_baseline_train_frame,
    build_feature_matrix,
    build_targeted_weather_feature_matrix,
)
from dacon_wind.metric import CAPACITY_KWH, TARGET_COLS, calculate_metric
from dacon_wind.models import LGBMResult


EXP_ID = "lgbm_010_targeted_feature_v2_cv"
RUN_ID = "lgbm_010_targeted_feature_v2_cv_valid_2024"
RANDOM_SEED = 42
SCALES = (1.00, 1.03, 1.08, 1.095, 1.10)
SCORE_EPS = 1e-10

ENS_001_VALIDATION_REFERENCE = {
    "exp_id": "ens_001_simple_avg",
    "total_score": 0.6037465036,
    "one_minus_nmae": 0.8680552297,
    "ficr": 0.3394377775,
}

LGBM_008_VALIDATION_REFERENCE = {
    "exp_id": "lgbm_008_scale_upper_sweep_cv",
    "selected_scale": 1.10,
    "total_score": 0.6184267418,
    "one_minus_nmae": 0.8671491370,
    "ficr": 0.3697043467,
}

CURRENT_BEST_PUBLIC_REFERENCE = {
    "submission": "lgbm_008_scale_110_submit",
    "total_score": 0.6211026154,
    "one_minus_nmae": 0.8634833072,
    "ficr": 0.3787219236,
}

BEST_PARAMS: dict[str, Any] = {
    "learning_rate": 0.03,
    "n_estimators": 1000,
    "num_leaves": 15,
    "max_depth": -1,
    "min_child_samples": 20,
    "reg_lambda": 5.0,
    "reg_alpha": 0.0,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "objective": "regression",
    "random_state": RANDOM_SEED,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": -1,
    "verbosity": -1,
}


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_new_artifacts(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = "\n".join(f"- {path.relative_to(ROOT)}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing artifact(s):\n"
            f"{formatted}\n"
            "Remove or rename them before rerunning this validation script."
        )


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    eps = np.finfo(float).eps
    return numerator / denominator.mask(denominator.abs() <= eps, np.nan)


def add_feature(
    out: pd.DataFrame,
    metadata: dict[str, list[dict[str, Any]]],
    feature_name: str,
    values: pd.Series,
    sources: list[str],
    family: str,
) -> None:
    out[feature_name] = values
    metadata["created"].append(
        {"feature": feature_name, "sources": sources, "family": family}
    )


def skip_feature(
    metadata: dict[str, list[dict[str, Any]]],
    feature_name: str,
    required_sources: list[str],
    family: str,
) -> None:
    metadata["skipped"].append(
        {
            "feature": feature_name,
            "missing_sources": required_sources,
            "family": family,
        }
    )


def build_targeted_feature_v2_matrix(
    train_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    features = build_targeted_weather_feature_matrix(train_frame)
    out = pd.DataFrame(index=features.index)
    metadata: dict[str, list[dict[str, Any]]] = {"created": [], "skipped": []}

    speed_cols = [
        "ldaps_h10_wind_speed",
        "gfs_h10_wind_speed",
        "gfs_h80_wind_speed",
        "gfs_h100_wind_speed",
    ]
    available_speed_cols: list[str] = []
    for col in speed_cols:
        if col not in features.columns:
            skip_feature(metadata, f"{col}_sq", [col], "wind_speed_power")
            skip_feature(metadata, f"{col}_cubed", [col], "wind_speed_power")
            continue
        available_speed_cols.append(col)
        speed = features[col].astype(float)
        add_feature(out, metadata, f"{col}_sq", speed * speed, [col], "wind_speed_power")
        add_feature(
            out,
            metadata,
            f"{col}_cubed",
            speed * speed * speed,
            [col],
            "wind_speed_power",
        )

    gust_col = "gfs_surface_gust"
    if gust_col in features.columns:
        gust = features[gust_col].astype(float)
        for col in available_speed_cols:
            speed = features[col].astype(float)
            add_feature(
                out,
                metadata,
                f"gfs_surface_gust_minus_{col}",
                gust - speed,
                [gust_col, col],
                "gust_margin",
            )
            add_feature(
                out,
                metadata,
                f"gfs_surface_gust_to_{col}_ratio",
                safe_ratio(gust, speed),
                [gust_col, col],
                "gust_ratio",
            )
    else:
        for col in speed_cols:
            skip_feature(metadata, f"gfs_surface_gust_minus_{col}", [gust_col, col], "gust_margin")
            skip_feature(metadata, f"gfs_surface_gust_to_{col}_ratio", [gust_col, col], "gust_ratio")

    comparable_pairs = [
        ("gfs_h10_wind_speed", "ldaps_h10_wind_speed", "h10_wind_speed"),
        (
            "gfs_heightAboveGround_10_10u_mean",
            "ldaps_heightAboveGround_10_10u_mean",
            "h10_u_component",
        ),
        (
            "gfs_heightAboveGround_10_10v_mean",
            "ldaps_heightAboveGround_10_10v_mean",
            "h10_v_component",
        ),
    ]
    for gfs_col, ldaps_col, name in comparable_pairs:
        if gfs_col not in features.columns or ldaps_col not in features.columns:
            missing = [col for col in (gfs_col, ldaps_col) if col not in features.columns]
            skip_feature(metadata, f"v2_ldaps_gfs_{name}_mean", missing, "ldaps_gfs_consensus")
            skip_feature(metadata, f"v2_gfs_minus_ldaps_{name}", missing, "ldaps_gfs_difference")
            skip_feature(metadata, f"v2_abs_gfs_minus_ldaps_{name}", missing, "ldaps_gfs_difference")
            continue
        gfs = features[gfs_col].astype(float)
        ldaps = features[ldaps_col].astype(float)
        diff = gfs - ldaps
        add_feature(
            out,
            metadata,
            f"v2_ldaps_gfs_{name}_mean",
            (gfs + ldaps) / 2.0,
            [gfs_col, ldaps_col],
            "ldaps_gfs_consensus",
        )
        add_feature(
            out,
            metadata,
            f"v2_gfs_minus_ldaps_{name}",
            diff,
            [gfs_col, ldaps_col],
            "ldaps_gfs_difference",
        )
        add_feature(
            out,
            metadata,
            f"v2_abs_gfs_minus_ldaps_{name}",
            diff.abs(),
            [gfs_col, ldaps_col],
            "ldaps_gfs_difference",
        )

    direction_cols = [col for col in features.columns if "direction" in col.lower()]
    if direction_cols:
        for col in direction_cols:
            radians = np.deg2rad(features[col].astype(float))
            add_feature(out, metadata, f"{col}_sin", np.sin(radians), [col], "wind_direction")
            add_feature(out, metadata, f"{col}_cos", np.cos(radians), [col], "wind_direction")
    else:
        skip_feature(
            metadata,
            "wind_direction_sin_cos",
            ["weather direction columns containing 'direction'"],
            "wind_direction",
        )

    hour = features["hour"].astype(float)
    month = features["month"].astype(float)
    season = ((features["month"].astype(int) % 12) // 3).astype(float)
    add_feature(
        out,
        metadata,
        "hour_x_month",
        hour * month,
        ["hour", "month"],
        "time_interaction",
    )
    add_feature(
        out,
        metadata,
        "hour_sin_x_month_sin",
        features["hour_sin"].astype(float) * features["month_sin"].astype(float),
        ["hour_sin", "month_sin"],
        "time_interaction",
    )
    add_feature(
        out,
        metadata,
        "hour_cos_x_month_cos",
        features["hour_cos"].astype(float) * features["month_cos"].astype(float),
        ["hour_cos", "month_cos"],
        "time_interaction",
    )
    add_feature(out, metadata, "season", season, ["month"], "time_interaction")
    add_feature(
        out,
        metadata,
        "hour_x_season",
        hour * season,
        ["hour", "month"],
        "time_interaction",
    )
    add_feature(
        out,
        metadata,
        "is_daytime_06_18",
        features["hour"].between(6, 18).astype(int),
        ["hour"],
        "day_night_flag",
    )
    add_feature(
        out,
        metadata,
        "is_nighttime_19_05",
        (~features["hour"].between(6, 18)).astype(int),
        ["hour"],
        "day_night_flag",
    )

    ramp_cols = [
        col
        for col in [
            "ldaps_h10_wind_speed",
            "gfs_h10_wind_speed",
            "gfs_h80_wind_speed",
            "gfs_h100_wind_speed",
            "gfs_surface_gust",
        ]
        if col in features.columns
    ]
    ordered_index = pd.to_datetime(train_frame["forecast_kst_dtm"]).sort_values().index
    for col in ramp_cols:
        ordered_values = features.loc[ordered_index, col].astype(float)
        diff = ordered_values.diff().reindex(features.index)
        add_feature(
            out,
            metadata,
            f"{col}_diff_from_previous_forecast_hour",
            diff,
            [col, "forecast_kst_dtm"],
            "same_source_weather_ramp",
        )
    for col in set(speed_cols + [gust_col]) - set(ramp_cols):
        skip_feature(
            metadata,
            f"{col}_diff_from_previous_forecast_hour",
            [col],
            "same_source_weather_ramp",
        )

    return pd.concat([features, out], axis=1), metadata


def train_variant_models(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
) -> LGBMResult:
    imputer = SimpleImputer(strategy="median")
    x_train_imp = pd.DataFrame(
        imputer.fit_transform(x_train),
        columns=x_train.columns,
        index=x_train.index,
    )
    x_valid_imp = pd.DataFrame(
        imputer.transform(x_valid),
        columns=x_valid.columns,
        index=x_valid.index,
    )

    predictions = pd.DataFrame(index=x_valid.index)
    models: dict[str, LGBMRegressor] = {}
    train_rows: dict[str, int] = {}

    for target in TARGET_COLS:
        train_mask = y_train[target].notna()
        if not train_mask.any():
            raise ValueError(f"No non-null training labels for {target}.")

        model = LGBMRegressor(**BEST_PARAMS)
        model.fit(x_train_imp.loc[train_mask], y_train.loc[train_mask, target])

        pred = model.predict(x_valid_imp)
        predictions[target] = np.clip(pred, 0.0, CAPACITY_KWH[target])
        models[target] = model
        train_rows[target] = int(train_mask.sum())

    return LGBMResult(
        imputer=imputer,
        models=models,
        feature_columns=list(x_train.columns),
        predictions=predictions,
        train_rows=train_rows,
    )


def score_prediction(y_true: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    return calculate_metric(
        y_true.reset_index(drop=True),
        prediction.loc[:, list(TARGET_COLS)].reset_index(drop=True),
    )


def build_prediction_frame(
    forecast_times: pd.Series,
    prediction: pd.DataFrame,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {"forecast_kst_dtm": pd.to_datetime(forecast_times).reset_index(drop=True)}
    )
    for target in TARGET_COLS:
        out[target] = prediction[target].to_numpy(dtype=float)
    return out


def weighted_average(
    predictions: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    first = next(iter(predictions.values()))
    out = pd.DataFrame(index=first.index)
    for target in TARGET_COLS:
        values = sum(
            predictions[name][target].to_numpy(dtype=float) * weight
            for name, weight in weights.items()
        )
        out[target] = np.clip(values, 0.0, CAPACITY_KWH[target])
    return out


def scale_prediction(prediction: pd.DataFrame, scale: float) -> pd.DataFrame:
    out = prediction.copy()
    for target in TARGET_COLS:
        out[target] = (out[target] * scale).clip(0.0, CAPACITY_KWH[target])
    return out


def write_prediction(path: Path, prediction: pd.DataFrame) -> None:
    output = prediction.copy()
    output["forecast_kst_dtm"] = pd.to_datetime(output["forecast_kst_dtm"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    output.to_csv(path, index=False, encoding="utf-8-sig")


def format_score(scores: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_score": scores["total_score"],
        "one_minus_nmae": scores["one_minus_nmae"],
        "ficr": scores["ficr"],
        "group_nmae": scores["group_nmae"],
        "group_ficr": scores["group_ficr"],
    }


def candidate_uses_v2(candidate: dict[str, Any]) -> bool:
    if candidate["candidate"] == "lgbm_010_targeted_feature_v2":
        return True
    weights = candidate.get("weights", {})
    return weights.get("lgbm_010_targeted_feature_v2", 0.0) > 0.0


def final_decision(
    best_v2_including_raw: dict[str, Any],
    best_scaled: dict[str, Any],
) -> str:
    raw_useful = (
        best_v2_including_raw["total_score"]
        > ENS_001_VALIDATION_REFERENCE["total_score"] + SCORE_EPS
    )
    scaled_submission_candidate = (
        best_scaled["total_score"]
        > LGBM_008_VALIDATION_REFERENCE["total_score"] + SCORE_EPS
    )
    if scaled_submission_candidate:
        return (
            "feature set useful and conservative scaled validation beats lgbm_008; "
            "future submission candidate, no submission created"
        )
    if raw_useful:
        return (
            "v2-including raw candidate beats ens_001, but scaled score does not beat "
            "lgbm_008; keep lgbm_008_scale_110_submit as current best public reference"
        )
    return (
        "v2 features do not beat ens_001 validation and scaled score does not beat "
        "lgbm_008 beyond numerical tolerance; keep lgbm_008_scale_110_submit as "
        "current best public reference and continue feature/model experiments"
    )


def print_results(
    variant_scores: list[dict[str, Any]],
    ensemble_scores: list[dict[str, Any]],
    scale_scores: list[dict[str, Any]],
    best_raw: dict[str, Any],
    best_v2_including_raw: dict[str, Any],
    best_scaled: dict[str, Any],
    decision: str,
) -> None:
    print("per_variant_raw_scores")
    print("candidate,total_score,one_minus_nmae,ficr")
    for row in variant_scores:
        print(
            f"{row['candidate']},"
            f"{row['total_score']:.10f},"
            f"{row['one_minus_nmae']:.10f},"
            f"{row['ficr']:.10f}"
        )

    print("ensemble_weight_search_top_results")
    print("candidate,w_lgbm003,w_lgbm005,w_lgbm010,total_score,one_minus_nmae,ficr")
    for row in ensemble_scores[:10]:
        weights = row["weights"]
        print(
            f"{row['candidate']},"
            f"{weights.get('lgbm_003_tuned_baseline_recheck', 0.0):.2f},"
            f"{weights.get('lgbm_005_targeted_weather_recheck', 0.0):.2f},"
            f"{weights.get('lgbm_010_targeted_feature_v2', 0.0):.2f},"
            f"{row['total_score']:.10f},"
            f"{row['one_minus_nmae']:.10f},"
            f"{row['ficr']:.10f}"
        )

    print("conservative_global_scale_scores")
    print("scale,total_score,one_minus_nmae,ficr,beats_lgbm_008_validation")
    for row in scale_scores:
        print(
            f"{row['scale']:.3f},"
            f"{row['total_score']:.10f},"
            f"{row['one_minus_nmae']:.10f},"
            f"{row['ficr']:.10f},"
            f"{row['beats_lgbm_008_validation_total_score']}"
        )

    print(f"selected_best_raw_candidate: {best_raw['candidate']}")
    print(f"selected_best_raw_total_score: {best_raw['total_score']:.10f}")
    print(f"selected_best_v2_including_raw_candidate: {best_v2_including_raw['candidate']}")
    print(
        "selected_best_v2_including_raw_total_score: "
        f"{best_v2_including_raw['total_score']:.10f}"
    )
    print(f"selected_best_scaled_candidate: {best_scaled['candidate']}")
    print(f"selected_best_scaled_scale: {best_scaled['scale']:.3f}")
    print(f"selected_best_scaled_total_score: {best_scaled['total_score']:.10f}")
    print(
        "beats_lgbm_008_validation_reference: "
        f"{best_scaled['beats_lgbm_008_validation_total_score']}"
    )
    print(f"final_decision: {decision}")


def main() -> None:
    set_reproducible_seed(RANDOM_SEED)

    prediction_dir = ROOT / "outputs" / "predictions"
    log_dir = ROOT / "outputs" / "logs"
    model_dir = ROOT / "outputs" / "models"
    for path in (prediction_dir, log_dir, model_dir):
        path.mkdir(parents=True, exist_ok=True)

    raw_prediction_path = prediction_dir / f"{RUN_ID}_raw.csv"
    best_scaled_prediction_path = prediction_dir / f"{RUN_ID}_best_scaled.csv"
    log_path = log_dir / f"{RUN_ID}.json"
    model_path = model_dir / f"{RUN_ID}.joblib"
    ensure_new_artifacts(
        [raw_prediction_path, best_scaled_prediction_path, log_path, model_path]
    )

    data = load_train_data(ROOT / "data" / "raw" / "train")
    train_frame = build_baseline_train_frame(data.labels, data.ldaps, data.gfs)
    train_mask, valid_mask = split_train_valid_2024(train_frame["forecast_kst_dtm"])
    y_train = train_frame.loc[train_mask, list(TARGET_COLS)]
    y_valid = train_frame.loc[valid_mask, list(TARGET_COLS)]
    valid_times = train_frame.loc[valid_mask, "forecast_kst_dtm"]

    baseline_features = build_feature_matrix(train_frame)
    targeted_features = build_targeted_weather_feature_matrix(train_frame)
    v2_features, v2_metadata = build_targeted_feature_v2_matrix(train_frame)

    feature_variants = {
        "lgbm_003_tuned_baseline_recheck": {
            "features": baseline_features,
            "description": "existing baseline calendar + mean LDAPS/GFS feature matrix",
        },
        "lgbm_005_targeted_weather_recheck": {
            "features": targeted_features,
            "description": "existing targeted-weather feature matrix",
        },
        "lgbm_010_targeted_feature_v2": {
            "features": v2_features,
            "description": "baseline + targeted-weather + safe row-local v2 weather/time features",
        },
    }

    results: dict[str, LGBMResult] = {}
    predictions: dict[str, pd.DataFrame] = {}
    variant_scores: list[dict[str, Any]] = []

    for name, variant in feature_variants.items():
        features = variant["features"]
        result = train_variant_models(
            x_train=features.loc[train_mask],
            y_train=y_train,
            x_valid=features.loc[valid_mask],
        )
        results[name] = result
        predictions[name] = result.predictions.reset_index(drop=True)
        scores = score_prediction(y_valid, result.predictions)
        variant_scores.append(
            {
                "candidate": name,
                "feature_count": len(result.feature_columns),
                **format_score(scores),
            }
        )

    named_ensembles = [
        (
            "ensemble_050_lgbm003_050_lgbm010",
            {
                "lgbm_003_tuned_baseline_recheck": 0.5,
                "lgbm_010_targeted_feature_v2": 0.5,
            },
        ),
        (
            "ensemble_050_lgbm005_050_lgbm010",
            {
                "lgbm_005_targeted_weather_recheck": 0.5,
                "lgbm_010_targeted_feature_v2": 0.5,
            },
        ),
        (
            "ensemble_034_lgbm003_033_lgbm005_033_lgbm010",
            {
                "lgbm_003_tuned_baseline_recheck": 0.34,
                "lgbm_005_targeted_weather_recheck": 0.33,
                "lgbm_010_targeted_feature_v2": 0.33,
            },
        ),
    ]

    all_raw_candidates: dict[str, pd.DataFrame] = dict(predictions)
    ensemble_scores: list[dict[str, Any]] = []
    for candidate, weights in named_ensembles:
        pred = weighted_average(predictions, weights)
        all_raw_candidates[candidate] = pred
        scores = score_prediction(y_valid, pred)
        ensemble_scores.append({"candidate": candidate, "weights": weights, **format_score(scores)})

    names = [
        "lgbm_003_tuned_baseline_recheck",
        "lgbm_005_targeted_weather_recheck",
        "lgbm_010_targeted_feature_v2",
    ]
    for i in range(11):
        for j in range(11 - i):
            k = 10 - i - j
            weights = {
                names[0]: i / 10.0,
                names[1]: j / 10.0,
                names[2]: k / 10.0,
            }
            candidate = f"weight_search_{i:02d}_{j:02d}_{k:02d}"
            pred = weighted_average(predictions, weights)
            all_raw_candidates[candidate] = pred
            scores = score_prediction(y_valid, pred)
            ensemble_scores.append(
                {"candidate": candidate, "weights": weights, **format_score(scores)}
            )

    variant_scores_by_total = sorted(
        variant_scores,
        key=lambda item: item["total_score"],
        reverse=True,
    )
    ensemble_scores_by_total = sorted(
        ensemble_scores,
        key=lambda item: item["total_score"],
        reverse=True,
    )
    raw_score_rows = variant_scores + ensemble_scores
    best_raw = max(raw_score_rows, key=lambda item: item["total_score"])
    v2_including_raw_rows = [row for row in raw_score_rows if candidate_uses_v2(row)]
    best_v2_including_raw = max(
        v2_including_raw_rows,
        key=lambda item: item["total_score"],
    )
    best_raw_prediction = build_prediction_frame(valid_times, all_raw_candidates[best_raw["candidate"]])

    scale_scores: list[dict[str, Any]] = []
    scaled_predictions: dict[float, pd.DataFrame] = {}
    for scale in SCALES:
        scaled = scale_prediction(best_raw_prediction, scale)
        scaled_predictions[scale] = scaled
        scores = score_prediction(y_valid, scaled)
        scale_scores.append(
            {
                "candidate": f"{best_raw['candidate']}_global_scale_{scale:.3f}",
                "scale": scale,
                **format_score(scores),
                "beats_lgbm_008_validation_total_score": (
                    scores["total_score"]
                    > LGBM_008_VALIDATION_REFERENCE["total_score"] + SCORE_EPS
                ),
            }
        )

    scale_scores_by_total = sorted(
        scale_scores,
        key=lambda item: item["total_score"],
        reverse=True,
    )
    best_scaled = scale_scores_by_total[0]
    decision = final_decision(best_v2_including_raw, best_scaled)

    write_prediction(raw_prediction_path, best_raw_prediction)
    write_prediction(best_scaled_prediction_path, scaled_predictions[best_scaled["scale"]])

    baseline_feature_count = len(baseline_features.columns)
    targeted_feature_columns = list(targeted_features.columns[baseline_feature_count:])
    v2_added_feature_columns = list(v2_features.columns[len(targeted_features.columns):])

    summary = {
        "exp_id": EXP_ID,
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Validation-only targeted feature v2 LightGBM experiment. No test "
            "predictions and no submission CSV are created."
        ),
        "random_seed": RANDOM_SEED,
        "data": {
            "train_labels": "data/raw/train/train_labels.csv",
            "ldaps_train": "data/raw/train/ldaps_train.csv",
            "gfs_train": "data/raw/train/gfs_train.csv",
            "scope": "training data from data/raw/train only",
        },
        "validation_split": {
            "function": "split_train_valid_2024",
            "train": "year < 2024",
            "valid": "year == 2024",
            "train_rows": int(train_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
            "forecast_start": str(pd.to_datetime(valid_times).min()),
            "forecast_end": str(pd.to_datetime(valid_times).max()),
        },
        "feature_variants": {
            name: {
                "description": variant["description"],
                "feature_count": len(results[name].feature_columns),
                "feature_columns": results[name].feature_columns,
            }
            for name, variant in feature_variants.items()
        },
        "existing_targeted_weather_features": {
            "targeted_wind_vector_pairs": [
                {"u_col": u_col, "v_col": v_col, "name": name}
                for u_col, v_col, name in TARGETED_WIND_VECTOR_PAIRS
            ],
            "targeted_component_diffs": [
                {"left_col": left_col, "right_col": right_col, "name": name}
                for left_col, right_col, name in TARGETED_COMPONENT_DIFFS
            ],
            "targeted_feature_columns": targeted_feature_columns,
        },
        "targeted_feature_v2": {
            "created_features": v2_metadata["created"],
            "skipped_features": v2_metadata["skipped"],
            "added_feature_count": len(v2_added_feature_columns),
            "added_feature_columns": v2_added_feature_columns,
            "leakage_note": (
                "All v2 features are derived from current row weather/time fields or "
                "same-source previous forecast timestamp weather differences. No "
                "target values or test-period distribution statistics are used."
            ),
        },
        "preprocessing": {
            "missing_value_imputation": "SimpleImputer(strategy='median') fit on training rows only for each variant.",
            "prediction_clipping": "Each target prediction clipped to [0, group_capacity].",
        },
        "model": {
            "class": "lightgbm.LGBMRegressor",
            "per_target_models": list(TARGET_COLS),
            "params": BEST_PARAMS,
        },
        "per_variant_raw_scores": variant_scores_by_total,
        "ensemble_weight_search_scores": ensemble_scores_by_total,
        "best_raw_candidate": best_raw,
        "best_v2_including_raw_candidate": best_v2_including_raw,
        "conservative_global_scale_scores": scale_scores_by_total,
        "selected_best_scaled_candidate": best_scaled,
        "comparisons": {
            "vs_ens_001_simple_avg_validation": {
                "reference": ENS_001_VALIDATION_REFERENCE,
                "best_raw_total_score_delta": (
                    best_raw["total_score"] - ENS_001_VALIDATION_REFERENCE["total_score"]
                ),
                "best_raw_beats_reference": (
                    best_v2_including_raw["total_score"]
                    > ENS_001_VALIDATION_REFERENCE["total_score"] + SCORE_EPS
                ),
            },
            "vs_lgbm_008_validation_global_1p10": {
                "reference": LGBM_008_VALIDATION_REFERENCE,
                "best_scaled_total_score_delta": (
                    best_scaled["total_score"] - LGBM_008_VALIDATION_REFERENCE["total_score"]
                ),
                "best_scaled_beats_reference": (
                    best_scaled["total_score"]
                    > LGBM_008_VALIDATION_REFERENCE["total_score"] + SCORE_EPS
                ),
            },
            "vs_current_best_public_lgbm_008_scale_110_submit": {
                "reference": CURRENT_BEST_PUBLIC_REFERENCE,
                "best_scaled_total_score_minus_public_reference": (
                    best_scaled["total_score"] - CURRENT_BEST_PUBLIC_REFERENCE["total_score"]
                ),
            },
        },
        "final_decision": decision,
        "artifacts": {
            "raw_best_valid_predictions": str(raw_prediction_path.relative_to(ROOT)),
            "best_scaled_valid_predictions": str(
                best_scaled_prediction_path.relative_to(ROOT)
            ),
            "run_summary": str(log_path.relative_to(ROOT)),
            "model": str(model_path.relative_to(ROOT)),
        },
    }

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    joblib.dump(
        {
            "exp_id": EXP_ID,
            "run_id": RUN_ID,
            "random_seed": RANDOM_SEED,
            "models": {
                name: {
                    "imputer": result.imputer,
                    "models": result.models,
                    "feature_columns": result.feature_columns,
                    "train_rows": result.train_rows,
                }
                for name, result in results.items()
            },
            "model_params": BEST_PARAMS,
            "capacity_kwh": CAPACITY_KWH,
            "per_variant_raw_scores": variant_scores_by_total,
            "ensemble_weight_search_scores": ensemble_scores_by_total,
            "best_raw_candidate": best_raw,
            "best_v2_including_raw_candidate": best_v2_including_raw,
            "conservative_global_scale_scores": scale_scores_by_total,
            "selected_best_scaled_candidate": best_scaled,
            "targeted_feature_v2": summary["targeted_feature_v2"],
            "final_decision": decision,
        },
        model_path,
    )

    print_results(
        variant_scores_by_total,
        ensemble_scores_by_total,
        scale_scores_by_total,
        best_raw,
        best_v2_including_raw,
        best_scaled,
        decision,
    )
    print(f"Saved raw best validation predictions: {raw_prediction_path.relative_to(ROOT)}")
    print(
        "Saved best conservative scaled validation predictions: "
        f"{best_scaled_prediction_path.relative_to(ROOT)}"
    )
    print(f"Saved run summary: {log_path.relative_to(ROOT)}")
    print(f"Saved model artifact: {model_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

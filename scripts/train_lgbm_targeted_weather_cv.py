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


EXP_ID = "lgbm_005_targeted_weather"
RUN_ID = "lgbm_005_targeted_weather_valid_2024"
RANDOM_SEED = 42
BASELINE_LOCAL_TOTAL_SCORE = 0.6033279875

BEST_PARAMS: dict[str, Any] = {
    "num_leaves": 15,
    "min_child_samples": 20,
    "learning_rate": 0.03,
    "n_estimators": 1000,
    "reg_lambda": 5.0,
    "reg_alpha": 0.0,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "max_depth": -1,
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
            "Remove or rename them before rerunning this script."
        )


def train_targeted_weather_lgbm_models(
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


def main() -> None:
    set_reproducible_seed(RANDOM_SEED)

    prediction_dir = ROOT / "outputs" / "predictions"
    log_dir = ROOT / "outputs" / "logs"
    model_dir = ROOT / "outputs" / "models"
    for path in (prediction_dir, log_dir, model_dir):
        path.mkdir(parents=True, exist_ok=True)

    prediction_path = prediction_dir / f"{RUN_ID}.csv"
    log_path = log_dir / f"{RUN_ID}.json"
    model_path = model_dir / f"{RUN_ID}.joblib"
    ensure_new_artifacts([prediction_path, log_path, model_path])

    data = load_train_data(ROOT / "data" / "raw" / "train")
    train_frame = build_baseline_train_frame(data.labels, data.ldaps, data.gfs)
    baseline_features = build_feature_matrix(train_frame)
    features = build_targeted_weather_feature_matrix(train_frame)

    train_mask, valid_mask = split_train_valid_2024(train_frame["forecast_kst_dtm"])
    x_train = features.loc[train_mask]
    y_train = train_frame.loc[train_mask, list(TARGET_COLS)]
    x_valid = features.loc[valid_mask]
    y_valid = train_frame.loc[valid_mask, list(TARGET_COLS)]

    result = train_targeted_weather_lgbm_models(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
    )

    valid_predictions = train_frame.loc[valid_mask, ["forecast_kst_dtm"]].copy()
    for target in TARGET_COLS:
        valid_predictions[target] = result.predictions[target].to_numpy()
    valid_predictions["forecast_kst_dtm"] = pd.to_datetime(
        valid_predictions["forecast_kst_dtm"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    valid_predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    scores = calculate_metric(
        y_valid.reset_index(drop=True),
        result.predictions.reset_index(drop=True),
    )
    beats_lgbm_003_tuned = scores["total_score"] > BASELINE_LOCAL_TOTAL_SCORE
    baseline_feature_count = len(baseline_features.columns)
    targeted_feature_columns = result.feature_columns[baseline_feature_count:]

    summary = {
        "exp_id": EXP_ID,
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "data": {
            "train_labels": "data/raw/train/train_labels.csv",
            "ldaps_train": "data/raw/train/ldaps_train.csv",
            "gfs_train": "data/raw/train/gfs_train.csv",
        },
        "split": {
            "train": "year < 2024",
            "valid": "year == 2024",
            "train_rows": int(train_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
            "target_train_rows": result.train_rows,
        },
        "features": {
            "approach": (
                "Baseline calendar features plus mean-aggregated LDAPS/GFS by "
                "forecast_kst_dtm, with targeted wind-speed, vertical-shear, "
                "gust-margin, and LDAPS/GFS 10m wind difference features. "
                "No broad row-wise weather aggregations and no lgbm_002 broad "
                "wind-vector derivative feature set are used."
            ),
            "targeted_wind_vector_pairs": [
                {"u_col": u_col, "v_col": v_col, "name": name}
                for u_col, v_col, name in TARGETED_WIND_VECTOR_PAIRS
            ],
            "targeted_component_diffs": [
                {"left_col": left_col, "right_col": right_col, "name": name}
                for left_col, right_col, name in TARGETED_COMPONENT_DIFFS
            ],
            "feature_count": len(result.feature_columns),
            "targeted_feature_count": len(targeted_feature_columns),
            "targeted_feature_columns": targeted_feature_columns,
            "feature_columns": result.feature_columns,
        },
        "preprocessing": {
            "missing_value_imputation": "SimpleImputer(strategy='median') fit on training rows only.",
            "prediction_clipping": "Per target group, clipped to [0, group_capacity].",
        },
        "model": {
            "class": "lightgbm.LGBMRegressor",
            "per_target_models": list(TARGET_COLS),
            "params": BEST_PARAMS,
        },
        "artifacts": {
            "valid_predictions": str(prediction_path.relative_to(ROOT)),
            "metric_summary": str(log_path.relative_to(ROOT)),
            "model": str(model_path.relative_to(ROOT)),
        },
        "baseline_comparison": {
            "baseline_exp_id": "lgbm_003_tuned",
            "baseline_local_total_score": BASELINE_LOCAL_TOTAL_SCORE,
            "beats_lgbm_003_tuned": beats_lgbm_003_tuned,
        },
        "metrics": scores,
    }

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    joblib.dump(
        {
            "exp_id": EXP_ID,
            "run_id": RUN_ID,
            "random_seed": RANDOM_SEED,
            "imputer": result.imputer,
            "models": result.models,
            "feature_columns": result.feature_columns,
            "targeted_feature_columns": targeted_feature_columns,
            "model_params": BEST_PARAMS,
            "targeted_wind_vector_pairs": TARGETED_WIND_VECTOR_PAIRS,
            "targeted_component_diffs": TARGETED_COMPONENT_DIFFS,
            "capacity_kwh": CAPACITY_KWH,
            "metrics": scores,
            "baseline_comparison": summary["baseline_comparison"],
        },
        model_path,
    )

    print(f"total_score: {scores['total_score']:.10f}")
    print(f"one_minus_nmae: {scores['one_minus_nmae']:.10f}")
    print(f"ficr: {scores['ficr']:.10f}")
    print(f"beats_lgbm_003_tuned: {beats_lgbm_003_tuned}")
    print(f"group_nmae: {json.dumps(scores['group_nmae'], sort_keys=True)}")
    print(f"group_ficr: {json.dumps(scores['group_ficr'], sort_keys=True)}")
    print(f"Saved validation predictions: {prediction_path.relative_to(ROOT)}")
    print(f"Saved metric summary: {log_path.relative_to(ROOT)}")
    print(f"Saved model artifact: {model_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

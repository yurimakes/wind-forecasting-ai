from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dacon_wind.cv import split_train_valid_2024
from dacon_wind.data import load_train_data
from dacon_wind.features import build_baseline_train_frame, build_feature_matrix
from dacon_wind.metric import TARGET_COLS, calculate_metric
from dacon_wind.models import baseline_rf_params, train_baseline_rf_models


EXP_ID = "baseline_rf_001_valid_2024"
RANDOM_SEED = 42


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    set_reproducible_seed(RANDOM_SEED)

    prediction_dir = ROOT / "outputs" / "predictions"
    log_dir = ROOT / "outputs" / "logs"
    model_dir = ROOT / "outputs" / "models"
    for path in (prediction_dir, log_dir, model_dir):
        path.mkdir(parents=True, exist_ok=True)

    prediction_path = prediction_dir / f"{EXP_ID}.csv"
    log_path = log_dir / f"{EXP_ID}.json"
    model_path = model_dir / f"{EXP_ID}.joblib"

    data = load_train_data(ROOT / "data" / "raw" / "train")
    train_frame = build_baseline_train_frame(data.labels, data.ldaps, data.gfs)
    features = build_feature_matrix(train_frame)

    train_mask, valid_mask = split_train_valid_2024(train_frame["forecast_kst_dtm"])
    x_train = features.loc[train_mask]
    y_train = train_frame.loc[train_mask, list(TARGET_COLS)]
    x_valid = features.loc[valid_mask]
    y_valid = train_frame.loc[valid_mask, list(TARGET_COLS)]

    result = train_baseline_rf_models(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        random_state=RANDOM_SEED,
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
    summary = {
        "exp_id": EXP_ID,
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
            "approach": "Baseline notebook calendar features plus mean-aggregated LDAPS/GFS by forecast_kst_dtm.",
            "feature_count": len(result.feature_columns),
            "feature_columns": result.feature_columns,
        },
        "model": {
            "class": "sklearn.ensemble.RandomForestRegressor",
            "per_target_models": list(TARGET_COLS),
            "params": baseline_rf_params(random_state=RANDOM_SEED),
        },
        "artifacts": {
            "valid_predictions": str(prediction_path.relative_to(ROOT)),
            "metric_summary": str(log_path.relative_to(ROOT)),
            "model": str(model_path.relative_to(ROOT)),
        },
        "metrics": scores,
    }

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    joblib.dump(
        {
            "exp_id": EXP_ID,
            "random_seed": RANDOM_SEED,
            "imputer": result.imputer,
            "models": result.models,
            "feature_columns": result.feature_columns,
            "model_params": baseline_rf_params(random_state=RANDOM_SEED),
        },
        model_path,
    )

    print(f"total_score: {scores['total_score']:.10f}")
    print(f"one_minus_nmae: {scores['one_minus_nmae']:.10f}")
    print(f"ficr: {scores['ficr']:.10f}")
    print(f"group_nmae: {json.dumps(scores['group_nmae'], sort_keys=True)}")
    print(f"group_ficr: {json.dumps(scores['group_ficr'], sort_keys=True)}")
    print(f"Saved validation predictions: {prediction_path.relative_to(ROOT)}")
    print(f"Saved metric summary: {log_path.relative_to(ROOT)}")
    print(f"Saved model artifact: {model_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

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

from dacon_wind.data import load_test_data, load_train_data
from dacon_wind.features import (
    build_baseline_test_frame,
    build_baseline_train_frame,
    build_feature_matrix,
)
from dacon_wind.metric import CAPACITY_KWH, TARGET_COLS
from dacon_wind.models import lgbm_v1_params, train_lgbm_v1_models
from validate_submission import validate_submission


EXP_ID = "lgbm_001"
RUN_ID = "lgbm_001_submit_2025"
RANDOM_SEED = 42


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
            "Remove or rename them before rerunning this final submission script."
        )


def main() -> None:
    set_reproducible_seed(RANDOM_SEED)

    submission_dir = ROOT / "submissions"
    prediction_dir = ROOT / "outputs" / "predictions"
    log_dir = ROOT / "outputs" / "logs"
    model_dir = ROOT / "outputs" / "models"
    for path in (submission_dir, prediction_dir, log_dir, model_dir):
        path.mkdir(parents=True, exist_ok=True)

    submission_path = submission_dir / "lgbm_001.csv"
    prediction_path = prediction_dir / "lgbm_001_test_2025.csv"
    log_path = log_dir / "lgbm_001_submit_2025.json"
    model_path = model_dir / "lgbm_001_final.joblib"
    ensure_new_artifacts([submission_path, prediction_path, log_path, model_path])

    train_data = load_train_data(ROOT / "data" / "raw" / "train")
    test_data = load_test_data(
        test_dir=ROOT / "data" / "raw" / "test",
        sample_path=ROOT / "data" / "raw" / "sample_submission.csv",
    )

    train_frame = build_baseline_train_frame(
        train_data.labels,
        train_data.ldaps,
        train_data.gfs,
    )
    test_frame = build_baseline_test_frame(
        test_data.sample_submission,
        test_data.ldaps,
        test_data.gfs,
    )

    x_train = build_feature_matrix(train_frame)
    y_train = train_frame.loc[:, list(TARGET_COLS)]
    x_test = build_feature_matrix(test_frame).reindex(columns=x_train.columns)

    result = train_lgbm_v1_models(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_test,
        random_state=RANDOM_SEED,
    )

    submission = test_data.sample_submission.copy()
    for target in TARGET_COLS:
        submission[target] = np.clip(
            result.predictions[target].to_numpy(dtype=float),
            0.0,
            CAPACITY_KWH[target],
        )

    sample_columns = list(test_data.sample_submission.columns)
    submission = submission.loc[:, sample_columns]
    submission["forecast_kst_dtm"] = pd.to_datetime(
        submission["forecast_kst_dtm"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    submission.to_csv(submission_path, index=False, encoding="utf-8-sig")
    submission.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    validation_errors = validate_submission(
        submission_path,
        ROOT / "data" / "raw" / "sample_submission.csv",
    )
    if validation_errors:
        raise ValueError(
            "Generated submission failed validation:\n"
            + "\n".join(f"- {error}" for error in validation_errors)
        )

    summary = {
        "exp_id": EXP_ID,
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "data": {
            "train_labels": "data/raw/train/train_labels.csv",
            "ldaps_train": "data/raw/train/ldaps_train.csv",
            "gfs_train": "data/raw/train/gfs_train.csv",
            "ldaps_test": "data/raw/test/ldaps_test.csv",
            "gfs_test": "data/raw/test/gfs_test.csv",
            "sample_submission": "data/raw/sample_submission.csv",
        },
        "training": {
            "rows": int(len(train_frame)),
            "target_train_rows": result.train_rows,
            "scope": "all available train labels",
        },
        "inference": {
            "rows": int(len(submission)),
            "forecast_start": str(test_frame["forecast_kst_dtm"].min()),
            "forecast_end": str(test_frame["forecast_kst_dtm"].max()),
        },
        "features": {
            "approach": "Baseline calendar features plus mean-aggregated LDAPS/GFS by forecast_kst_dtm.",
            "feature_count": len(result.feature_columns),
            "feature_columns": result.feature_columns,
        },
        "model": {
            "class": "lightgbm.LGBMRegressor",
            "per_target_models": list(TARGET_COLS),
            "params": lgbm_v1_params(random_state=RANDOM_SEED),
        },
        "postprocessing": {
            "clip_min": 0.0,
            "clip_max": CAPACITY_KWH,
            "submission_columns": sample_columns,
        },
        "artifacts": {
            "submission": str(submission_path.relative_to(ROOT)),
            "test_predictions": str(prediction_path.relative_to(ROOT)),
            "run_summary": str(log_path.relative_to(ROOT)),
            "model": str(model_path.relative_to(ROOT)),
        },
        "validation": {
            "script": "scripts/validate_submission.py",
            "passed": True,
        },
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
            "model_params": lgbm_v1_params(random_state=RANDOM_SEED),
            "capacity_kwh": CAPACITY_KWH,
        },
        model_path,
    )

    print(f"Saved submission: {submission_path.relative_to(ROOT)}")
    print(f"Saved test predictions: {prediction_path.relative_to(ROOT)}")
    print(f"Saved run summary: {log_path.relative_to(ROOT)}")
    print(f"Saved model artifact: {model_path.relative_to(ROOT)}")
    print("Submission validation passed.")


if __name__ == "__main__":
    main()

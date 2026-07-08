from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dacon_wind.cv import split_train_valid_2024
from dacon_wind.data import load_train_data
from dacon_wind.features import build_baseline_train_frame
from dacon_wind.metric import CAPACITY_KWH, TARGET_COLS, calculate_metric


EXP_ID = "ens_001_simple_avg"
RUN_ID = "ens_001_simple_avg_valid_2024"
BASELINE_LOCAL_TOTAL_SCORE = 0.6033279875

PREDICTION_FILES = {
    "lgbm_003_tuned_valid_2024": ROOT
    / "outputs"
    / "predictions"
    / "lgbm_003_tuned_valid_2024.csv",
    "lgbm_005_targeted_weather_valid_2024": ROOT
    / "outputs"
    / "predictions"
    / "lgbm_005_targeted_weather_valid_2024.csv",
    "cat_001_baseline_valid_2024": ROOT
    / "outputs"
    / "predictions"
    / "cat_001_baseline_valid_2024.csv",
    "xgb_001_baseline_valid_2024": ROOT
    / "outputs"
    / "predictions"
    / "xgb_001_baseline_valid_2024.csv",
}

ENSEMBLE_CANDIDATES = (
    {
        "name": "ens_001_lgbm003_lgbm005_avg",
        "weights": {
            "lgbm_003_tuned_valid_2024": 0.5,
            "lgbm_005_targeted_weather_valid_2024": 0.5,
        },
    },
    {
        "name": "ens_001_lgbm003_lgbm005_xgb_avg",
        "weights": {
            "lgbm_003_tuned_valid_2024": 1.0 / 3.0,
            "lgbm_005_targeted_weather_valid_2024": 1.0 / 3.0,
            "xgb_001_baseline_valid_2024": 1.0 / 3.0,
        },
    },
    {
        "name": "ens_001_lgbm003_lgbm005_cat_xgb_avg",
        "weights": {
            "lgbm_003_tuned_valid_2024": 0.25,
            "lgbm_005_targeted_weather_valid_2024": 0.25,
            "cat_001_baseline_valid_2024": 0.25,
            "xgb_001_baseline_valid_2024": 0.25,
        },
    },
    {
        "name": "ens_001_weighted_lgbm003_lgbm005",
        "weights": {
            "lgbm_003_tuned_valid_2024": 0.7,
            "lgbm_005_targeted_weather_valid_2024": 0.3,
        },
    },
    {
        "name": "ens_001_weighted_lgbm003_lgbm005_xgb",
        "weights": {
            "lgbm_003_tuned_valid_2024": 0.6,
            "lgbm_005_targeted_weather_valid_2024": 0.25,
            "xgb_001_baseline_valid_2024": 0.15,
        },
    },
)


def ensure_new_artifacts(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = "\n".join(f"- {path.relative_to(ROOT)}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing artifact(s):\n"
            f"{formatted}\n"
            "Remove or rename them before rerunning this script."
        )


def load_valid_truth() -> pd.DataFrame:
    data = load_train_data(ROOT / "data" / "raw" / "train")
    train_frame = build_baseline_train_frame(data.labels, data.ldaps, data.gfs)
    _, valid_mask = split_train_valid_2024(train_frame["forecast_kst_dtm"])

    truth = train_frame.loc[valid_mask, ["forecast_kst_dtm", *TARGET_COLS]].copy()
    truth["forecast_kst_dtm"] = pd.to_datetime(truth["forecast_kst_dtm"])
    truth = truth.sort_values("forecast_kst_dtm").reset_index(drop=True)

    if truth["forecast_kst_dtm"].duplicated().any():
        raise ValueError("Validation truth contains duplicate forecast_kst_dtm values.")

    return truth


def load_prediction(path: Path, expected_times: pd.Series) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path.relative_to(ROOT)}")

    prediction = pd.read_csv(path, encoding="utf-8-sig")
    required_cols = ["forecast_kst_dtm", *TARGET_COLS]
    missing = [col for col in required_cols if col not in prediction.columns]
    if missing:
        raise ValueError(f"{path.relative_to(ROOT)} is missing columns: {missing}")

    prediction = prediction.loc[:, required_cols].copy()
    prediction["forecast_kst_dtm"] = pd.to_datetime(prediction["forecast_kst_dtm"])
    prediction = prediction.sort_values("forecast_kst_dtm").reset_index(drop=True)

    if prediction["forecast_kst_dtm"].duplicated().any():
        raise ValueError(f"{path.relative_to(ROOT)} contains duplicate forecast_kst_dtm values.")

    if len(prediction) != len(expected_times):
        raise ValueError(
            f"{path.relative_to(ROOT)} row count {len(prediction)} does not match "
            f"validation truth row count {len(expected_times)}."
        )

    if not prediction["forecast_kst_dtm"].equals(expected_times.reset_index(drop=True)):
        raise ValueError(f"{path.relative_to(ROOT)} forecast_kst_dtm alignment mismatch.")

    return prediction


def clip_predictions(prediction: pd.DataFrame) -> pd.DataFrame:
    clipped = prediction.copy()
    for target in TARGET_COLS:
        clipped[target] = clipped[target].clip(lower=0.0, upper=CAPACITY_KWH[target])
    return clipped


def build_ensemble(
    predictions: dict[str, pd.DataFrame],
    weights: dict[str, float],
    forecast_times: pd.Series,
) -> pd.DataFrame:
    weight_sum = float(sum(weights.values()))
    if not np.isclose(weight_sum, 1.0):
        raise ValueError(f"Ensemble weights must sum to 1.0, got {weight_sum}.")

    ensemble = pd.DataFrame({"forecast_kst_dtm": forecast_times.reset_index(drop=True)})
    for target in TARGET_COLS:
        values = np.zeros(len(forecast_times), dtype=float)
        for model_name, weight in weights.items():
            values += predictions[model_name][target].to_numpy(dtype=float) * weight
        ensemble[target] = values

    return clip_predictions(ensemble)


def decision_for_score(total_score: float) -> str:
    if total_score > BASELINE_LOCAL_TOTAL_SCORE:
        return "submission candidate"
    if 0.6025 <= total_score <= BASELINE_LOCAL_TOTAL_SCORE:
        return "possible submission candidate"
    return "validation-only ensemble reference"


def main() -> None:
    prediction_dir = ROOT / "outputs" / "predictions"
    log_dir = ROOT / "outputs" / "logs"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_prediction_path = prediction_dir / f"{RUN_ID}_best.csv"
    log_path = log_dir / f"{RUN_ID}.json"
    ensure_new_artifacts([best_prediction_path, log_path])

    truth = load_valid_truth()
    forecast_times = truth["forecast_kst_dtm"]
    y_true = truth.loc[:, list(TARGET_COLS)].reset_index(drop=True)

    predictions = {
        name: load_prediction(path, forecast_times)
        for name, path in PREDICTION_FILES.items()
    }

    candidate_results: list[dict[str, Any]] = []
    candidate_predictions: dict[str, pd.DataFrame] = {}

    for candidate in ENSEMBLE_CANDIDATES:
        name = candidate["name"]
        weights = candidate["weights"]
        ensemble = build_ensemble(predictions, weights, forecast_times)
        scores = calculate_metric(
            y_true,
            ensemble.loc[:, list(TARGET_COLS)].reset_index(drop=True),
        )

        result = {
            "candidate_name": name,
            "member_models": list(weights.keys()),
            "weights": weights,
            "total_score": scores["total_score"],
            "one_minus_nmae": scores["one_minus_nmae"],
            "ficr": scores["ficr"],
            "group_nmae": scores["group_nmae"],
            "group_ficr": scores["group_ficr"],
            "beats_lgbm_003_tuned": scores["total_score"] > BASELINE_LOCAL_TOTAL_SCORE,
            "decision": decision_for_score(scores["total_score"]),
        }
        candidate_results.append(result)
        candidate_predictions[name] = ensemble

    best_result = max(candidate_results, key=lambda item: item["total_score"])
    best_prediction = candidate_predictions[best_result["candidate_name"]].copy()
    best_prediction["forecast_kst_dtm"] = pd.to_datetime(
        best_prediction["forecast_kst_dtm"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    best_prediction.to_csv(best_prediction_path, index=False, encoding="utf-8-sig")

    summary = {
        "exp_id": EXP_ID,
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Validation-only test of simple averages over existing 2024 "
            "validation predictions. No models are trained and no submission "
            "or test prediction file is created."
        ),
        "baseline_comparison": {
            "baseline_exp_id": "lgbm_003_tuned",
            "baseline_local_total_score": BASELINE_LOCAL_TOTAL_SCORE,
        },
        "data": {
            "train_labels": "data/raw/train/train_labels.csv",
            "ldaps_train": "data/raw/train/ldaps_train.csv",
            "gfs_train": "data/raw/train/gfs_train.csv",
        },
        "split": {
            "valid": "year == 2024",
            "valid_rows": int(len(truth)),
            "forecast_start": str(forecast_times.min()),
            "forecast_end": str(forecast_times.max()),
        },
        "input_prediction_files": {
            name: str(path.relative_to(ROOT)) for name, path in PREDICTION_FILES.items()
        },
        "alignment": {
            "key": "forecast_kst_dtm",
            "status": "all candidate predictions exactly match sorted 2024 validation timestamps",
        },
        "postprocessing": {
            "prediction_clipping": "Per target group, clipped to [0, group_capacity] before scoring.",
        },
        "candidates": candidate_results,
        "selected_best_candidate": best_result,
        "artifacts": {
            "best_valid_predictions": str(best_prediction_path.relative_to(ROOT)),
            "run_summary": str(log_path.relative_to(ROOT)),
        },
    }

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("candidate,total_score,one_minus_nmae,ficr,beats_lgbm_003_tuned,decision")
    for result in candidate_results:
        print(
            f"{result['candidate_name']},"
            f"{result['total_score']:.10f},"
            f"{result['one_minus_nmae']:.10f},"
            f"{result['ficr']:.10f},"
            f"{result['beats_lgbm_003_tuned']},"
            f"{result['decision']}"
        )
    print(f"selected_best_candidate: {best_result['candidate_name']}")
    print(f"best_total_score: {best_result['total_score']:.10f}")
    print(f"best_decision: {best_result['decision']}")
    print(f"Saved best validation predictions: {best_prediction_path.relative_to(ROOT)}")
    print(f"Saved run summary: {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

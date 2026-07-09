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


EXP_ID = "ens_002_weight_search_cv"
RUN_ID = "ens_002_weight_search_cv_valid_2024"
ENS_001_LOCAL_TOTAL_SCORE = 0.6037465036
LGBM_003_LOCAL_TOTAL_SCORE = 0.6033279875

LGBM_003_NAME = "lgbm_003_tuned_valid_2024"
LGBM_005_NAME = "lgbm_005_targeted_weather_valid_2024"

PREDICTION_FILES = {
    LGBM_003_NAME: ROOT
    / "outputs"
    / "predictions"
    / "lgbm_003_tuned_valid_2024.csv",
    LGBM_005_NAME: ROOT
    / "outputs"
    / "predictions"
    / "lgbm_005_targeted_weather_valid_2024.csv",
}


def ensure_new_artifacts(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = "\n".join(f"- {path.relative_to(ROOT)}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing artifact(s):\n"
            f"{formatted}\n"
            "Remove or rename them before rerunning this script."
        )


def build_weight_grid() -> list[float]:
    coarse = [round(value / 10.0, 2) for value in range(0, 11)]
    fine = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    return sorted(set(coarse + fine))


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


def build_weighted_ensemble(
    lgbm_003: pd.DataFrame,
    lgbm_005: pd.DataFrame,
    forecast_times: pd.Series,
    lgbm_003_weight: float,
) -> pd.DataFrame:
    lgbm_005_weight = 1.0 - lgbm_003_weight
    ensemble = pd.DataFrame({"forecast_kst_dtm": forecast_times.reset_index(drop=True)})
    for target in TARGET_COLS:
        ensemble[target] = (
            lgbm_003[target].to_numpy(dtype=float) * lgbm_003_weight
            + lgbm_005[target].to_numpy(dtype=float) * lgbm_005_weight
        )
    return clip_predictions(ensemble)


def decision_for_score(total_score: float) -> str:
    if total_score > ENS_001_LOCAL_TOTAL_SCORE:
        return "improved submission candidate"
    if np.isclose(total_score, ENS_001_LOCAL_TOTAL_SCORE, rtol=0.0, atol=0.0001):
        return "keep ens_001_simple_avg_submit as current best"
    return "validation-only reference"


def result_for_candidate(
    weight: float,
    scores: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_name": f"ens_002_lgbm003_{weight:.2f}_lgbm005_{1.0 - weight:.2f}",
        "weights": {
            "lgbm_003_tuned": float(weight),
            "lgbm_005_targeted_weather": float(1.0 - weight),
        },
        "total_score": scores["total_score"],
        "one_minus_nmae": scores["one_minus_nmae"],
        "ficr": scores["ficr"],
        "group_nmae": scores["group_nmae"],
        "group_ficr": scores["group_ficr"],
        "beats_ens_001_validation": scores["total_score"] > ENS_001_LOCAL_TOTAL_SCORE,
        "beats_lgbm_003_tuned": scores["total_score"] > LGBM_003_LOCAL_TOTAL_SCORE,
        "decision": decision_for_score(scores["total_score"]),
    }


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

    for weight in build_weight_grid():
        ensemble = build_weighted_ensemble(
            predictions[LGBM_003_NAME],
            predictions[LGBM_005_NAME],
            forecast_times,
            weight,
        )
        scores = calculate_metric(
            y_true,
            ensemble.loc[:, list(TARGET_COLS)].reset_index(drop=True),
        )
        result = result_for_candidate(weight, scores)
        candidate_results.append(result)
        candidate_predictions[result["candidate_name"]] = ensemble

    candidate_results = sorted(
        candidate_results,
        key=lambda item: item["total_score"],
        reverse=True,
    )
    best_result = candidate_results[0]

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
            "Validation-only weighted ensemble search between lgbm_003_tuned "
            "and lgbm_005_targeted_weather 2024 validation predictions. No "
            "models are trained, no test predictions are created, and no "
            "submission CSV is created."
        ),
        "comparison_references": {
            "ens_001_simple_avg_validation_total_score": ENS_001_LOCAL_TOTAL_SCORE,
            "lgbm_003_tuned_validation_total_score": LGBM_003_LOCAL_TOTAL_SCORE,
        },
        "data": {
            "train_labels": "data/raw/train/train_labels.csv",
            "ldaps_train": "data/raw/train/ldaps_train.csv",
            "gfs_train": "data/raw/train/gfs_train.csv",
        },
        "split": {
            "valid": "year == 2024 via split_train_valid_2024",
            "valid_rows": int(len(truth)),
            "forecast_start": str(forecast_times.min()),
            "forecast_end": str(forecast_times.max()),
        },
        "input_prediction_files": {
            name: str(path.relative_to(ROOT)) for name, path in PREDICTION_FILES.items()
        },
        "alignment": {
            "key": "forecast_kst_dtm",
            "status": "both member predictions exactly match sorted 2024 validation timestamps",
        },
        "weight_grid": {
            "coarse_lgbm_003_weights": [round(value / 10.0, 2) for value in range(0, 11)],
            "fine_lgbm_003_weights": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
            "evaluated_lgbm_003_weights": build_weight_grid(),
            "duplicate_policy": "coarse and fine weights are de-duplicated before scoring",
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

    print("candidate,total_score,one_minus_nmae,ficr,beats_ens_001_validation,beats_lgbm_003_tuned,decision")
    for result in candidate_results:
        print(
            f"{result['candidate_name']},"
            f"{result['total_score']:.10f},"
            f"{result['one_minus_nmae']:.10f},"
            f"{result['ficr']:.10f},"
            f"{result['beats_ens_001_validation']},"
            f"{result['beats_lgbm_003_tuned']},"
            f"{result['decision']}"
        )
    print(f"selected_best_candidate: {best_result['candidate_name']}")
    print(f"best_total_score: {best_result['total_score']:.10f}")
    print(f"best_one_minus_nmae: {best_result['one_minus_nmae']:.10f}")
    print(f"best_ficr: {best_result['ficr']:.10f}")
    print(f"best_decision: {best_result['decision']}")
    print(f"Saved best validation predictions: {best_prediction_path.relative_to(ROOT)}")
    print(f"Saved run summary: {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

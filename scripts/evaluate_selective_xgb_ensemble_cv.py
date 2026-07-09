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


EXP_ID = "ens_003_include_xgb_selective"
RUN_ID = "ens_003_include_xgb_selective_valid_2024"
ENS_001_LOCAL_TOTAL_SCORE = 0.6037465036
LGBM_003_LOCAL_TOTAL_SCORE = 0.6033279875

LGBM_003_NAME = "lgbm_003_tuned"
LGBM_005_NAME = "lgbm_005_targeted_weather"
XGB_001_NAME = "xgb_001_baseline"

PREDICTION_FILES = {
    LGBM_003_NAME: ROOT
    / "outputs"
    / "predictions"
    / "lgbm_003_tuned_valid_2024.csv",
    LGBM_005_NAME: ROOT
    / "outputs"
    / "predictions"
    / "lgbm_005_targeted_weather_valid_2024.csv",
    XGB_001_NAME: ROOT
    / "outputs"
    / "predictions"
    / "xgb_001_baseline_valid_2024.csv",
}

ENSEMBLE_CANDIDATES = (
    {
        "name": "ens_003_reference_lgbm003_lgbm005_50_50",
        "weights": {
            LGBM_003_NAME: 0.50,
            LGBM_005_NAME: 0.50,
            XGB_001_NAME: 0.00,
        },
    },
    {
        "name": "ens_003_xgb005_a",
        "weights": {
            LGBM_003_NAME: 0.50,
            LGBM_005_NAME: 0.45,
            XGB_001_NAME: 0.05,
        },
    },
    {
        "name": "ens_003_xgb005_b",
        "weights": {
            LGBM_003_NAME: 0.55,
            LGBM_005_NAME: 0.40,
            XGB_001_NAME: 0.05,
        },
    },
    {
        "name": "ens_003_xgb005_c",
        "weights": {
            LGBM_003_NAME: 0.45,
            LGBM_005_NAME: 0.50,
            XGB_001_NAME: 0.05,
        },
    },
    {
        "name": "ens_003_xgb010_a",
        "weights": {
            LGBM_003_NAME: 0.50,
            LGBM_005_NAME: 0.40,
            XGB_001_NAME: 0.10,
        },
    },
    {
        "name": "ens_003_xgb010_b",
        "weights": {
            LGBM_003_NAME: 0.45,
            LGBM_005_NAME: 0.45,
            XGB_001_NAME: 0.10,
        },
    },
    {
        "name": "ens_003_xgb010_c",
        "weights": {
            LGBM_003_NAME: 0.55,
            LGBM_005_NAME: 0.35,
            XGB_001_NAME: 0.10,
        },
    },
    {
        "name": "ens_003_xgb015_a",
        "weights": {
            LGBM_003_NAME: 0.45,
            LGBM_005_NAME: 0.40,
            XGB_001_NAME: 0.15,
        },
    },
    {
        "name": "ens_003_xgb015_b",
        "weights": {
            LGBM_003_NAME: 0.50,
            LGBM_005_NAME: 0.35,
            XGB_001_NAME: 0.15,
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


def candidate_uses_xgb(weights: dict[str, float]) -> bool:
    return weights.get(XGB_001_NAME, 0.0) > 0.0


def decision_for_best(best_result: dict[str, Any], reference_result: dict[str, Any]) -> str:
    if (
        best_result["total_score"] > ENS_001_LOCAL_TOTAL_SCORE
        and candidate_uses_xgb(best_result["weights"])
    ):
        return "improved submission candidate"
    if best_result["candidate_name"] == reference_result["candidate_name"]:
        return "keep ens_001_simple_avg_submit as current best"
    if (
        candidate_uses_xgb(best_result["weights"])
        and best_result["ficr"] > reference_result["ficr"]
        and best_result["total_score"] < reference_result["total_score"]
    ):
        return "FICR-diversity reference"
    return "validation-only reference"


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
            "weights": weights,
            "total_score": scores["total_score"],
            "one_minus_nmae": scores["one_minus_nmae"],
            "ficr": scores["ficr"],
            "group_nmae": scores["group_nmae"],
            "group_ficr": scores["group_ficr"],
            "beats_ens_001_validation": scores["total_score"] > ENS_001_LOCAL_TOTAL_SCORE,
            "beats_lgbm_003_tuned": scores["total_score"] > LGBM_003_LOCAL_TOTAL_SCORE,
            "uses_xgb": candidate_uses_xgb(weights),
        }
        candidate_results.append(result)
        candidate_predictions[name] = ensemble

    candidate_results = sorted(
        candidate_results,
        key=lambda item: item["total_score"],
        reverse=True,
    )
    best_result = candidate_results[0]
    reference_result = next(
        result
        for result in candidate_results
        if result["candidate_name"] == "ens_003_reference_lgbm003_lgbm005_50_50"
    )
    final_decision = decision_for_best(best_result, reference_result)
    xgb_candidates_beat_reference = any(
        result["uses_xgb"] and result["total_score"] > reference_result["total_score"]
        for result in candidate_results
    )

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
            "Validation-only selective XGB ensemble test. Existing 2024 validation "
            "predictions are averaged with fixed small XGB weights. No models are "
            "trained, no test predictions are created, and no submission CSV is created."
        ),
        "comparison_references": {
            "current_best_public_submission": "ens_001_simple_avg_submit",
            "ens_001_validation_total_score": ENS_001_LOCAL_TOTAL_SCORE,
            "lgbm_003_tuned_validation_total_score": LGBM_003_LOCAL_TOTAL_SCORE,
            "xgb_001_baseline_validation_total_score": 0.5988239498,
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
            "status": "all member predictions exactly match sorted 2024 validation timestamps",
        },
        "postprocessing": {
            "prediction_clipping": "Per target group, clipped to [0, group_capacity] before scoring.",
        },
        "candidates": candidate_results,
        "reference_candidate": reference_result,
        "xgb_candidates_beat_reference": xgb_candidates_beat_reference,
        "selected_best_candidate": best_result,
        "decision": final_decision,
        "artifacts": {
            "best_valid_predictions": str(best_prediction_path.relative_to(ROOT)),
            "run_summary": str(log_path.relative_to(ROOT)),
        },
    }

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        "candidate,total_score,one_minus_nmae,ficr,xgb_weight,"
        "beats_ens_001_validation,beats_lgbm_003_tuned"
    )
    for result in candidate_results:
        print(
            f"{result['candidate_name']},"
            f"{result['total_score']:.10f},"
            f"{result['one_minus_nmae']:.10f},"
            f"{result['ficr']:.10f},"
            f"{result['weights'][XGB_001_NAME]:.2f},"
            f"{result['beats_ens_001_validation']},"
            f"{result['beats_lgbm_003_tuned']}"
        )
    print(f"selected_best_candidate: {best_result['candidate_name']}")
    print(f"best_total_score: {best_result['total_score']:.10f}")
    print(f"best_one_minus_nmae: {best_result['one_minus_nmae']:.10f}")
    print(f"best_ficr: {best_result['ficr']:.10f}")
    print(f"xgb_candidates_beat_reference: {xgb_candidates_beat_reference}")
    print(f"decision: {final_decision}")
    print(f"Saved best validation predictions: {best_prediction_path.relative_to(ROOT)}")
    print(f"Saved run summary: {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

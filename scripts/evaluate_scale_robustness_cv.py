from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dacon_wind.cv import split_train_valid_2024
from dacon_wind.data import load_train_data
from dacon_wind.features import build_baseline_train_frame
from dacon_wind.metric import CAPACITY_KWH, TARGET_COLS, calculate_metric


EXP_ID = "lgbm_007_scale_robustness_cv"
RUN_ID = "lgbm_007_scale_robustness_cv_valid_2024"
INPUT_PREDICTION_PATH = (
    ROOT / "outputs" / "predictions" / "ens_001_simple_avg_valid_2024_best.csv"
)

LGBM_006_SCALE_103_TOTAL_SCORE = 0.6109322217
LGBM_006_SCALE_103_FICR = 0.3525742158

SCALES = sorted(
    {
        1.00,
        1.01,
        1.02,
        1.03,
        1.04,
        1.05,
        1.06,
        1.07,
        1.08,
        1.025,
        1.030,
        1.035,
        1.040,
    }
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

    expected_times = expected_times.reset_index(drop=True)
    if not prediction["forecast_kst_dtm"].equals(expected_times):
        raise ValueError(f"{path.relative_to(ROOT)} forecast_kst_dtm alignment mismatch.")

    return prediction


def scaled_prediction(base_prediction: pd.DataFrame, scale: float) -> pd.DataFrame:
    prediction = base_prediction.copy()
    for target in TARGET_COLS:
        prediction[target] = (prediction[target] * scale).clip(
            lower=0.0,
            upper=CAPACITY_KWH[target],
        )
    return prediction


def score_scale(
    scale: float,
    base_prediction: pd.DataFrame,
    y_true: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    prediction = scaled_prediction(base_prediction, scale)
    scores = calculate_metric(
        y_true,
        prediction.loc[:, list(TARGET_COLS)].reset_index(drop=True),
    )

    result = {
        "scale": scale,
        "candidate_name": f"{EXP_ID}_scale_{scale:.3f}".replace(".", "p"),
        "total_score": scores["total_score"],
        "one_minus_nmae": scores["one_minus_nmae"],
        "ficr": scores["ficr"],
        "group_nmae": scores["group_nmae"],
        "group_ficr": scores["group_ficr"],
        "beats_lgbm_006_scale_103_total_score": (
            scores["total_score"] > LGBM_006_SCALE_103_TOTAL_SCORE
        ),
        "beats_lgbm_006_scale_103_ficr": scores["ficr"] > LGBM_006_SCALE_103_FICR,
    }
    return result, prediction


def is_near_best(candidate: dict[str, Any], best: dict[str, Any]) -> bool:
    return best["total_score"] - candidate["total_score"] <= 0.0002


def final_decision(
    best_total_score: dict[str, Any],
    best_ficr: dict[str, Any],
    scale_103: dict[str, Any],
) -> str:
    if (
        best_total_score["scale"] != 1.03
        and best_total_score["beats_lgbm_006_scale_103_total_score"]
        and best_total_score["beats_lgbm_006_scale_103_ficr"]
    ):
        return "possible submission candidate"
    if scale_103["scale"] == best_total_score["scale"] or is_near_best(scale_103, best_total_score):
        return "keep lgbm_006_ficr_focus_submit as current best"
    if (
        best_ficr["ficr"] > LGBM_006_SCALE_103_FICR
        and best_total_score["total_score"] <= LGBM_006_SCALE_103_TOTAL_SCORE
    ):
        return "FICR-risk reference"
    return "keep lgbm_006_ficr_focus_submit as current best"


def write_prediction(path: Path, prediction: pd.DataFrame) -> None:
    output = prediction.copy()
    output["forecast_kst_dtm"] = pd.to_datetime(output["forecast_kst_dtm"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    output.to_csv(path, index=False, encoding="utf-8-sig")


def print_results(
    candidates_by_total: list[dict[str, Any]],
    candidates_by_ficr: list[dict[str, Any]],
    best_total_score: dict[str, Any],
    best_ficr: dict[str, Any],
    any_clear_beat_103: bool,
    decision: str,
) -> None:
    print("candidates_sorted_by_total_score")
    print("scale,total_score,one_minus_nmae,ficr,beats_103_total,beats_103_ficr")
    for result in candidates_by_total:
        print(
            f"{result['scale']:.3f},"
            f"{result['total_score']:.10f},"
            f"{result['one_minus_nmae']:.10f},"
            f"{result['ficr']:.10f},"
            f"{result['beats_lgbm_006_scale_103_total_score']},"
            f"{result['beats_lgbm_006_scale_103_ficr']}"
        )

    print("candidates_sorted_by_ficr")
    print("scale,total_score,one_minus_nmae,ficr,beats_103_total,beats_103_ficr")
    for result in candidates_by_ficr:
        print(
            f"{result['scale']:.3f},"
            f"{result['total_score']:.10f},"
            f"{result['one_minus_nmae']:.10f},"
            f"{result['ficr']:.10f},"
            f"{result['beats_lgbm_006_scale_103_total_score']},"
            f"{result['beats_lgbm_006_scale_103_ficr']}"
        )

    print(f"selected_best_by_total_score_scale: {best_total_score['scale']:.3f}")
    print(f"selected_best_by_total_score: {best_total_score['total_score']:.10f}")
    print(f"selected_best_by_ficr_scale: {best_ficr['scale']:.3f}")
    print(f"selected_best_by_ficr: {best_ficr['ficr']:.10f}")
    print(f"any_scale_clearly_beat_1.03: {any_clear_beat_103}")
    print(f"final_decision: {decision}")


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
    base_prediction = load_prediction(INPUT_PREDICTION_PATH, forecast_times)

    predictions_by_scale: dict[float, pd.DataFrame] = {}
    candidate_results: list[dict[str, Any]] = []
    for scale in SCALES:
        result, prediction = score_scale(scale, base_prediction, y_true)
        candidate_results.append(result)
        predictions_by_scale[scale] = prediction

    candidates_by_total = sorted(
        candidate_results,
        key=lambda item: item["total_score"],
        reverse=True,
    )
    candidates_by_ficr = sorted(
        candidate_results,
        key=lambda item: item["ficr"],
        reverse=True,
    )
    best_total_score = candidates_by_total[0]
    best_ficr = candidates_by_ficr[0]
    scale_103 = next(item for item in candidate_results if item["scale"] == 1.03)
    any_clear_beat_103 = any(
        item["scale"] != 1.03
        and item["beats_lgbm_006_scale_103_total_score"]
        and item["beats_lgbm_006_scale_103_ficr"]
        for item in candidate_results
    )
    decision = final_decision(best_total_score, best_ficr, scale_103)

    write_prediction(
        best_prediction_path,
        predictions_by_scale[best_total_score["scale"]],
    )

    summary = {
        "exp_id": EXP_ID,
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Check whether the global scaling effect found in "
            "lgbm_006_ficr_focus is robust around scale 1.03."
        ),
        "current_best_public_reference": {
            "submission": "lgbm_006_ficr_focus_submit",
            "public_total_score": 0.6158048399,
            "public_one_minus_nmae": 0.8679909923,
            "public_ficr": 0.3636186875,
            "public_rank_at_submission_time": 245,
        },
        "lgbm_006_validation_reference": {
            "selected_candidate": "lgbm_006_global_scale_103",
            "scale": 1.03,
            "total_score": LGBM_006_SCALE_103_TOTAL_SCORE,
            "one_minus_nmae": 0.8692902276,
            "ficr": LGBM_006_SCALE_103_FICR,
        },
        "source_validation_prediction": str(INPUT_PREDICTION_PATH.relative_to(ROOT)),
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
        "alignment": {
            "key": "forecast_kst_dtm",
            "status": "input validation prediction exactly matches sorted 2024 validation timestamps",
        },
        "scoring": {
            "metric": "src/dacon_wind/metric.py calculate_metric",
            "prediction_clipping": "Per target group, clipped to [0, group_capacity] before scoring.",
            "scales": SCALES,
        },
        "candidates": candidates_by_total,
        "selected_best_by_total_score": best_total_score,
        "selected_best_by_ficr": best_ficr,
        "any_scale_clearly_beat_1_03": any_clear_beat_103,
        "final_decision": decision,
        "artifacts": {
            "best_valid_predictions": str(best_prediction_path.relative_to(ROOT)),
            "run_summary": str(log_path.relative_to(ROOT)),
        },
    }

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print_results(
        candidates_by_total,
        candidates_by_ficr,
        best_total_score,
        best_ficr,
        any_clear_beat_103,
        decision,
    )
    print(f"Saved best validation predictions: {best_prediction_path.relative_to(ROOT)}")
    print(f"Saved run summary: {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

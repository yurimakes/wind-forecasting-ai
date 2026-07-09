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


EXP_ID = "lgbm_006_ficr_focus"
RUN_ID = "lgbm_006_ficr_focus_valid_2024"
ENS_001_LOCAL_TOTAL_SCORE = 0.6037465036
ENS_001_LOCAL_FICR = 0.3394377775
INPUT_PREDICTION_PATH = (
    ROOT / "outputs" / "predictions" / "ens_001_simple_avg_valid_2024_best.csv"
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


def make_candidate(
    name: str,
    family: str,
    parameters: dict[str, Any],
    prediction: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "parameters": parameters,
        "prediction": clip_predictions(prediction),
    }


def global_scale_candidate(base: pd.DataFrame, scale: float) -> pd.DataFrame:
    adjusted = base.copy()
    for target in TARGET_COLS:
        adjusted[target] = adjusted[target] * scale
    return adjusted


def group_scale_candidate(base: pd.DataFrame, scales: dict[str, float]) -> pd.DataFrame:
    adjusted = base.copy()
    for target, scale in scales.items():
        adjusted[target] = adjusted[target] * scale
    return adjusted


def ramp_candidate(base: pd.DataFrame, alpha: float) -> pd.DataFrame:
    adjusted = base.copy()
    for target in TARGET_COLS:
        rolling_mean = adjusted[target].rolling(window=24, min_periods=1).mean()
        adjusted[target] = rolling_mean + alpha * (adjusted[target] - rolling_mean)
    return adjusted


def high_shrink_candidate(base: pd.DataFrame, beta: float) -> pd.DataFrame:
    adjusted = base.copy()
    for target in TARGET_COLS:
        threshold = float(adjusted[target].quantile(0.90))
        high_mask = adjusted[target] > threshold
        adjusted.loc[high_mask, target] = (
            threshold + beta * (adjusted.loc[high_mask, target] - threshold)
        )
    return adjusted


def build_candidates(base: pd.DataFrame) -> list[dict[str, Any]]:
    candidates = [
        make_candidate(
            "lgbm_006_reference_ens001_no_change",
            "reference",
            {"source": "ens_001_simple_avg_valid_2024_best"},
            base.copy(),
        )
    ]

    for scale in [0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03]:
        candidates.append(
            make_candidate(
                f"lgbm_006_global_scale_{int(round(scale * 100)):03d}",
                "global_scale",
                {"scale": scale},
                global_scale_candidate(base, scale),
            )
        )

    group_scale_sets = [
        {"kpx_group_1": 1.01, "kpx_group_2": 1.00, "kpx_group_3": 1.00},
        {"kpx_group_1": 1.00, "kpx_group_2": 1.01, "kpx_group_3": 1.00},
        {"kpx_group_1": 1.00, "kpx_group_2": 1.00, "kpx_group_3": 1.01},
        {"kpx_group_1": 0.99, "kpx_group_2": 1.00, "kpx_group_3": 1.00},
        {"kpx_group_1": 1.00, "kpx_group_2": 0.99, "kpx_group_3": 1.00},
        {"kpx_group_1": 1.00, "kpx_group_2": 1.00, "kpx_group_3": 0.99},
        {"kpx_group_1": 1.01, "kpx_group_2": 1.01, "kpx_group_3": 1.00},
        {"kpx_group_1": 1.01, "kpx_group_2": 1.00, "kpx_group_3": 1.01},
        {"kpx_group_1": 1.00, "kpx_group_2": 1.01, "kpx_group_3": 1.01},
    ]
    for idx, scales in enumerate(group_scale_sets, start=1):
        suffix = "_".join(
            f"g{group_idx}_{int(round(scales[target] * 100)):03d}"
            for group_idx, target in enumerate(TARGET_COLS, start=1)
        )
        candidates.append(
            make_candidate(
                f"lgbm_006_group_scale_{idx:02d}_{suffix}",
                "group_specific_scale",
                {"scales": scales},
                group_scale_candidate(base, scales),
            )
        )

    for alpha in [0.90, 0.95, 1.00, 1.05, 1.10]:
        candidates.append(
            make_candidate(
                f"lgbm_006_ramp_alpha_{int(round(alpha * 100)):03d}",
                "ramp_amplification",
                {"alpha": alpha, "rolling_window_hours": 24, "min_periods": 1},
                ramp_candidate(base, alpha),
            )
        )

    for beta in [0.90, 0.95]:
        candidates.append(
            make_candidate(
                f"lgbm_006_high_shrink_{int(round(beta * 100)):03d}",
                "high_end_shrink",
                {"beta": beta, "quantile": 0.90},
                high_shrink_candidate(base, beta),
            )
        )

    return candidates


def score_candidate(
    candidate: dict[str, Any],
    y_true: pd.DataFrame,
) -> dict[str, Any]:
    prediction = candidate["prediction"]
    scores = calculate_metric(
        y_true,
        prediction.loc[:, list(TARGET_COLS)].reset_index(drop=True),
    )
    return {
        "candidate_name": candidate["name"],
        "postprocessing_family": candidate["family"],
        "parameters": candidate["parameters"],
        "total_score": scores["total_score"],
        "one_minus_nmae": scores["one_minus_nmae"],
        "ficr": scores["ficr"],
        "group_nmae": scores["group_nmae"],
        "group_ficr": scores["group_ficr"],
        "beats_ens_001_validation_total_score": (
            scores["total_score"] > ENS_001_LOCAL_TOTAL_SCORE
        ),
        "beats_ens_001_validation_ficr": scores["ficr"] > ENS_001_LOCAL_FICR,
    }


def decision_for_results(
    best_total_score: dict[str, Any],
    best_ficr: dict[str, Any],
) -> str:
    if best_total_score["total_score"] > ENS_001_LOCAL_TOTAL_SCORE:
        return "improved submission candidate"
    if (
        best_ficr["ficr"] > ENS_001_LOCAL_FICR
        and best_ficr["total_score"] < ENS_001_LOCAL_TOTAL_SCORE
    ):
        return "FICR-diversity reference"
    if best_total_score["candidate_name"] == "lgbm_006_reference_ens001_no_change":
        return "keep ens_001_simple_avg_submit as current best"
    return "failed postprocessing reference"


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

    candidate_predictions: dict[str, pd.DataFrame] = {}
    candidate_results: list[dict[str, Any]] = []
    for candidate in build_candidates(base_prediction):
        candidate_predictions[candidate["name"]] = candidate["prediction"]
        candidate_results.append(score_candidate(candidate, y_true))

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
    final_decision = decision_for_results(best_total_score, best_ficr)

    best_prediction = candidate_predictions[best_total_score["candidate_name"]].copy()
    best_prediction["forecast_kst_dtm"] = pd.to_datetime(
        best_prediction["forecast_kst_dtm"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    best_prediction.to_csv(best_prediction_path, index=False, encoding="utf-8-sig")

    summary = {
        "exp_id": EXP_ID,
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Validation-only FICR-focused postprocessing test on the current best "
            "validation prediction. No models are trained, no test predictions are "
            "created, and no submission CSV is created."
        ),
        "comparison_references": {
            "current_best_public_submission": "ens_001_simple_avg_submit",
            "ens_001_validation_prediction": str(INPUT_PREDICTION_PATH.relative_to(ROOT)),
            "ens_001_validation_total_score": ENS_001_LOCAL_TOTAL_SCORE,
            "ens_001_validation_ficr": ENS_001_LOCAL_FICR,
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
        "alignment": {
            "key": "forecast_kst_dtm",
            "status": "input validation prediction exactly matches sorted 2024 validation timestamps",
        },
        "postprocessing": {
            "families": [
                "reference",
                "global_scale",
                "group_specific_scale",
                "ramp_amplification",
                "high_end_shrink",
            ],
            "prediction_clipping": "Per target group, clipped to [0, group_capacity] before scoring.",
        },
        "candidates": candidates_by_total,
        "selected_best_by_total_score": best_total_score,
        "selected_best_by_ficr": best_ficr,
        "any_candidate_beat_ens_001_total_score": any(
            item["beats_ens_001_validation_total_score"] for item in candidate_results
        ),
        "any_candidate_beat_ens_001_ficr": any(
            item["beats_ens_001_validation_ficr"] for item in candidate_results
        ),
        "final_decision": final_decision,
        "artifacts": {
            "best_valid_predictions": str(best_prediction_path.relative_to(ROOT)),
            "run_summary": str(log_path.relative_to(ROOT)),
        },
    }

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("candidates_sorted_by_total_score")
    print("candidate,total_score,one_minus_nmae,ficr,family,beats_total,beats_ficr")
    for result in candidates_by_total:
        print(
            f"{result['candidate_name']},"
            f"{result['total_score']:.10f},"
            f"{result['one_minus_nmae']:.10f},"
            f"{result['ficr']:.10f},"
            f"{result['postprocessing_family']},"
            f"{result['beats_ens_001_validation_total_score']},"
            f"{result['beats_ens_001_validation_ficr']}"
        )
    print("top_10_candidates_sorted_by_ficr")
    print("candidate,total_score,one_minus_nmae,ficr,family,beats_total,beats_ficr")
    for result in candidates_by_ficr[:10]:
        print(
            f"{result['candidate_name']},"
            f"{result['total_score']:.10f},"
            f"{result['one_minus_nmae']:.10f},"
            f"{result['ficr']:.10f},"
            f"{result['postprocessing_family']},"
            f"{result['beats_ens_001_validation_total_score']},"
            f"{result['beats_ens_001_validation_ficr']}"
        )
    print(f"selected_best_by_total_score: {best_total_score['candidate_name']}")
    print(f"best_total_score: {best_total_score['total_score']:.10f}")
    print(f"selected_best_by_ficr: {best_ficr['candidate_name']}")
    print(f"best_ficr: {best_ficr['ficr']:.10f}")
    print(
        "any_candidate_beat_ens_001_total_score: "
        f"{summary['any_candidate_beat_ens_001_total_score']}"
    )
    print(
        "any_candidate_beat_ens_001_ficr: "
        f"{summary['any_candidate_beat_ens_001_ficr']}"
    )
    print(f"final_decision: {final_decision}")
    print(f"Saved best validation predictions: {best_prediction_path.relative_to(ROOT)}")
    print(f"Saved run summary: {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

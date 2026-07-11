from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import product
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


EXP_ID = "ens_004_group_scale_cv"
RUN_ID = "ens_004_group_scale_cv_valid_2024"
INPUT_PREDICTION_PATH = (
    ROOT / "outputs" / "predictions" / "ens_001_simple_avg_valid_2024_best.csv"
)

LGBM_008_TOTAL_SCORE = 0.6184267418
LGBM_008_ONE_MINUS_NMAE = 0.8671491370
LGBM_008_FICR = 0.3697043467

BASELINE_GLOBAL_SCALES = [1.00, 1.03, 1.08, 1.095, 1.10, 1.105]
COARSE_SCALES = [1.00, 1.03, 1.06, 1.08, 1.095, 1.10, 1.105, 1.12]
FINE_STEP = 0.005
FINE_RADIUS = 0.02
FINE_MIN = 0.98
FINE_MAX = 1.16


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


def scale_key(group_scales: dict[str, float]) -> tuple[float, ...]:
    return tuple(float(group_scales[target]) for target in TARGET_COLS)


def format_scale(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def candidate_name(prefix: str, group_scales: dict[str, float]) -> str:
    parts = [format_scale(group_scales[target]).replace(".", "p") for target in TARGET_COLS]
    return f"{EXP_ID}_{prefix}_g1_{parts[0]}_g2_{parts[1]}_g3_{parts[2]}"


def scaled_prediction(
    base_prediction: pd.DataFrame,
    group_scales: dict[str, float],
) -> pd.DataFrame:
    prediction = base_prediction.copy()
    for target in TARGET_COLS:
        prediction[target] = (prediction[target] * group_scales[target]).clip(
            lower=0.0,
            upper=CAPACITY_KWH[target],
        )
    return prediction


def score_candidate(
    group_scales: dict[str, float],
    base_prediction: pd.DataFrame,
    y_true: pd.DataFrame,
    search_stage: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    prediction = scaled_prediction(base_prediction, group_scales)
    scores = calculate_metric(
        y_true,
        prediction.loc[:, list(TARGET_COLS)].reset_index(drop=True),
    )

    result = {
        "candidate_name": candidate_name(search_stage, group_scales),
        "search_stage": search_stage,
        "group_scales": {target: float(group_scales[target]) for target in TARGET_COLS},
        "total_score": scores["total_score"],
        "one_minus_nmae": scores["one_minus_nmae"],
        "ficr": scores["ficr"],
        "group_nmae": scores["group_nmae"],
        "group_ficr": scores["group_ficr"],
        "delta_vs_lgbm_008_total_score": scores["total_score"] - LGBM_008_TOTAL_SCORE,
        "delta_vs_lgbm_008_one_minus_nmae": (
            scores["one_minus_nmae"] - LGBM_008_ONE_MINUS_NMAE
        ),
        "delta_vs_lgbm_008_ficr": scores["ficr"] - LGBM_008_FICR,
    }
    return result, prediction


def evaluate_candidates(
    candidates: list[dict[str, float]],
    base_prediction: pd.DataFrame,
    y_true: pd.DataFrame,
    search_stage: str,
) -> tuple[list[dict[str, Any]], dict[tuple[float, ...], pd.DataFrame]]:
    results: list[dict[str, Any]] = []
    predictions: dict[tuple[float, ...], pd.DataFrame] = {}
    seen: set[tuple[float, ...]] = set()

    for group_scales in candidates:
        key = scale_key(group_scales)
        if key in seen:
            continue
        seen.add(key)
        result, prediction = score_candidate(group_scales, base_prediction, y_true, search_stage)
        results.append(result)
        predictions[key] = prediction

    return results, predictions


def make_global_candidates(scales: list[float]) -> list[dict[str, float]]:
    return [{target: scale for target in TARGET_COLS} for scale in scales]


def make_coarse_candidates() -> list[dict[str, float]]:
    candidates = []
    for values in product(COARSE_SCALES, repeat=len(TARGET_COLS)):
        candidates.append(dict(zip(TARGET_COLS, values)))
    return candidates


def fine_values(center: float) -> list[float]:
    start = max(FINE_MIN, center - FINE_RADIUS)
    end = min(FINE_MAX, center + FINE_RADIUS)
    values = []
    current = round(start, 3)
    while current <= end + 1e-9:
        values.append(round(current, 3))
        current = round(current + FINE_STEP, 3)
    return values


def make_fine_candidates(best_coarse: dict[str, Any]) -> list[dict[str, float]]:
    values_by_target = [
        fine_values(float(best_coarse["group_scales"][target])) for target in TARGET_COLS
    ]
    candidates = []
    seen: set[tuple[float, ...]] = set()
    for values in product(*values_by_target):
        key = tuple(values)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(dict(zip(TARGET_COLS, values)))
    return candidates


def sort_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(results, key=lambda item: item["total_score"], reverse=True)


def final_decision(best: dict[str, Any]) -> str:
    if (
        best["total_score"] > LGBM_008_TOTAL_SCORE
        and best["one_minus_nmae"] >= LGBM_008_ONE_MINUS_NMAE
    ):
        return "strong future submission candidate"
    if best["total_score"] > LGBM_008_TOTAL_SCORE:
        return "possible FICR-driven candidate with private-risk caution"
    return "keep lgbm_008_scale_110_submit as current reference; do not create submission code"


def write_prediction(path: Path, prediction: pd.DataFrame) -> None:
    output = prediction.copy()
    output["forecast_kst_dtm"] = pd.to_datetime(output["forecast_kst_dtm"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    output.to_csv(path, index=False, encoding="utf-8-sig")


def print_candidate(label: str, result: dict[str, Any]) -> None:
    scales = ", ".join(
        f"{target}={result['group_scales'][target]:.3f}" for target in TARGET_COLS
    )
    print(label)
    print(
        f"{scales}, total_score={result['total_score']:.10f}, "
        f"one_minus_nmae={result['one_minus_nmae']:.10f}, ficr={result['ficr']:.10f}"
    )


def print_results(
    baseline_global_scores: list[dict[str, Any]],
    coarse_best: dict[str, Any],
    fine_best: dict[str, Any],
    selected_best: dict[str, Any],
    decision: str,
) -> None:
    print("baseline_global_scale_scores")
    print("scale,total_score,one_minus_nmae,ficr,delta_total_vs_lgbm008")
    for result in sorted(baseline_global_scores, key=lambda item: item["group_scales"]["kpx_group_1"]):
        scale = result["group_scales"]["kpx_group_1"]
        print(
            f"{scale:.3f},"
            f"{result['total_score']:.10f},"
            f"{result['one_minus_nmae']:.10f},"
            f"{result['ficr']:.10f},"
            f"{result['delta_vs_lgbm_008_total_score']:.10f}"
        )

    print_candidate("coarse_grid_best", coarse_best)
    print_candidate("fine_grid_best", fine_best)
    print_candidate("selected_best_group_scales_and_score", selected_best)
    print(
        "beats_lgbm_008_validation_reference: "
        f"{selected_best['total_score'] > LGBM_008_TOTAL_SCORE}"
    )
    print(
        "preserves_lgbm_008_one_minus_nmae: "
        f"{selected_best['one_minus_nmae'] >= LGBM_008_ONE_MINUS_NMAE}"
    )
    print(
        "comparison_vs_lgbm_008: "
        f"delta_total={selected_best['delta_vs_lgbm_008_total_score']:.10f}, "
        f"delta_one_minus_nmae={selected_best['delta_vs_lgbm_008_one_minus_nmae']:.10f}, "
        f"delta_ficr={selected_best['delta_vs_lgbm_008_ficr']:.10f}"
    )
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

    baseline_global_scores, _ = evaluate_candidates(
        make_global_candidates(BASELINE_GLOBAL_SCALES),
        base_prediction,
        y_true,
        "baseline_global",
    )

    coarse_results, _ = evaluate_candidates(
        make_coarse_candidates(),
        base_prediction,
        y_true,
        "coarse",
    )
    coarse_best = sort_results(coarse_results)[0]

    fine_results, fine_predictions = evaluate_candidates(
        make_fine_candidates(coarse_best),
        base_prediction,
        y_true,
        "fine",
    )
    fine_best = sort_results(fine_results)[0]
    selected_best = fine_best
    decision = final_decision(selected_best)

    write_prediction(
        best_prediction_path,
        fine_predictions[scale_key(selected_best["group_scales"])],
    )

    summary = {
        "exp_id": EXP_ID,
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Search separate validation-only scale factors for kpx_group_1, "
            "kpx_group_2, and kpx_group_3 instead of using one global scale."
        ),
        "current_best_public_reference": {
            "submission": "lgbm_008_scale_110_submit",
            "public_total_score": 0.6211026154,
            "public_one_minus_nmae": 0.8634833072,
            "public_ficr": 0.3787219236,
            "method": (
                "Global scale 1.10 applied to "
                "outputs/predictions/ens_001_simple_avg_test.csv, then capacity clipped."
            ),
        },
        "lgbm_008_validation_reference": {
            "experiment": "lgbm_008_scale_upper_sweep_cv",
            "global_scale": 1.10,
            "total_score": LGBM_008_TOTAL_SCORE,
            "one_minus_nmae": LGBM_008_ONE_MINUS_NMAE,
            "ficr": LGBM_008_FICR,
        },
        "source_validation_prediction": str(INPUT_PREDICTION_PATH.relative_to(ROOT)),
        "data": {
            "train_labels": "data/raw/train/train_labels.csv",
            "ldaps_train": "data/raw/train/ldaps_train.csv",
            "gfs_train": "data/raw/train/gfs_train.csv",
        },
        "validation_split_information": {
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
            "prediction_scaling": "Each target column is multiplied by its own group scale.",
            "prediction_clipping": "Per target group, clipped to [0, group_capacity] before scoring.",
            "baseline_global_scales": BASELINE_GLOBAL_SCALES,
            "coarse_group_scales": COARSE_SCALES,
            "fine_grid": {
                "center": coarse_best["group_scales"],
                "radius": FINE_RADIUS,
                "step": FINE_STEP,
                "clip_range": [FINE_MIN, FINE_MAX],
                "candidate_count": len(fine_results),
            },
        },
        "baseline_global_scale_scores": baseline_global_scores,
        "coarse_grid": {
            "candidate_count": len(coarse_results),
            "best": coarse_best,
            "top_20_by_total_score": sort_results(coarse_results)[:20],
        },
        "fine_grid": {
            "candidate_count": len(fine_results),
            "best": fine_best,
            "top_20_by_total_score": sort_results(fine_results)[:20],
        },
        "selected_best_group_scales": selected_best["group_scales"],
        "total_score": selected_best["total_score"],
        "one_minus_nmae": selected_best["one_minus_nmae"],
        "ficr": selected_best["ficr"],
        "group_nmae": selected_best["group_nmae"],
        "group_ficr": selected_best["group_ficr"],
        "comparison_versus_lgbm_008_validation": {
            "reference_total_score": LGBM_008_TOTAL_SCORE,
            "reference_one_minus_nmae": LGBM_008_ONE_MINUS_NMAE,
            "reference_ficr": LGBM_008_FICR,
            "delta_total_score": selected_best["delta_vs_lgbm_008_total_score"],
            "delta_one_minus_nmae": selected_best["delta_vs_lgbm_008_one_minus_nmae"],
            "delta_ficr": selected_best["delta_vs_lgbm_008_ficr"],
            "beats_total_score": selected_best["total_score"] > LGBM_008_TOTAL_SCORE,
            "preserves_or_improves_one_minus_nmae": (
                selected_best["one_minus_nmae"] >= LGBM_008_ONE_MINUS_NMAE
            ),
        },
        "final_decision": decision,
        "artifacts": {
            "best_valid_predictions": str(best_prediction_path.relative_to(ROOT)),
            "run_summary": str(log_path.relative_to(ROOT)),
        },
    }

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print_results(
        baseline_global_scores,
        coarse_best,
        fine_best,
        selected_best,
        decision,
    )
    print(f"Saved best validation predictions: {best_prediction_path.relative_to(ROOT)}")
    print(f"Saved run summary: {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

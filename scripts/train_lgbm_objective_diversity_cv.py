from __future__ import annotations

import itertools
import json
import random
import sys
from dataclasses import dataclass
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


EXP_ID = "lgbm_011_objective_diversity_cv"
RUN_ID = "lgbm_011_objective_diversity_cv_valid_2024"
RANDOM_SEED = 42
SCORE_EPS = 1e-10
SCALES = (1.00, 1.03, 1.08, 1.095, 1.10)
OBJECTIVES = ("regression", "regression_l1", "huber", "poisson", "tweedie")

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

BASE_PARAMS: dict[str, Any] = {
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
    "random_state": RANDOM_SEED,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": -1,
    "verbosity": -1,
}


@dataclass(frozen=True)
class CandidateResult:
    imputer: SimpleImputer
    models: dict[str, LGBMRegressor]
    feature_columns: list[str]
    predictions: pd.DataFrame
    train_rows: dict[str, int]
    params: dict[str, Any]


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


def objective_params(objective: str) -> dict[str, Any]:
    params = dict(BASE_PARAMS)
    params["objective"] = objective
    if objective == "tweedie":
        params["tweedie_variance_power"] = 1.5
    return params


def labels_are_nonnegative(y_train: pd.DataFrame) -> bool:
    for target in TARGET_COLS:
        values = y_train[target].dropna()
        if (values < 0).any():
            return False
    return True


def train_candidate(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    objective: str,
) -> CandidateResult:
    params = objective_params(objective)
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

        model = LGBMRegressor(**params)
        model.fit(x_train_imp.loc[train_mask], y_train.loc[train_mask, target])
        pred = model.predict(x_valid_imp)
        predictions[target] = np.clip(pred, 0.0, CAPACITY_KWH[target])
        models[target] = model
        train_rows[target] = int(train_mask.sum())

    return CandidateResult(
        imputer=imputer,
        models=models,
        feature_columns=list(x_train.columns),
        predictions=predictions,
        train_rows=train_rows,
        params=params,
    )


def score_prediction(y_true: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    return calculate_metric(
        y_true.reset_index(drop=True),
        prediction.loc[:, list(TARGET_COLS)].reset_index(drop=True),
    )


def format_score(scores: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_score": scores["total_score"],
        "one_minus_nmae": scores["one_minus_nmae"],
        "ficr": scores["ficr"],
        "group_nmae": scores["group_nmae"],
        "group_ficr": scores["group_ficr"],
    }


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


def weight_grid(names: list[str]) -> list[dict[str, float]]:
    if len(names) != 4:
        return []
    rows: list[dict[str, float]] = []
    for parts in itertools.product(range(11), repeat=4):
        if sum(parts) != 10:
            continue
        rows.append({name: part / 10.0 for name, part in zip(names, parts)})
    return rows


def final_decision(best_raw: dict[str, Any], best_scaled: dict[str, Any]) -> str:
    raw_useful = (
        best_raw["total_score"] > ENS_001_VALIDATION_REFERENCE["total_score"] + SCORE_EPS
    )
    scaled_submission_candidate = (
        best_scaled["total_score"] > LGBM_008_VALIDATION_REFERENCE["total_score"] + SCORE_EPS
    )
    if scaled_submission_candidate:
        return (
            "objective diversity is useful and conservative scaled validation beats "
            "lgbm_008 without group-wise scale; mark as future submission candidate, "
            "but no submission code or CSV created"
        )
    if raw_useful:
        return (
            "objective diversity is useful on raw validation, but conservative scaled "
            "score does not beat lgbm_008; keep lgbm_008_scale_110_submit as current "
            "best public reference"
        )
    return (
        "objective diversity does not beat ens_001 raw validation and scaled score "
        "does not beat lgbm_008; keep lgbm_008_scale_110_submit as current best "
        "public reference and continue validation-only feature/model experiments"
    )


def print_results(
    candidate_scores: list[dict[str, Any]],
    ensemble_scores: list[dict[str, Any]],
    weight_scores: list[dict[str, Any]],
    scale_scores: list[dict[str, Any]],
    best_raw: dict[str, Any],
    best_scaled: dict[str, Any],
    decision: str,
) -> None:
    print("per_objective_raw_scores")
    print("candidate,feature_set,objective,total_score,one_minus_nmae,ficr")
    for row in candidate_scores:
        print(
            f"{row['candidate']},"
            f"{row['feature_set']},"
            f"{row['objective']},"
            f"{row['total_score']:.10f},"
            f"{row['one_minus_nmae']:.10f},"
            f"{row['ficr']:.10f}"
        )

    print("ensemble_top_results")
    print("candidate,total_score,one_minus_nmae,ficr,weights")
    for row in ensemble_scores[:10]:
        print(
            f"{row['candidate']},"
            f"{row['total_score']:.10f},"
            f"{row['one_minus_nmae']:.10f},"
            f"{row['ficr']:.10f},"
            f"{json.dumps(row['weights'], sort_keys=True)}"
        )

    print("weight_search_top_results")
    print("candidate,total_score,one_minus_nmae,ficr,weights")
    for row in weight_scores[:10]:
        print(
            f"{row['candidate']},"
            f"{row['total_score']:.10f},"
            f"{row['one_minus_nmae']:.10f},"
            f"{row['ficr']:.10f},"
            f"{json.dumps(row['weights'], sort_keys=True)}"
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
    print(
        "beats_ens_001_validation_reference: "
        f"{best_raw['total_score'] > ENS_001_VALIDATION_REFERENCE['total_score'] + SCORE_EPS}"
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
    baseline_feature_count = len(baseline_features.columns)
    targeted_feature_columns = list(targeted_features.columns[baseline_feature_count:])

    feature_sets = {
        "baseline": {
            "features": baseline_features,
            "description": "existing lgbm_003 baseline calendar + mean LDAPS/GFS feature matrix",
        },
        "targeted": {
            "features": targeted_features,
            "description": "existing lgbm_005 targeted-weather feature matrix",
        },
    }

    nonnegative_labels = labels_are_nonnegative(y_train)
    attempted: list[dict[str, Any]] = []
    skipped_failed: list[dict[str, Any]] = []
    results: dict[str, CandidateResult] = {}
    predictions: dict[str, pd.DataFrame] = {}
    candidate_scores: list[dict[str, Any]] = []

    for feature_set, feature_info in feature_sets.items():
        features = feature_info["features"]
        for objective in OBJECTIVES:
            candidate = f"{feature_set}_{objective}"
            attempted.append(
                {
                    "candidate": candidate,
                    "feature_set": feature_set,
                    "objective": objective,
                }
            )
            if objective in {"poisson", "tweedie"} and not nonnegative_labels:
                skipped_failed.append(
                    {
                        "candidate": candidate,
                        "feature_set": feature_set,
                        "objective": objective,
                        "status": "skipped",
                        "reason": "training labels include negative values",
                    }
                )
                continue
            try:
                result = train_candidate(
                    x_train=features.loc[train_mask],
                    y_train=y_train,
                    x_valid=features.loc[valid_mask],
                    objective=objective,
                )
            except Exception as exc:
                skipped_failed.append(
                    {
                        "candidate": candidate,
                        "feature_set": feature_set,
                        "objective": objective,
                        "status": "failed",
                        "reason": repr(exc),
                    }
                )
                continue

            results[candidate] = result
            predictions[candidate] = result.predictions.reset_index(drop=True)
            scores = score_prediction(y_valid, result.predictions)
            candidate_scores.append(
                {
                    "candidate": candidate,
                    "feature_set": feature_set,
                    "objective": objective,
                    "feature_count": len(result.feature_columns),
                    "target_train_rows": result.train_rows,
                    **format_score(scores),
                }
            )

    if not candidate_scores:
        raise RuntimeError("No objective-diversity candidates completed successfully.")

    all_raw_predictions = dict(predictions)
    ensemble_scores: list[dict[str, Any]] = []

    reference_weights = {"baseline_regression": 0.5, "targeted_regression": 0.5}
    if all(name in predictions for name in reference_weights):
        pred = weighted_average(predictions, reference_weights)
        all_raw_predictions["reference_050_baseline_regression_050_targeted_regression"] = pred
        scores = score_prediction(y_valid, pred)
        ensemble_scores.append(
            {
                "candidate": "reference_050_baseline_regression_050_targeted_regression",
                "type": "existing_reference_style",
                "weights": reference_weights,
                **format_score(scores),
            }
        )
    else:
        skipped_failed.append(
            {
                "candidate": "reference_050_baseline_regression_050_targeted_regression",
                "status": "skipped",
                "reason": "baseline_regression or targeted_regression prediction missing",
            }
        )

    top_candidates = [
        row["candidate"]
        for row in sorted(candidate_scores, key=lambda item: item["total_score"], reverse=True)
    ][:4]
    for k in (2, 3, 4):
        if len(top_candidates) < k:
            continue
        names = top_candidates[:k]
        weights = {name: 1.0 / k for name in names}
        candidate = f"equal_average_top_{k}_raw_objective_candidates"
        pred = weighted_average(predictions, weights)
        all_raw_predictions[candidate] = pred
        scores = score_prediction(y_valid, pred)
        ensemble_scores.append(
            {
                "candidate": candidate,
                "type": f"equal_average_top_{k}",
                "weights": weights,
                **format_score(scores),
            }
        )

    weight_search_scores: list[dict[str, Any]] = []
    for i, weights in enumerate(weight_grid(top_candidates), start=1):
        candidate = f"top4_weight_search_step_0p1_{i:03d}"
        pred = weighted_average(predictions, weights)
        all_raw_predictions[candidate] = pred
        scores = score_prediction(y_valid, pred)
        weight_search_scores.append(
            {
                "candidate": candidate,
                "type": "top4_weight_search_step_0.1",
                "weights": weights,
                **format_score(scores),
            }
        )

    candidate_scores_by_total = sorted(
        candidate_scores,
        key=lambda item: item["total_score"],
        reverse=True,
    )
    ensemble_scores_by_total = sorted(
        ensemble_scores,
        key=lambda item: item["total_score"],
        reverse=True,
    )
    weight_scores_by_total = sorted(
        weight_search_scores,
        key=lambda item: item["total_score"],
        reverse=True,
    )
    raw_score_rows = candidate_scores + ensemble_scores + weight_search_scores
    best_raw = max(raw_score_rows, key=lambda item: item["total_score"])
    best_raw_prediction = build_prediction_frame(
        valid_times,
        all_raw_predictions[best_raw["candidate"]],
    )

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
    decision = final_decision(best_raw, best_scaled)

    write_prediction(raw_prediction_path, best_raw_prediction)
    write_prediction(best_scaled_prediction_path, scaled_predictions[best_scaled["scale"]])

    summary = {
        "exp_id": EXP_ID,
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Validation-only LightGBM objective-diversity experiment on existing "
            "safe feature sets. No test predictions and no submission CSV are created."
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
        "feature_sets_used": {
            name: {
                "description": info["description"],
                "feature_count": int(len(info["features"].columns)),
            }
            for name, info in feature_sets.items()
        },
        "targeted_weather_features": {
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
        "objective_candidates_attempted": attempted,
        "skipped_failed_objectives": skipped_failed,
        "preprocessing": {
            "missing_value_imputation": "SimpleImputer(strategy='median') fit on training rows only for each feature_set/objective candidate.",
            "prediction_clipping": "Each target prediction clipped to [0, group_capacity].",
        },
        "model": {
            "class": "lightgbm.LGBMRegressor",
            "per_target_models": list(TARGET_COLS),
            "base_params": BASE_PARAMS,
            "objectives": list(OBJECTIVES),
        },
        "per_candidate_raw_scores": candidate_scores_by_total,
        "ensemble_scores": ensemble_scores_by_total,
        "weight_search_scores": weight_scores_by_total,
        "best_raw_candidate": best_raw,
        "conservative_global_scale_scores": scale_scores_by_total,
        "selected_best_scaled_candidate": best_scaled,
        "comparisons": {
            "vs_ens_001_simple_avg_validation": {
                "reference": ENS_001_VALIDATION_REFERENCE,
                "best_raw_total_score_delta": (
                    best_raw["total_score"] - ENS_001_VALIDATION_REFERENCE["total_score"]
                ),
                "best_raw_beats_reference": (
                    best_raw["total_score"]
                    > ENS_001_VALIDATION_REFERENCE["total_score"] + SCORE_EPS
                ),
            },
            "vs_lgbm_008_validation_global_1p10": {
                "reference": LGBM_008_VALIDATION_REFERENCE,
                "best_scaled_total_score_delta": (
                    best_scaled["total_score"]
                    - LGBM_008_VALIDATION_REFERENCE["total_score"]
                ),
                "best_scaled_beats_reference": (
                    best_scaled["total_score"]
                    > LGBM_008_VALIDATION_REFERENCE["total_score"] + SCORE_EPS
                ),
            },
            "vs_current_best_public_lgbm_008_scale_110_submit": {
                "reference": CURRENT_BEST_PUBLIC_REFERENCE,
                "best_scaled_total_score_minus_public_reference": (
                    best_scaled["total_score"]
                    - CURRENT_BEST_PUBLIC_REFERENCE["total_score"]
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
                    "params": result.params,
                }
                for name, result in results.items()
            },
            "capacity_kwh": CAPACITY_KWH,
            "per_candidate_raw_scores": candidate_scores_by_total,
            "ensemble_scores": ensemble_scores_by_total,
            "weight_search_scores": weight_scores_by_total,
            "best_raw_candidate": best_raw,
            "conservative_global_scale_scores": scale_scores_by_total,
            "selected_best_scaled_candidate": best_scaled,
            "final_decision": decision,
        },
        model_path,
    )

    print_results(
        candidate_scores_by_total,
        ensemble_scores_by_total,
        weight_scores_by_total,
        scale_scores_by_total,
        best_raw,
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

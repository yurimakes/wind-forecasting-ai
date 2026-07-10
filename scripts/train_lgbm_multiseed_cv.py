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
from dacon_wind.features import build_baseline_train_frame, build_feature_matrix
from dacon_wind.metric import CAPACITY_KWH, TARGET_COLS, calculate_metric
from dacon_wind.models import LGBMResult


EXP_ID = "lgbm_009_multi_seed_lgbm_cv"
RUN_ID = "lgbm_009_multi_seed_lgbm_cv_valid_2024"
SEEDS = (42, 2024, 2025, 777, 1004)
SCALES = (1.00, 1.03, 1.08, 1.095, 1.10, 1.105)

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
    "method": (
        "scale 1.10 applied to outputs/predictions/ens_001_simple_avg_test.csv, "
        "then capacity clipped"
    ),
}

BEST_PARAMS_BASE: dict[str, Any] = {
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
            "Remove or rename them before rerunning this validation script."
        )


def lgbm_params(seed: int) -> dict[str, Any]:
    params = dict(BEST_PARAMS_BASE)
    params["random_state"] = seed
    return params


def train_seed_models(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    seed: int,
) -> LGBMResult:
    set_reproducible_seed(seed)

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

        model = LGBMRegressor(**lgbm_params(seed))
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


def score_prediction(y_true: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    return calculate_metric(
        y_true.reset_index(drop=True),
        prediction.loc[:, list(TARGET_COLS)].reset_index(drop=True),
    )


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


def scale_prediction(prediction: pd.DataFrame, scale: float) -> pd.DataFrame:
    out = prediction.copy()
    for target in TARGET_COLS:
        out[target] = (out[target] * scale).clip(
            lower=0.0,
            upper=CAPACITY_KWH[target],
        )
    return out


def write_prediction(path: Path, prediction: pd.DataFrame) -> None:
    output = prediction.copy()
    output["forecast_kst_dtm"] = pd.to_datetime(output["forecast_kst_dtm"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    output.to_csv(path, index=False, encoding="utf-8-sig")


def score_scaled_candidates(
    y_true: pd.DataFrame,
    raw_prediction: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[float, pd.DataFrame]]:
    candidate_scores: list[dict[str, Any]] = []
    predictions_by_scale: dict[float, pd.DataFrame] = {}

    for scale in SCALES:
        scaled = scale_prediction(raw_prediction, scale)
        scores = score_prediction(y_true, scaled)
        candidate_scores.append(
            {
                "scale": scale,
                "total_score": scores["total_score"],
                "one_minus_nmae": scores["one_minus_nmae"],
                "ficr": scores["ficr"],
                "group_nmae": scores["group_nmae"],
                "group_ficr": scores["group_ficr"],
                "beats_lgbm_008_validation_total_score": (
                    scores["total_score"]
                    > LGBM_008_VALIDATION_REFERENCE["total_score"]
                ),
                "preserves_lgbm_008_validation_one_minus_nmae": (
                    scores["one_minus_nmae"]
                    >= LGBM_008_VALIDATION_REFERENCE["one_minus_nmae"]
                ),
            }
        )
        predictions_by_scale[scale] = scaled

    return candidate_scores, predictions_by_scale


def final_decision(best_scaled: dict[str, Any]) -> str:
    beats_lgbm_008 = best_scaled["beats_lgbm_008_validation_total_score"]
    preserves_nmae = best_scaled["preserves_lgbm_008_validation_one_minus_nmae"]
    improves_ficr = best_scaled["ficr"] > LGBM_008_VALIDATION_REFERENCE["ficr"]

    if beats_lgbm_008 and preserves_nmae:
        return "strong candidate for future submission generation"
    if improves_ficr and not preserves_nmae:
        return "FICR-risk; keep lgbm_008_scale_110_submit as current reference"
    return "does not beat lgbm_008 validation; keep lgbm_008_scale_110_submit as current reference"


def print_results(
    per_seed_scores: list[dict[str, Any]],
    raw_score: dict[str, Any],
    scaled_scores_by_total: list[dict[str, Any]],
    best_scaled: dict[str, Any],
    decision: str,
) -> None:
    print("per_seed_scores")
    print("seed,total_score,one_minus_nmae,ficr")
    for result in per_seed_scores:
        print(
            f"{result['seed']},"
            f"{result['total_score']:.10f},"
            f"{result['one_minus_nmae']:.10f},"
            f"{result['ficr']:.10f}"
        )

    print("raw_ensemble_score")
    print(
        f"total_score={raw_score['total_score']:.10f},"
        f"one_minus_nmae={raw_score['one_minus_nmae']:.10f},"
        f"ficr={raw_score['ficr']:.10f}"
    )

    print("scaled_candidate_scores_sorted_by_total_score")
    print("scale,total_score,one_minus_nmae,ficr,beats_lgbm_008_total,preserves_lgbm_008_nmae")
    for result in scaled_scores_by_total:
        print(
            f"{result['scale']:.3f},"
            f"{result['total_score']:.10f},"
            f"{result['one_minus_nmae']:.10f},"
            f"{result['ficr']:.10f},"
            f"{result['beats_lgbm_008_validation_total_score']},"
            f"{result['preserves_lgbm_008_validation_one_minus_nmae']}"
        )

    print(f"selected_best_scale: {best_scaled['scale']:.3f}")
    print(f"selected_best_total_score: {best_scaled['total_score']:.10f}")
    print(f"selected_best_one_minus_nmae: {best_scaled['one_minus_nmae']:.10f}")
    print(f"selected_best_ficr: {best_scaled['ficr']:.10f}")
    print(
        "beats_lgbm_008_validation_reference: "
        f"{best_scaled['beats_lgbm_008_validation_total_score']}"
    )
    print(f"final_decision: {decision}")


def main() -> None:
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
    features = build_feature_matrix(train_frame)

    train_mask, valid_mask = split_train_valid_2024(train_frame["forecast_kst_dtm"])
    x_train = features.loc[train_mask]
    y_train = train_frame.loc[train_mask, list(TARGET_COLS)]
    x_valid = features.loc[valid_mask]
    y_valid = train_frame.loc[valid_mask, list(TARGET_COLS)]
    valid_times = train_frame.loc[valid_mask, "forecast_kst_dtm"]

    seed_results: dict[int, LGBMResult] = {}
    seed_predictions: list[pd.DataFrame] = []
    per_seed_scores: list[dict[str, Any]] = []

    for seed in SEEDS:
        result = train_seed_models(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            seed=seed,
        )
        seed_results[seed] = result
        seed_predictions.append(result.predictions.reset_index(drop=True))
        scores = score_prediction(y_valid, result.predictions)
        per_seed_scores.append(
            {
                "seed": seed,
                "total_score": scores["total_score"],
                "one_minus_nmae": scores["one_minus_nmae"],
                "ficr": scores["ficr"],
                "group_nmae": scores["group_nmae"],
                "group_ficr": scores["group_ficr"],
                "target_train_rows": result.train_rows,
            }
        )

    raw_average = pd.DataFrame(index=seed_predictions[0].index)
    for target in TARGET_COLS:
        stacked = np.column_stack(
            [prediction[target].to_numpy(dtype=float) for prediction in seed_predictions]
        )
        raw_average[target] = np.clip(
            np.mean(stacked, axis=1),
            0.0,
            CAPACITY_KWH[target],
        )

    raw_prediction = build_prediction_frame(valid_times, raw_average)
    raw_score = score_prediction(y_valid, raw_prediction)

    scaled_scores, predictions_by_scale = score_scaled_candidates(
        y_valid,
        raw_prediction,
    )
    scaled_scores_by_total = sorted(
        scaled_scores,
        key=lambda item: item["total_score"],
        reverse=True,
    )
    best_scaled = scaled_scores_by_total[0]
    decision = final_decision(best_scaled)

    write_prediction(raw_prediction_path, raw_prediction)
    write_prediction(
        best_scaled_prediction_path,
        predictions_by_scale[best_scaled["scale"]],
    )

    baseline_reference = {
        "exp_id": "ens_001_simple_avg",
        "total_score": 0.6037465036,
        "one_minus_nmae": 0.8680552297,
        "ficr": 0.3394377775,
    }

    summary = {
        "exp_id": EXP_ID,
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Validation-only multi-seed LightGBM ensemble to test prediction "
            "stability and 1-nMAE before considering any future submission."
        ),
        "random_seeds": list(SEEDS),
        "data": {
            "train_labels": "data/raw/train/train_labels.csv",
            "ldaps_train": "data/raw/train/ldaps_train.csv",
            "gfs_train": "data/raw/train/gfs_train.csv",
        },
        "split": {
            "function": "split_train_valid_2024",
            "train": "year < 2024",
            "valid": "year == 2024",
            "train_rows": int(train_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
            "forecast_start": str(pd.to_datetime(valid_times).min()),
            "forecast_end": str(pd.to_datetime(valid_times).max()),
        },
        "features": {
            "approach": "Baseline calendar features plus mean-aggregated LDAPS/GFS by forecast_kst_dtm.",
            "feature_count": int(len(features.columns)),
            "feature_columns": list(features.columns),
        },
        "preprocessing": {
            "missing_value_imputation": "SimpleImputer(strategy='median') fit on training rows only per seed.",
            "prediction_clipping": "Per target group, clipped to [0, group_capacity] before scoring.",
        },
        "model": {
            "class": "lightgbm.LGBMRegressor",
            "per_target_models": list(TARGET_COLS),
            "params_without_seed": BEST_PARAMS_BASE,
            "seeds": list(SEEDS),
        },
        "references": {
            "ens_001_simple_avg_validation_raw": baseline_reference,
            "lgbm_008_validation_reference": LGBM_008_VALIDATION_REFERENCE,
            "current_best_public_reference": CURRENT_BEST_PUBLIC_REFERENCE,
        },
        "per_seed_scores": per_seed_scores,
        "raw_ensemble_score": raw_score,
        "scaled_candidate_scores": scaled_scores_by_total,
        "selected_best_scale": best_scaled["scale"],
        "selected_best_scaled_score": best_scaled,
        "comparison": {
            "vs_lgbm_008_validation_total_score_delta": (
                best_scaled["total_score"]
                - LGBM_008_VALIDATION_REFERENCE["total_score"]
            ),
            "vs_lgbm_008_validation_one_minus_nmae_delta": (
                best_scaled["one_minus_nmae"]
                - LGBM_008_VALIDATION_REFERENCE["one_minus_nmae"]
            ),
            "vs_lgbm_008_validation_ficr_delta": (
                best_scaled["ficr"] - LGBM_008_VALIDATION_REFERENCE["ficr"]
            ),
            "vs_current_best_public_total_score_delta": (
                best_scaled["total_score"]
                - CURRENT_BEST_PUBLIC_REFERENCE["total_score"]
            ),
            "vs_current_best_public_one_minus_nmae_delta": (
                best_scaled["one_minus_nmae"]
                - CURRENT_BEST_PUBLIC_REFERENCE["one_minus_nmae"]
            ),
            "vs_current_best_public_ficr_delta": (
                best_scaled["ficr"] - CURRENT_BEST_PUBLIC_REFERENCE["ficr"]
            ),
        },
        "final_decision": decision,
        "artifacts": {
            "raw_valid_predictions": str(raw_prediction_path.relative_to(ROOT)),
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
            "random_seeds": list(SEEDS),
            "imputers": {
                seed: result.imputer for seed, result in seed_results.items()
            },
            "models": {seed: result.models for seed, result in seed_results.items()},
            "feature_columns": list(features.columns),
            "model_params_without_seed": BEST_PARAMS_BASE,
            "capacity_kwh": CAPACITY_KWH,
            "per_seed_scores": per_seed_scores,
            "raw_ensemble_score": raw_score,
            "scaled_candidate_scores": scaled_scores_by_total,
            "selected_best_scale": best_scaled["scale"],
            "selected_best_scaled_score": best_scaled,
            "final_decision": decision,
        },
        model_path,
    )

    print_results(
        per_seed_scores,
        raw_score,
        scaled_scores_by_total,
        best_scaled,
        decision,
    )
    print(f"Saved raw validation predictions: {raw_prediction_path.relative_to(ROOT)}")
    print(
        "Saved best scaled validation predictions: "
        f"{best_scaled_prediction_path.relative_to(ROOT)}"
    )
    print(f"Saved run summary: {log_path.relative_to(ROOT)}")
    print(f"Saved model artifact: {model_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

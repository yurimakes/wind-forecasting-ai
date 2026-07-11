from __future__ import annotations

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

from dacon_wind.data import load_test_data, load_train_data
from dacon_wind.features import (
    TARGETED_COMPONENT_DIFFS,
    TARGETED_WIND_VECTOR_PAIRS,
    build_baseline_test_frame,
    build_baseline_train_frame,
    build_feature_matrix,
    build_targeted_weather_feature_matrix,
)
from dacon_wind.metric import CAPACITY_KWH, TARGET_COLS
from validate_submission import validate_submission


EXP_ID = "lgbm_011_objective_diversity_submit"
RUN_ID = "lgbm_011_objective_diversity_submit"
SUBMISSION_NAME = "lgbm_011_objective_diversity.csv"
RANDOM_SEED = 42
GLOBAL_SCALE = 1.095

SAMPLE_SUBMISSION = ROOT / "data" / "raw" / "sample_submission.csv"
OUTPUT_TEST_PREDICTION = (
    ROOT / "outputs" / "predictions" / "lgbm_011_objective_diversity_test.csv"
)
OUTPUT_SUBMISSION = ROOT / "submissions" / SUBMISSION_NAME
OUTPUT_LOG = ROOT / "outputs" / "logs" / "lgbm_011_objective_diversity_submit.json"
OUTPUT_MODEL = ROOT / "outputs" / "models" / "lgbm_011_objective_diversity_submit.joblib"

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

ENSEMBLE_WEIGHTS = {
    "baseline_regression_l1": 0.3,
    "targeted_regression_l1": 0.6,
    "targeted_regression": 0.1,
}

CANDIDATES = {
    "baseline_regression_l1": {
        "feature_set": "baseline",
        "objective": "regression_l1",
    },
    "targeted_regression_l1": {
        "feature_set": "targeted_weather",
        "objective": "regression_l1",
    },
    "targeted_regression": {
        "feature_set": "targeted_weather",
        "objective": "regression",
    },
}

VALIDATION_REFERENCE = {
    "experiment": "lgbm_011_objective_diversity_cv",
    "best_raw_candidate": "top4_weight_search_step_0p1_200",
    "best_raw_total_score": 0.6077784238,
    "best_raw_one_minus_nmae": 0.8683304324,
    "best_raw_ficr": 0.3472264153,
    "best_raw_weights": ENSEMBLE_WEIGHTS,
    "selected_global_scale": GLOBAL_SCALE,
    "best_scaled_total_score": 0.6214914951,
    "best_scaled_one_minus_nmae": 0.8674785360,
    "best_scaled_ficr": 0.3755044543,
    "comparison": {
        "beats_ens_001_raw_validation_total_score": 0.6037465036,
        "beats_lgbm_008_validation_total_score": 0.6184267418,
        "does_not_use_group_wise_scaling": True,
    },
    "decision": "future submission candidate",
}

CURRENT_BEST_PUBLIC_REFERENCE = {
    "experiment": "lgbm_008_scale_110_submit",
    "submission": "submissions/lgbm_008_scale_110.csv",
    "public_total_score": 0.6211026154,
    "public_one_minus_nmae": 0.8634833072,
    "public_ficr": 0.3787219236,
    "final_ranking_note": "Final ranking still depends on the private leaderboard.",
}


@dataclass(frozen=True)
class FullTrainCandidate:
    imputer: SimpleImputer
    models: dict[str, LGBMRegressor]
    feature_columns: list[str]
    train_rows_by_target: dict[str, int]
    params: dict[str, Any]
    predictions: pd.DataFrame


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
            "Remove or rename them before rerunning this script."
        )


def objective_params(objective: str) -> dict[str, Any]:
    params = dict(BASE_PARAMS)
    params["objective"] = objective
    return params


def assert_aligned(name: str, prediction: pd.DataFrame, sample: pd.DataFrame) -> None:
    if len(prediction) != len(sample):
        raise ValueError(
            f"{name} row count {len(prediction)} does not match sample row count {len(sample)}."
        )
    if not prediction["forecast_id"].equals(sample["forecast_id"]):
        raise ValueError(f"{name} forecast_id alignment mismatch.")
    if not prediction["forecast_kst_dtm"].equals(sample["forecast_kst_dtm"]):
        raise ValueError(f"{name} forecast_kst_dtm alignment mismatch.")


def align_test_features(
    feature_set_name: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> pd.DataFrame:
    missing = [col for col in x_train.columns if col not in x_test.columns]
    extra = [col for col in x_test.columns if col not in x_train.columns]
    if missing or extra:
        raise ValueError(
            f"{feature_set_name} feature mismatch. Missing in test={missing}; extra in test={extra}"
        )
    return x_test.loc[:, list(x_train.columns)].copy()


def train_full_candidate(
    candidate_name: str,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    objective: str,
) -> FullTrainCandidate:
    params = objective_params(objective)
    imputer = SimpleImputer(strategy="median")
    x_train_imp = pd.DataFrame(
        imputer.fit_transform(x_train),
        columns=x_train.columns,
        index=x_train.index,
    )
    x_test_imp = pd.DataFrame(
        imputer.transform(x_test),
        columns=x_test.columns,
        index=x_test.index,
    )

    predictions = pd.DataFrame(index=x_test.index)
    models: dict[str, LGBMRegressor] = {}
    train_rows_by_target: dict[str, int] = {}
    for target in TARGET_COLS:
        train_mask = y_train[target].notna()
        if not train_mask.any():
            raise ValueError(f"No non-null full-train labels for {candidate_name} {target}.")

        model = LGBMRegressor(**params)
        model.fit(x_train_imp.loc[train_mask], y_train.loc[train_mask, target])
        predictions[target] = model.predict(x_test_imp)
        models[target] = model
        train_rows_by_target[target] = int(train_mask.sum())

    return FullTrainCandidate(
        imputer=imputer,
        models=models,
        feature_columns=list(x_train.columns),
        train_rows_by_target=train_rows_by_target,
        params=params,
        predictions=predictions,
    )


def clip_targets(
    prediction: pd.DataFrame,
    stage: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    clipped = prediction.copy()
    lower_counts: dict[str, int] = {}
    upper_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    for target in TARGET_COLS:
        values = prediction[target].astype(float)
        capped = values.clip(lower=0.0, upper=CAPACITY_KWH[target])
        clipped[target] = capped
        lower_counts[target] = int((values < 0.0).sum())
        upper_counts[target] = int((values > CAPACITY_KWH[target]).sum())
        total_counts[target] = int((values != capped).sum())
    return clipped, {
        "stage": stage,
        "lower_clip_counts": lower_counts,
        "upper_clip_counts": upper_counts,
        "total_clip_counts": total_counts,
    }


def summarize_targets(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        target: {
            "min": float(pd.to_numeric(df[target]).min()),
            "max": float(pd.to_numeric(df[target]).max()),
            "mean": float(pd.to_numeric(df[target]).mean()),
        }
        for target in TARGET_COLS
    }


def format_submission_frame(sample: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    output = sample.copy()
    for target in TARGET_COLS:
        output[target] = prediction[target].to_numpy(dtype=float)
    output = output.loc[:, list(sample.columns)]
    output["forecast_kst_dtm"] = pd.to_datetime(output["forecast_kst_dtm"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return output


def main() -> None:
    set_reproducible_seed(RANDOM_SEED)
    for path in (
        OUTPUT_TEST_PREDICTION.parent,
        OUTPUT_SUBMISSION.parent,
        OUTPUT_LOG.parent,
        OUTPUT_MODEL.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
    ensure_new_artifacts(
        [OUTPUT_TEST_PREDICTION, OUTPUT_SUBMISSION, OUTPUT_LOG, OUTPUT_MODEL]
    )

    train_data = load_train_data(ROOT / "data" / "raw" / "train")
    test_data = load_test_data(
        ROOT / "data" / "raw" / "test",
        SAMPLE_SUBMISSION,
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
    sample = test_data.sample_submission.copy()

    assert_aligned("baseline test frame", test_frame, sample)

    baseline_train_features = build_feature_matrix(train_frame)
    baseline_test_features = align_test_features(
        "baseline",
        baseline_train_features,
        build_feature_matrix(test_frame),
    )
    targeted_train_features = build_targeted_weather_feature_matrix(train_frame)
    targeted_test_features = align_test_features(
        "targeted_weather",
        targeted_train_features,
        build_targeted_weather_feature_matrix(test_frame),
    )
    y_train = train_frame.loc[:, list(TARGET_COLS)]

    feature_sets = {
        "baseline": {
            "train": baseline_train_features,
            "test": baseline_test_features,
            "description": "existing baseline calendar + mean LDAPS/GFS feature matrix",
        },
        "targeted_weather": {
            "train": targeted_train_features,
            "test": targeted_test_features,
            "description": "existing targeted-weather feature matrix used by lgbm_011",
        },
    }

    trained: dict[str, FullTrainCandidate] = {}
    member_predictions: dict[str, pd.DataFrame] = {}
    for candidate_name, candidate_config in CANDIDATES.items():
        feature_set = feature_sets[candidate_config["feature_set"]]
        result = train_full_candidate(
            candidate_name=candidate_name,
            x_train=feature_set["train"],
            y_train=y_train,
            x_test=feature_set["test"],
            objective=candidate_config["objective"],
        )
        trained[candidate_name] = result
        member_predictions[candidate_name] = result.predictions.reset_index(drop=True)

    raw_ensemble = pd.DataFrame(index=sample.index)
    for target in TARGET_COLS:
        raw_ensemble[target] = sum(
            member_predictions[name][target].to_numpy(dtype=float) * weight
            for name, weight in ENSEMBLE_WEIGHTS.items()
        )

    raw_clipped, raw_clip_counts = clip_targets(
        raw_ensemble,
        "raw_weighted_ensemble_before_global_scale",
    )
    scaled_before_clip = raw_clipped.copy()
    for target in TARGET_COLS:
        scaled_before_clip[target] = scaled_before_clip[target] * GLOBAL_SCALE
    scaled_clipped, scaled_clip_counts = clip_targets(
        scaled_before_clip,
        "after_global_scale_1p095",
    )

    prediction_output = format_submission_frame(sample, scaled_clipped)
    prediction_output.to_csv(OUTPUT_TEST_PREDICTION, index=False, encoding="utf-8-sig")
    prediction_output.to_csv(OUTPUT_SUBMISSION, index=False, encoding="utf-8-sig")

    validation_errors = validate_submission(OUTPUT_SUBMISSION, SAMPLE_SUBMISSION)
    if validation_errors:
        raise ValueError(
            "Generated submission failed validation:\n"
            + "\n".join(f"- {error}" for error in validation_errors)
        )

    validation_result = {
        "script": "scripts/validate_submission.py",
        "submission": str(OUTPUT_SUBMISSION.relative_to(ROOT)),
        "passed": True,
        "errors": [],
    }

    model_artifact = {
        "exp_id": EXP_ID,
        "run_id": RUN_ID,
        "random_seed": RANDOM_SEED,
        "base_params": BASE_PARAMS,
        "candidates": {
            name: {
                "feature_set": CANDIDATES[name]["feature_set"],
                "objective": CANDIDATES[name]["objective"],
                "weight": ENSEMBLE_WEIGHTS[name],
                "imputer": result.imputer,
                "models": result.models,
                "feature_columns": result.feature_columns,
                "train_rows_by_target": result.train_rows_by_target,
                "params": result.params,
            }
            for name, result in trained.items()
        },
        "capacity_kwh": CAPACITY_KWH,
        "global_scale": GLOBAL_SCALE,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
    }
    joblib.dump(model_artifact, OUTPUT_MODEL)

    summary = {
        "exp_id": EXP_ID,
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Train the lgbm_011 objective-diversity ensemble on all available train rows "
            "and create a globally scaled DACON submission."
        ),
        "random_seed": RANDOM_SEED,
        "data": {
            "train_scope": "all rows loaded from data/raw/train",
            "test_scope": "all rows loaded from data/raw/test and aligned to data/raw/sample_submission.csv",
            "full_train_rows": int(len(train_frame)),
            "test_rows": int(len(sample)),
            "target_non_null_train_rows": {
                target: int(y_train[target].notna().sum()) for target in TARGET_COLS
            },
        },
        "feature_sets": {
            name: {
                "description": info["description"],
                "feature_count": int(len(info["train"].columns)),
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
        },
        "objectives": {
            name: {
                "feature_set": CANDIDATES[name]["feature_set"],
                "objective": CANDIDATES[name]["objective"],
                "params": trained[name].params,
                "train_rows_by_target": trained[name].train_rows_by_target,
            }
            for name in CANDIDATES
        },
        "preprocessing": {
            "missing_value_imputation": (
                "SimpleImputer(strategy='median') fit on full training rows only "
                "for each feature_set/objective candidate."
            ),
            "uses_validation_labels": False,
            "uses_remote_model_api": False,
        },
        "ensemble": {
            "weights": ENSEMBLE_WEIGHTS,
            "raw_operation": (
                "0.3*baseline_regression_l1 + 0.6*targeted_regression_l1 + "
                "0.1*targeted_regression"
            ),
            "raw_clipping": "clip weighted raw ensemble predictions to [0, group_capacity]",
            "global_scale": GLOBAL_SCALE,
            "scaled_clipping": "clip scaled predictions again to [0, group_capacity]",
        },
        "alignment": {
            "sample_submission": str(SAMPLE_SUBMISSION.relative_to(ROOT)),
            "keys": ["forecast_id", "forecast_kst_dtm"],
            "status": "test frame and output submission match sample_submission order",
            "rows": int(len(sample)),
        },
        "capacity_clipping_counts": {
            "capacity_kwh": {target: CAPACITY_KWH[target] for target in TARGET_COLS},
            "raw_weighted_ensemble": raw_clip_counts,
            "after_global_scale": scaled_clip_counts,
        },
        "target_summary": {
            "raw_ensemble_before_clipping": summarize_targets(raw_ensemble),
            "raw_ensemble_after_clipping": summarize_targets(raw_clipped),
            "after_scale_before_final_clipping": summarize_targets(scaled_before_clip),
            "final_output": summarize_targets(prediction_output),
        },
        "validation_reference": VALIDATION_REFERENCE,
        "current_best_public_reference": CURRENT_BEST_PUBLIC_REFERENCE,
        "validation_result": validation_result,
        "public_score": "pending",
        "caution": (
            "This candidate beats the current validation reference without group-wise scaling, "
            "but final ranking depends on the private leaderboard. Avoid excessive public "
            "leaderboard probing and treat public feedback as a noisy split."
        ),
        "artifacts": {
            "test_predictions": str(OUTPUT_TEST_PREDICTION.relative_to(ROOT)),
            "submission": str(OUTPUT_SUBMISSION.relative_to(ROOT)),
            "run_summary": str(OUTPUT_LOG.relative_to(ROOT)),
            "model": str(OUTPUT_MODEL.relative_to(ROOT)),
        },
    }
    with OUTPUT_LOG.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved test predictions: {OUTPUT_TEST_PREDICTION.relative_to(ROOT)}")
    print(f"Saved submission: {OUTPUT_SUBMISSION.relative_to(ROOT)}")
    print(f"Saved run summary: {OUTPUT_LOG.relative_to(ROOT)}")
    print(f"Saved model artifact: {OUTPUT_MODEL.relative_to(ROOT)}")
    print("Submission validation passed.")
    print("Capacity clipping counts:")
    print(json.dumps(summary["capacity_clipping_counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

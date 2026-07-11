from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dacon_wind.data import load_train_data
from dacon_wind.features import (
    build_baseline_train_frame,
    build_feature_matrix,
    build_targeted_weather_feature_matrix,
)
from dacon_wind.metric import CAPACITY_KWH, TARGET_COLS, calculate_metric


EXP_ID = "cv_001_rolling_scale_robustness_cv"
RUN_ID = "cv_001_rolling_scale_robustness_cv"
RANDOM_SEED = 42
VALIDATION_YEARS = (2022, 2023, 2024)
MIN_TRAIN_ROWS_PER_TARGET = 24
STD_HIGH_THRESHOLD = 0.01

BEST_PARAMS: dict[str, Any] = {
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
    "random_state": RANDOM_SEED,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": -1,
    "verbosity": -1,
}

REFERENCE_PUBLIC = {
    "lgbm_008_scale_110_submit": {
        "public_total_score": 0.6211026154,
        "public_one_minus_nmae": 0.8634833072,
        "public_ficr": 0.3787219236,
    },
    "ens_004_group_scale_submit": {
        "submitted_at_kst": "2026-07-11 16:04:53",
        "dacon_submission_title": "ens_004_group_scale edit",
        "public_total_score": 0.6128355459,
        "public_one_minus_nmae": 0.8625281665,
        "public_ficr": 0.3631429254,
    },
}

CANDIDATE_SCALE_SETS: list[dict[str, Any]] = [
    {
        "candidate_id": "raw_no_scale",
        "category": "baseline_global",
        "scales": {target: 1.00 for target in TARGET_COLS},
    },
    *[
        {
            "candidate_id": f"global_{str(scale).replace('.', 'p')}",
            "category": "baseline_global",
            "scales": {target: scale for target in TARGET_COLS},
        }
        for scale in (1.03, 1.08, 1.095, 1.10, 1.105, 1.12)
    ],
    {
        "candidate_id": "ens_004_failed_group_scale_1p110_1p020_1p130",
        "category": "previous_failed_group_scale",
        "scales": {
            "kpx_group_1": 1.110,
            "kpx_group_2": 1.020,
            "kpx_group_3": 1.130,
        },
    },
    *[
        {
            "candidate_id": f"group_{g1:.3f}_{g2:.3f}_{g3:.3f}".replace(".", "p"),
            "category": "conservative_group_scale",
            "scales": {
                "kpx_group_1": g1,
                "kpx_group_2": g2,
                "kpx_group_3": g3,
            },
        }
        for g1, g2, g3 in (
            (1.08, 1.08, 1.08),
            (1.095, 1.095, 1.095),
            (1.10, 1.10, 1.10),
            (1.105, 1.105, 1.105),
            (1.10, 1.06, 1.12),
            (1.08, 1.06, 1.10),
            (1.10, 1.08, 1.10),
            (1.105, 1.08, 1.115),
        )
    ],
]


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
            "Remove or rename them before rerunning this validation-only script."
        )


def build_model_predictions(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
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
    train_rows: dict[str, int] = {}
    for target in TARGET_COLS:
        train_mask = y_train[target].notna()
        train_rows[target] = int(train_mask.sum())
        model = LGBMRegressor(**BEST_PARAMS)
        model.fit(x_train_imp.loc[train_mask], y_train.loc[train_mask, target])
        predictions[target] = np.clip(
            model.predict(x_valid_imp),
            0.0,
            CAPACITY_KWH[target],
        )

    return predictions, train_rows, list(x_train.columns)


def candidate_prediction(
    raw_prediction: pd.DataFrame,
    scales: dict[str, float],
) -> pd.DataFrame:
    prediction = pd.DataFrame(index=raw_prediction.index)
    for target in TARGET_COLS:
        prediction[target] = (raw_prediction[target] * scales[target]).clip(
            lower=0.0,
            upper=CAPACITY_KWH[target],
        )
    return prediction


def score_prediction(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, Any]:
    return calculate_metric(
        y_true.loc[:, list(TARGET_COLS)].reset_index(drop=True),
        y_pred.loc[:, list(TARGET_COLS)].reset_index(drop=True),
    )


def skip_reason_for_fold(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    validation_year: int,
) -> str | None:
    if valid_frame.empty:
        return f"no validation rows for forecast year {validation_year}"
    if train_frame.empty:
        return f"no training rows with forecast year < {validation_year}"

    train_counts = {target: int(train_frame[target].notna().sum()) for target in TARGET_COLS}
    valid_counts = {target: int(valid_frame[target].notna().sum()) for target in TARGET_COLS}
    low_train = {
        target: count
        for target, count in train_counts.items()
        if count < MIN_TRAIN_ROWS_PER_TARGET
    }
    low_valid = {target: count for target, count in valid_counts.items() if count == 0}
    if low_train:
        return (
            f"too few prior training rows for target(s) before {validation_year}: "
            f"{low_train}"
        )
    if low_valid:
        return f"no validation labels for target(s) in {validation_year}: {low_valid}"
    return None


def evaluate_fold(
    frame: pd.DataFrame,
    baseline_features: pd.DataFrame,
    targeted_features: pd.DataFrame,
    validation_year: int,
) -> tuple[dict[str, Any], pd.DataFrame | None, list[dict[str, Any]]]:
    years = pd.to_datetime(frame["forecast_kst_dtm"]).dt.year
    train_mask = years < validation_year
    valid_mask = years == validation_year

    train_frame = frame.loc[train_mask].copy()
    valid_frame = frame.loc[valid_mask].copy()
    train_counts = {target: int(train_frame[target].notna().sum()) for target in TARGET_COLS}
    valid_counts = {target: int(valid_frame[target].notna().sum()) for target in TARGET_COLS}

    reason = skip_reason_for_fold(train_frame, valid_frame, validation_year)
    fold_info: dict[str, Any] = {
        "validation_year": validation_year,
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "target_train_rows": train_counts,
        "target_valid_rows": valid_counts,
        "status": "skipped" if reason else "evaluated",
        "skip_reason": reason,
    }
    if reason:
        return fold_info, None, []

    x_train_003 = baseline_features.loc[train_mask].reset_index(drop=True)
    x_valid_003 = baseline_features.loc[valid_mask].reset_index(drop=True)
    x_train_005 = targeted_features.loc[train_mask].reset_index(drop=True)
    x_valid_005 = targeted_features.loc[valid_mask].reset_index(drop=True)
    y_train = train_frame.loc[:, list(TARGET_COLS)].reset_index(drop=True)
    y_true = valid_frame.loc[:, list(TARGET_COLS)].reset_index(drop=True)

    pred_003, train_rows_003, feature_cols_003 = build_model_predictions(
        x_train_003,
        y_train,
        x_valid_003.reindex(columns=x_train_003.columns),
    )
    pred_005, train_rows_005, feature_cols_005 = build_model_predictions(
        x_train_005,
        y_train,
        x_valid_005.reindex(columns=x_train_005.columns),
    )
    raw_prediction = (0.5 * pred_003) + (0.5 * pred_005)
    raw_prediction = raw_prediction.clip(
        lower=0.0,
        upper=pd.Series(CAPACITY_KWH),
        axis=1,
    )

    raw_scores = score_prediction(y_true, raw_prediction)
    fold_info.update(
        {
            "raw_score": raw_scores,
            "model_sources": {
                "lgbm_003_tuned_baseline": {
                    "target_train_rows": train_rows_003,
                    "feature_count": len(feature_cols_003),
                },
                "lgbm_005_targeted_weather": {
                    "target_train_rows": train_rows_005,
                    "feature_count": len(feature_cols_005),
                    "reused_existing_feature_function": True,
                },
                "raw_ensemble": "0.5*lgbm_003_tuned_baseline + 0.5*lgbm_005_targeted_weather",
            },
            "forecast_start": str(valid_frame["forecast_kst_dtm"].min()),
            "forecast_end": str(valid_frame["forecast_kst_dtm"].max()),
        }
    )

    oof = valid_frame.loc[:, ["forecast_kst_dtm", *TARGET_COLS]].copy()
    oof.insert(0, "validation_year", validation_year)
    for target in TARGET_COLS:
        oof[f"lgbm_003_pred_{target}"] = pred_003[target].to_numpy(dtype=float)
        oof[f"lgbm_005_pred_{target}"] = pred_005[target].to_numpy(dtype=float)
        oof[f"raw_pred_{target}"] = raw_prediction[target].to_numpy(dtype=float)

    candidate_results: list[dict[str, Any]] = []
    for candidate in CANDIDATE_SCALE_SETS:
        scaled = candidate_prediction(raw_prediction, candidate["scales"])
        scores = score_prediction(y_true, scaled)
        result = {
            "candidate_id": candidate["candidate_id"],
            "category": candidate["category"],
            "validation_year": validation_year,
            "scales": candidate["scales"],
            "total_score": scores["total_score"],
            "one_minus_nmae": scores["one_minus_nmae"],
            "ficr": scores["ficr"],
            "group_nmae": scores["group_nmae"],
            "group_ficr": scores["group_ficr"],
        }
        candidate_results.append(result)
        for target in TARGET_COLS:
            oof[f"{candidate['candidate_id']}_pred_{target}"] = scaled[target].to_numpy(
                dtype=float
            )

    return fold_info, oof, candidate_results


def aggregate_candidate_scores(
    fold_candidate_scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATE_SCALE_SETS:
        candidate_id = candidate["candidate_id"]
        scores = [
            item
            for item in fold_candidate_scores
            if item["candidate_id"] == candidate_id
        ]
        totals = [float(item["total_score"]) for item in scores]
        one_minus_nmae = [float(item["one_minus_nmae"]) for item in scores]
        ficr = [float(item["ficr"]) for item in scores]
        rows.append(
            {
                "candidate_id": candidate_id,
                "category": candidate["category"],
                "scales": candidate["scales"],
                "evaluated_years": [int(item["validation_year"]) for item in scores],
                "mean_total_score": float(np.mean(totals)) if totals else None,
                "min_total_score": float(np.min(totals)) if totals else None,
                "std_total_score": float(np.std(totals, ddof=0)) if totals else None,
                "mean_one_minus_nmae": float(np.mean(one_minus_nmae))
                if one_minus_nmae
                else None,
                "min_one_minus_nmae": float(np.min(one_minus_nmae))
                if one_minus_nmae
                else None,
                "mean_ficr": float(np.mean(ficr)) if ficr else None,
                "min_ficr": float(np.min(ficr)) if ficr else None,
                "per_year_scores": {
                    str(item["validation_year"]): {
                        "total_score": item["total_score"],
                        "one_minus_nmae": item["one_minus_nmae"],
                        "ficr": item["ficr"],
                    }
                    for item in scores
                },
                "high_std_total_score": (
                    bool(np.std(totals, ddof=0) >= STD_HIGH_THRESHOLD)
                    if len(totals) >= 2
                    else False
                ),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            item["mean_total_score"] is not None,
            item["mean_total_score"] if item["mean_total_score"] is not None else -np.inf,
            item["min_total_score"] if item["min_total_score"] is not None else -np.inf,
        ),
        reverse=True,
    )


def compare_candidates(
    selected: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {row["candidate_id"]: row for row in aggregate_rows}
    references = {
        "global_1p10": by_id["global_1p1"],
        "ens_004_failed_group_scale": by_id[
            "ens_004_failed_group_scale_1p110_1p020_1p130"
        ],
        "raw_no_scale": by_id["raw_no_scale"],
    }
    comparisons = {}
    for name, reference in references.items():
        comparisons[name] = {
            "reference_candidate_id": reference["candidate_id"],
            "delta_mean_total_score": (
                selected["mean_total_score"] - reference["mean_total_score"]
                if selected["mean_total_score"] is not None
                and reference["mean_total_score"] is not None
                else None
            ),
            "delta_min_total_score": (
                selected["min_total_score"] - reference["min_total_score"]
                if selected["min_total_score"] is not None
                and reference["min_total_score"] is not None
                else None
            ),
            "delta_mean_one_minus_nmae": (
                selected["mean_one_minus_nmae"] - reference["mean_one_minus_nmae"]
                if selected["mean_one_minus_nmae"] is not None
                and reference["mean_one_minus_nmae"] is not None
                else None
            ),
            "delta_mean_ficr": (
                selected["mean_ficr"] - reference["mean_ficr"]
                if selected["mean_ficr"] is not None and reference["mean_ficr"] is not None
                else None
            ),
        }
    return comparisons


def final_decision(
    selected: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    evaluated_years: list[int],
) -> tuple[str, bool, str]:
    by_id = {row["candidate_id"]: row for row in aggregate_rows}
    global_110 = by_id["global_1p1"]
    failed_group = by_id["ens_004_failed_group_scale_1p110_1p020_1p130"]
    enough_years = len(evaluated_years) >= 2

    failed_unstable = (
        failed_group["std_total_score"] is not None
        and failed_group["std_total_score"] >= STD_HIGH_THRESHOLD
    )
    if len(failed_group["evaluated_years"]) < 2:
        stability_text = (
            "Cannot determine across years because only "
            f"{failed_group['evaluated_years']} was evaluable under the "
            "strict rolling-year rule."
        )
    elif failed_unstable:
        stability_text = "Unstable across evaluated years."
    else:
        stability_text = "Not unstable by the configured std threshold."

    beats_global_mean = selected["mean_total_score"] > global_110["mean_total_score"]
    beats_global_min = selected["min_total_score"] > global_110["min_total_score"]
    if enough_years and beats_global_mean and beats_global_min:
        return (
            "robust future submission candidate",
            failed_unstable,
            stability_text,
        )
    if not enough_years:
        return (
            "insufficient rolling-year evidence; keep lgbm_008_scale_110_submit as current best public reference and shift next work to feature/model improvements",
            failed_unstable,
            stability_text,
        )
    return (
        "no candidate robustly beats global 1.10; keep lgbm_008_scale_110_submit as current best public reference and shift next work to feature/model improvements",
        failed_unstable,
        stability_text,
    )


def print_summary(
    fold_summaries: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    selected: dict[str, Any],
    comparisons: dict[str, Any],
    failed_group_stability: str,
    decision: str,
) -> None:
    print("per_year_raw_scores")
    print("year,status,total_score,one_minus_nmae,ficr,skip_reason")
    for fold in fold_summaries:
        raw = fold.get("raw_score") or {}
        print(
            f"{fold['validation_year']},"
            f"{fold['status']},"
            f"{raw.get('total_score', '')},"
            f"{raw.get('one_minus_nmae', '')},"
            f"{raw.get('ficr', '')},"
            f"{fold.get('skip_reason') or ''}"
        )

    print("candidate_aggregate_sorted_by_mean_total_score")
    print(
        "candidate_id,mean_total_score,min_total_score,std_total_score,"
        "mean_one_minus_nmae,min_one_minus_nmae,mean_ficr,min_ficr,evaluated_years"
    )
    for row in aggregate_rows:
        print(
            f"{row['candidate_id']},"
            f"{row['mean_total_score'] if row['mean_total_score'] is not None else ''},"
            f"{row['min_total_score'] if row['min_total_score'] is not None else ''},"
            f"{row['std_total_score'] if row['std_total_score'] is not None else ''},"
            f"{row['mean_one_minus_nmae'] if row['mean_one_minus_nmae'] is not None else ''},"
            f"{row['min_one_minus_nmae'] if row['min_one_minus_nmae'] is not None else ''},"
            f"{row['mean_ficr'] if row['mean_ficr'] is not None else ''},"
            f"{row['min_ficr'] if row['min_ficr'] is not None else ''},"
            f"{row['evaluated_years']}"
        )

    print("selected_robust_candidate")
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    print("selected_candidate_comparisons")
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))
    print(f"ens_004_failed_group_scale_stability: {failed_group_stability}")
    print(f"final_decision: {decision}")


def main() -> None:
    set_reproducible_seed(RANDOM_SEED)

    prediction_dir = ROOT / "outputs" / "predictions"
    log_dir = ROOT / "outputs" / "logs"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    oof_path = prediction_dir / f"{RUN_ID}_oof.csv"
    log_path = log_dir / f"{RUN_ID}.json"
    ensure_new_artifacts([oof_path, log_path])

    data = load_train_data(ROOT / "data" / "raw" / "train")
    frame = build_baseline_train_frame(data.labels, data.ldaps, data.gfs)
    frame = frame.sort_values("forecast_kst_dtm").reset_index(drop=True)
    baseline_features = build_feature_matrix(frame)
    targeted_features = build_targeted_weather_feature_matrix(frame)

    fold_summaries: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []
    fold_candidate_scores: list[dict[str, Any]] = []
    for validation_year in VALIDATION_YEARS:
        fold_info, oof, candidate_scores = evaluate_fold(
            frame,
            baseline_features,
            targeted_features,
            validation_year,
        )
        fold_summaries.append(fold_info)
        if oof is not None:
            oof_frames.append(oof)
        fold_candidate_scores.extend(candidate_scores)

    if not oof_frames:
        raise RuntimeError("No rolling-year folds were evaluable; no OOF artifact created.")

    oof_output = pd.concat(oof_frames, ignore_index=True)
    oof_output["forecast_kst_dtm"] = pd.to_datetime(
        oof_output["forecast_kst_dtm"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    oof_output.to_csv(oof_path, index=False, encoding="utf-8-sig")

    aggregate_rows = aggregate_candidate_scores(fold_candidate_scores)
    selected = aggregate_rows[0]
    evaluated_years = sorted(
        {
            int(score["validation_year"])
            for score in fold_candidate_scores
            if score["candidate_id"] == selected["candidate_id"]
        }
    )
    comparisons = compare_candidates(selected, aggregate_rows)
    decision, failed_unstable, failed_group_stability = final_decision(
        selected,
        aggregate_rows,
        evaluated_years,
    )

    summary = {
        "exp_id": EXP_ID,
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Validation-only rolling-year robustness check for global and group-wise "
            "scaling candidates using training data only."
        ),
        "random_seed": RANDOM_SEED,
        "data": {
            "train_labels": "data/raw/train/train_labels.csv",
            "ldaps_train": "data/raw/train/ldaps_train.csv",
            "gfs_train": "data/raw/train/gfs_train.csv",
        },
        "validation_years": list(VALIDATION_YEARS),
        "fold_rule": "train on forecast year < validation_year; validate on forecast year == validation_year",
        "skip_rule": {
            "min_train_rows_per_target": MIN_TRAIN_ROWS_PER_TARGET,
            "reason": (
                "Folds with no validation rows, no prior train rows, or too few "
                "prior rows for any target are skipped and recorded."
            ),
        },
        "modeling": {
            "lgbm_003_source": "baseline calendar + LDAPS/GFS mean features",
            "lgbm_005_source": (
                "targeted-weather feature function reused from src/dacon_wind/features.py"
            ),
            "raw_ensemble": "0.5*lgbm_003 prediction + 0.5*lgbm_005 prediction",
            "params": BEST_PARAMS,
            "prediction_clipping": "Every raw and scaled prediction is clipped to [0, group capacity].",
        },
        "candidate_scale_sets": CANDIDATE_SCALE_SETS,
        "fold_summaries": fold_summaries,
        "candidate_scores_by_fold": fold_candidate_scores,
        "candidate_aggregate_sorted_by_mean_total_score": aggregate_rows,
        "selected_robust_candidate_by_rule": selected,
        "comparisons": comparisons,
        "ens_004_failed_group_scale_unstable": failed_unstable,
        "ens_004_failed_group_scale_stability": failed_group_stability,
        "decision_rule": (
            "Primary highest mean_total_score; tie-breaker higher min_total_score; "
            "flag high std_total_score."
        ),
        "final_decision": decision,
        "public_reference": REFERENCE_PUBLIC,
        "artifacts": {
            "oof_predictions": str(oof_path.relative_to(ROOT)),
            "run_summary": str(log_path.relative_to(ROOT)),
        },
    }

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print_summary(
        fold_summaries,
        aggregate_rows,
        selected,
        comparisons,
        failed_group_stability,
        decision,
    )
    print(f"Saved OOF predictions: {oof_path.relative_to(ROOT)}")
    print(f"Saved run summary: {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

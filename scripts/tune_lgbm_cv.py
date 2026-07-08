from __future__ import annotations

import itertools
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
from dacon_wind.models import LGBMResult, lgbm_v1_params


EXP_ID = "lgbm_003_tuned"
RUN_ID = "lgbm_003_tuned_valid_2024"
RANDOM_SEED = 42

GRID = {
    "num_leaves": [15, 31, 63],
    "min_child_samples": [20, 40, 80],
    "learning_rate": [0.02, 0.03],
    "n_estimators": [600, 1000],
    "reg_lambda": [1.0, 5.0],
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
            "Remove or rename them before rerunning this tuning script."
        )


def iter_param_grid() -> list[dict[str, Any]]:
    keys = list(GRID)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(GRID[key] for key in keys))
    ]


def tuned_lgbm_params(overrides: dict[str, Any], random_state: int) -> dict[str, Any]:
    params = lgbm_v1_params(random_state=random_state)
    params.update(overrides)
    return params


def train_lgbm_models(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    params: dict[str, Any],
) -> LGBMResult:
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

    return LGBMResult(
        imputer=imputer,
        models=models,
        feature_columns=list(x_train.columns),
        predictions=predictions,
        train_rows=train_rows,
    )


def flatten_result_row(
    grid_index: int,
    params: dict[str, Any],
    scores: dict[str, Any],
) -> dict[str, Any]:
    row = {"grid_index": grid_index, **params}
    row.update(
        {
            "total_score": scores["total_score"],
            "one_minus_nmae": scores["one_minus_nmae"],
            "ficr": scores["ficr"],
        }
    )
    for target in TARGET_COLS:
        row[f"{target}_nmae"] = scores["group_nmae"][target]
        row[f"{target}_ficr"] = scores["group_ficr"][target]
    return row


def main() -> None:
    set_reproducible_seed(RANDOM_SEED)

    prediction_dir = ROOT / "outputs" / "predictions"
    log_dir = ROOT / "outputs" / "logs"
    model_dir = ROOT / "outputs" / "models"
    for path in (prediction_dir, log_dir, model_dir):
        path.mkdir(parents=True, exist_ok=True)

    tuning_results_path = log_dir / "lgbm_003_tuning_results.csv"
    prediction_path = prediction_dir / f"{RUN_ID}.csv"
    log_path = log_dir / f"{RUN_ID}.json"
    model_path = model_dir / f"{RUN_ID}.joblib"
    ensure_new_artifacts([tuning_results_path, prediction_path, log_path, model_path])

    data = load_train_data(ROOT / "data" / "raw" / "train")
    train_frame = build_baseline_train_frame(data.labels, data.ldaps, data.gfs)
    features = build_feature_matrix(train_frame)

    train_mask, valid_mask = split_train_valid_2024(train_frame["forecast_kst_dtm"])
    x_train = features.loc[train_mask]
    y_train = train_frame.loc[train_mask, list(TARGET_COLS)]
    x_valid = features.loc[valid_mask]
    y_valid = train_frame.loc[valid_mask, list(TARGET_COLS)]

    best_result: LGBMResult | None = None
    best_scores: dict[str, Any] | None = None
    best_params: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    param_grid = iter_param_grid()

    for grid_index, grid_params in enumerate(param_grid, start=1):
        model_params = tuned_lgbm_params(grid_params, random_state=RANDOM_SEED)
        result = train_lgbm_models(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            params=model_params,
        )
        scores = calculate_metric(
            y_valid.reset_index(drop=True),
            result.predictions.reset_index(drop=True),
        )
        rows.append(flatten_result_row(grid_index, grid_params, scores))

        if best_scores is None or scores["total_score"] > best_scores["total_score"]:
            best_result = result
            best_scores = scores
            best_params = model_params

        print(
            f"[{grid_index:02d}/{len(param_grid):02d}] "
            f"total_score={scores['total_score']:.10f} params={grid_params}"
        )

    if best_result is None or best_scores is None or best_params is None:
        raise RuntimeError("No tuning results were produced.")

    results = pd.DataFrame(rows).sort_values(
        ["total_score", "one_minus_nmae", "ficr"],
        ascending=[False, False, False],
    )
    results.to_csv(tuning_results_path, index=False, encoding="utf-8-sig")

    valid_predictions = train_frame.loc[valid_mask, ["forecast_kst_dtm"]].copy()
    for target in TARGET_COLS:
        valid_predictions[target] = best_result.predictions[target].to_numpy()
    valid_predictions["forecast_kst_dtm"] = pd.to_datetime(
        valid_predictions["forecast_kst_dtm"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    valid_predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    summary = {
        "exp_id": EXP_ID,
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "data": {
            "train_labels": "data/raw/train/train_labels.csv",
            "ldaps_train": "data/raw/train/ldaps_train.csv",
            "gfs_train": "data/raw/train/gfs_train.csv",
        },
        "split": {
            "train": "year < 2024",
            "valid": "year == 2024",
            "train_rows": int(train_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
            "target_train_rows": best_result.train_rows,
        },
        "features": {
            "approach": "Baseline calendar features plus mean-aggregated LDAPS/GFS by forecast_kst_dtm.",
            "feature_count": len(best_result.feature_columns),
            "feature_columns": best_result.feature_columns,
        },
        "model": {
            "class": "lightgbm.LGBMRegressor",
            "per_target_models": list(TARGET_COLS),
            "base_params": lgbm_v1_params(random_state=RANDOM_SEED),
            "grid": GRID,
            "best_params": best_params,
        },
        "artifacts": {
            "tuning_results": str(tuning_results_path.relative_to(ROOT)),
            "valid_predictions": str(prediction_path.relative_to(ROOT)),
            "metric_summary": str(log_path.relative_to(ROOT)),
            "model": str(model_path.relative_to(ROOT)),
        },
        "metrics": best_scores,
    }

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    joblib.dump(
        {
            "exp_id": EXP_ID,
            "run_id": RUN_ID,
            "random_seed": RANDOM_SEED,
            "imputer": best_result.imputer,
            "models": best_result.models,
            "feature_columns": best_result.feature_columns,
            "model_params": best_params,
            "grid": GRID,
            "metrics": best_scores,
        },
        model_path,
    )

    print(f"best total_score: {best_scores['total_score']:.10f}")
    print(f"best one_minus_nmae: {best_scores['one_minus_nmae']:.10f}")
    print(f"best ficr: {best_scores['ficr']:.10f}")
    print(f"best group_nmae: {json.dumps(best_scores['group_nmae'], sort_keys=True)}")
    print(f"best group_ficr: {json.dumps(best_scores['group_ficr'], sort_keys=True)}")
    print(f"best params: {json.dumps(best_params, sort_keys=True)}")
    print(f"Saved tuning results: {tuning_results_path.relative_to(ROOT)}")
    print(f"Saved validation predictions: {prediction_path.relative_to(ROOT)}")
    print(f"Saved metric summary: {log_path.relative_to(ROOT)}")
    print(f"Saved model artifact: {model_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from dacon_wind.metric import CAPACITY_KWH, TARGET_COLS


@dataclass(frozen=True)
class BaselineRFResult:
    imputer: SimpleImputer
    models: dict[str, RandomForestRegressor]
    feature_columns: list[str]
    predictions: pd.DataFrame
    train_rows: dict[str, int]


@dataclass(frozen=True)
class LGBMResult:
    imputer: SimpleImputer
    models: dict[str, LGBMRegressor]
    feature_columns: list[str]
    predictions: pd.DataFrame
    train_rows: dict[str, int]


def baseline_rf_params(random_state: int = 42) -> dict[str, Any]:
    return {
        "n_estimators": 120,
        "max_depth": 14,
        "min_samples_leaf": 8,
        "max_features": "sqrt",
        "random_state": random_state,
        "n_jobs": -1,
    }


def train_baseline_rf_models(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    random_state: int = 42,
) -> BaselineRFResult:
    """Train one baseline RandomForestRegressor per target group."""
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
    models: dict[str, RandomForestRegressor] = {}
    train_rows: dict[str, int] = {}

    for target in TARGET_COLS:
        train_mask = y_train[target].notna()
        if not train_mask.any():
            raise ValueError(f"No non-null training labels for {target}.")

        model = RandomForestRegressor(**baseline_rf_params(random_state=random_state))
        model.fit(x_train_imp.loc[train_mask], y_train.loc[train_mask, target])

        pred = model.predict(x_valid_imp)
        predictions[target] = np.clip(pred, 0.0, CAPACITY_KWH[target])
        models[target] = model
        train_rows[target] = int(train_mask.sum())

    return BaselineRFResult(
        imputer=imputer,
        models=models,
        feature_columns=list(x_train.columns),
        predictions=predictions,
        train_rows=train_rows,
    )


def lgbm_v1_params(random_state: int = 42) -> dict[str, Any]:
    return {
        "objective": "regression",
        "n_estimators": 800,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def train_lgbm_v1_models(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    random_state: int = 42,
) -> LGBMResult:
    """Train one LightGBM v1 regressor per target group."""
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

        model = LGBMRegressor(**lgbm_v1_params(random_state=random_state))
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

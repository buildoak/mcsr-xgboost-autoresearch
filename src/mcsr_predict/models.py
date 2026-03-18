"""Model training utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from xgboost import XGBClassifier

from .config import RANDOM_STATE


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> XGBClassifier:
    """Train an XGBoost classifier with early stopping on the time-split holdout."""

    model = XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        eval_metric="auc",
        early_stopping_rounds=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    return model

"""Evaluation helpers for trained models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_metrics(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute core binary classification metrics."""

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "prediction_mean": float(np.mean(y_prob)),
        "prediction_std": float(np.std(y_prob)),
    }


def get_feature_importance(model: Any, feature_names: list[str]) -> list[tuple[str, float]]:
    """Return feature importances sorted descending."""

    importance_values = getattr(model, "feature_importances_", None)
    if importance_values is None:
        return []

    pairs = [(name, float(score)) for name, score in zip(feature_names, importance_values)]
    return sorted(pairs, key=lambda item: item[1], reverse=True)


def print_evaluation_report(
    metrics: dict[str, float], feature_importance: list[tuple[str, float]], top_n: int = 20
) -> None:
    """Print a concise evaluation summary."""

    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Prediction distribution: mean={metrics['prediction_mean']:.4f}, std={metrics['prediction_std']:.4f}")
    print(f"Top {top_n} features by importance:")
    for name, score in feature_importance[:top_n]:
        print(f"  {name}: {score:.6f}")

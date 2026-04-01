"""End-to-end pipeline for MCSR ranked match outcome prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import TEST_FRACTION, VALIDATION_FRACTION
from .evaluate import compute_metrics, get_feature_importance, print_evaluation_report
from .features import FeatureBuilder
from .models import train_xgboost


def load_matches(path: Path) -> list[dict[str, Any]]:
    """Load jsonl matches from disk."""

    matches: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            matches.append(json.loads(line))
    return matches


def filter_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop decayed matches and records without seed metadata."""

    return [m for m in matches if not m.get("decayed", False) and m.get("seed") is not None]


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    validation_fraction: float = VALIDATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Split chronologically into train/validation/test partitions."""

    if validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("Validation and test fractions must be positive.")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("Validation plus test fractions must leave a non-empty training split.")

    n_rows = len(X)
    train_end = int(n_rows * (1.0 - validation_fraction - test_fraction))
    val_end = int(n_rows * (1.0 - test_fraction))
    if train_end <= 0 or val_end <= train_end or val_end >= n_rows:
        raise ValueError("Dataset is too small for the requested temporal split.")

    return {
        "X_train": X.iloc[:train_end],
        "y_train": y.iloc[:train_end],
        "X_val": X.iloc[train_end:val_end],
        "y_val": y.iloc[train_end:val_end],
        "X_test": X.iloc[val_end:],
        "y_test": y.iloc[val_end:],
    }


def _evaluate_naive_higher_elo(X_test: pd.DataFrame, y_test: pd.Series) -> float:
    predictions = (X_test["elo_diff_platform"] >= 0.0).astype(int)
    return float((predictions == y_test).mean())


def _train_subset_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    model = train_xgboost(X_train, y_train, X_val, y_val)
    metrics = compute_metrics(model, X_test, y_test)
    return {"model": model, "metrics": metrics}


def run_pipeline(data_path: Path | None = None) -> dict[str, Any]:
    """Train and evaluate the XGBoost predictor on honest temporal splits."""

    if data_path is None:
        data_path = Path(__file__).resolve().parents[2] / "data" / "matches.jsonl"

    all_matches = load_matches(data_path)
    filtered = filter_matches(all_matches)
    filtered.sort(key=lambda m: int(m.get("date", 0)))

    builder = FeatureBuilder()
    X, y = builder.build_dataset(filtered)
    if X.empty:
        raise RuntimeError("No features were generated. Check input filtering and schema.")

    splits = temporal_split(X, y)
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    full_result = _train_subset_model(X_train, y_train, X_val, y_val, X_test, y_test)
    model = full_result["model"]
    metrics = full_result["metrics"]
    feature_importance = get_feature_importance(model, list(X.columns))

    elo_columns = ["elo_diff_platform"]
    elo_result = _train_subset_model(
        X_train[elo_columns],
        y_train,
        X_val[elo_columns],
        y_val,
        X_test[elo_columns],
        y_test,
    )

    no_elo_columns = [column for column in X.columns if "elo" not in column]
    no_elo_result = _train_subset_model(
        X_train[no_elo_columns],
        y_train,
        X_val[no_elo_columns],
        y_val,
        X_test[no_elo_columns],
        y_test,
    )

    naive_accuracy = _evaluate_naive_higher_elo(X_test, y_test)

    print(f"Raw matches loaded: {len(all_matches)}")
    print(f"Matches after filter: {len(filtered)}")
    print(f"Number of matches used: {len(X)}")
    print(f"Skipped unresolved winner: {builder.dataset_stats['skipped_unresolved_winner']}")
    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    print_evaluation_report(metrics, feature_importance, top_n=20)
    print(f"ELO-only ROC-AUC: {elo_result['metrics']['roc_auc']:.4f}")
    print(f"Naive higher-ELO-wins accuracy: {naive_accuracy:.4f}")
    print(f"No-ELO ROC-AUC: {no_elo_result['metrics']['roc_auc']:.4f}")
    print(f"Feature count: {X.shape[1]}")

    return {
        "counts": {
            "raw_matches": len(all_matches),
            "filtered_matches": len(filtered),
            "matches_used": len(X),
            "skipped_unresolved_winner": builder.dataset_stats["skipped_unresolved_winner"],
            "train_size": len(X_train),
            "validation_size": len(X_val),
            "test_size": len(X_test),
            "feature_count": X.shape[1],
        },
        "full_model": metrics,
        "elo_only": elo_result["metrics"],
        "naive_higher_elo_accuracy": naive_accuracy,
        "no_elo": no_elo_result["metrics"],
        "feature_importance": feature_importance[:10],
    }


if __name__ == "__main__":
    run_pipeline()

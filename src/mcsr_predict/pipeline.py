"""End-to-end pipeline for MCSR ranked match outcome prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import TEST_FRACTION
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


def run_pipeline(data_path: Path | None = None) -> None:
    """Train and evaluate the XGBoost predictor on time-split data."""

    if data_path is None:
        data_path = Path(__file__).resolve().parents[2] / "data" / "matches.jsonl"

    all_matches = load_matches(data_path)
    filtered = filter_matches(all_matches)
    filtered.sort(key=lambda m: int(m.get("date", 0)))

    builder = FeatureBuilder()
    X, y = builder.build_dataset(filtered)
    if X.empty:
        raise RuntimeError("No features were generated. Check input filtering and schema.")

    split_idx = int(len(X) * (1.0 - TEST_FRACTION))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = train_xgboost(X_train, y_train, X_test, y_test)

    metrics = compute_metrics(model, X_test, y_test)
    feature_importance = get_feature_importance(model, list(X.columns))

    print(f"Raw matches loaded: {len(all_matches)}")
    print(f"Matches after filter: {len(filtered)}")
    print(f"Number of matches used: {len(X)}")
    print(f"Skipped unresolved winner: {builder.dataset_stats['skipped_unresolved_winner']}")
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print_evaluation_report(metrics, feature_importance, top_n=20)
    print(f"Feature count: {X.shape[1]}")


if __name__ == "__main__":
    run_pipeline()

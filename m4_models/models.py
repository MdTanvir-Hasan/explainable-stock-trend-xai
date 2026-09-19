"""M4 — SP3b baseline and primary models over the walk-forward folds.

Logistic Regression (scaled, class-weighted) and XGBoost are trained on each
fold's training window and scored on the non-overlapping test block. Scaling is
fitted on the training fold only, so no test information reaches preprocessing.

Outputs:
    results/models/<model>_fold<k>.joblib
    results/explanations/predictions.parquet
    results/tables/model_metrics_{per_fold,pooled,per_stock}.csv
    results/tables/baseline_metrics.csv
    results/run_metadata.json
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from m0_setup.config import load_config
from m3_walkforward.validation import walk_forward_folds
from m4_models.evaluation import classification_metrics


def build_models(config: dict) -> dict:
    """Fresh estimators per fold (fitting must not carry state between folds)."""
    seed = config["seed"]
    logistic = config["models"]["logistic"]
    xgb = config["models"]["xgboost"]

    return {
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=logistic["C"],
                        max_iter=logistic["max_iter"],
                        class_weight=logistic["class_weight"],
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "xgboost": XGBClassifier(
            n_estimators=xgb["n_estimators"],
            max_depth=xgb["max_depth"],
            learning_rate=xgb["learning_rate"],
            subsample=xgb["subsample"],
            colsample_bytree=xgb["colsample_bytree"],
            reg_lambda=xgb["reg_lambda"],
            eval_metric=xgb["eval_metric"],
            random_state=seed,
            n_jobs=1,  # single-threaded: hist training is otherwise non-deterministic
            tree_method="hist",
        ),
    }


def _fold_rows(table: pd.DataFrame, dates) -> pd.DataFrame:
    return table[table["date"].isin(pd.DatetimeIndex(dates))]


def baseline_metrics(table: pd.DataFrame, folds, config: dict) -> pd.DataFrame:
    """Majority-class and day-to-day persistence baselines."""
    majority_true, majority_pred = [], []
    persistence_true, persistence_score = [], []

    for fold in folds:
        train = _fold_rows(table, fold.train)
        test = _fold_rows(table, fold.test)
        majority = int(train["target"].mean() >= 0.5)

        majority_true.append(test["target"].to_numpy())
        majority_pred.append(np.full(len(test), majority))
        persistence_true.append(test["target"].to_numpy())
        persistence_score.append(test["ret_1"].to_numpy())

    records = [
        {"model": "majority_class", **classification_metrics(np.concatenate(majority_true), np.concatenate(majority_pred))},
        {
            "model": "persistence",
            **classification_metrics(np.concatenate(persistence_true), np.concatenate(persistence_score), threshold=0.0),
        },
    ]
    return pd.DataFrame(records)


def write_metadata(config: dict, extra: dict) -> None:
    """Record seeds, config hash, git commit and timestamp for reproducibility."""
    config_file = Path(config["paths"]["results"]).parent / "config.yaml"
    config_hash = hashlib.sha256(config_file.read_bytes()).hexdigest()[:16]

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=config_file.parent,
        ).stdout.strip()
    except Exception:
        git_commit = None

    metadata = {
        "seed": config["seed"],
        "config_hash": config_hash,
        "git_commit": git_commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    (Path(config["paths"]["results"]) / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def run(config: dict) -> pd.DataFrame:
    """Train both models across all folds and write metrics, predictions and models."""
    processed = Path(config["paths"]["processed"])
    table = pd.read_parquet(processed / "model_table.parquet")
    features = json.loads((processed / "feature_columns.json").read_text())["features"]
    folds = walk_forward_folds(table["date"], config)

    models_dir = Path(config["paths"]["models"])
    explanations_dir = Path(config["paths"]["explanations"])
    tables_dir = Path(config["paths"]["tables"])
    for directory in (models_dir, explanations_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    predictions = []
    per_fold = []
    for fold in folds:
        train = _fold_rows(table, fold.train)
        test = _fold_rows(table, fold.test)
        x_train, y_train = train[features], train["target"]
        x_test, y_test = test[features], test["target"]

        for name, model in build_models(config).items():
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_test)[:, 1]
            joblib.dump(model, models_dir / f"{name}_fold{fold.index}.joblib")

            frame = test[["date", "stock"]].copy()
            frame["fold"] = fold.index
            frame["model"] = name
            frame["y_true"] = y_test.to_numpy()
            frame["y_prob"] = probability
            predictions.append(frame)

            per_fold.append(
                {"model": name, "fold": fold.index, **classification_metrics(y_test, probability)}
            )

    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_parquet(explanations_dir / "predictions.parquet", index=False)

    per_fold = pd.DataFrame(per_fold).sort_values(["model", "fold"])
    per_fold.to_csv(tables_dir / "model_metrics_per_fold.csv", index=False)

    pooled = pd.DataFrame(
        [
            {"model": name, **classification_metrics(group["y_true"], group["y_prob"])}
            for name, group in predictions.groupby("model")
        ]
    ).sort_values("model")
    pooled.to_csv(tables_dir / "model_metrics_pooled.csv", index=False)

    per_stock = pd.DataFrame(
        [
            {"model": name, "stock": stock, **classification_metrics(group["y_true"], group["y_prob"])}
            for (name, stock), group in predictions.groupby(["model", "stock"])
        ]
    ).sort_values(["model", "stock"])
    per_stock.to_csv(tables_dir / "model_metrics_per_stock.csv", index=False)

    baseline_metrics(table, folds, config).to_csv(
        tables_dir / "baseline_metrics.csv", index=False
    )

    write_metadata(
        config,
        {
            "stage": "M4",
            "n_folds": len(folds),
            "n_train_rows": len(table),
            "n_features": len(features),
        },
    )

    print(pooled.to_string(index=False))
    return predictions


if __name__ == "__main__":
    run(load_config())

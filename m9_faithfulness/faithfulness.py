"""M9 — SP6 faithfulness tests.

Permutation test: shuffling a feature that the explanation calls important should
change predictions more than shuffling one it calls unimportant. The test compares
the mean absolute change in predicted probability for the top-k and bottom-k
features by mean |SHAP|.

Placebo test: a randomised, irrelevant feature is added and the model retrained;
a faithful explainer should assign it near-zero importance.

Both tests are reported as run — a failure is a finding, not a bug.

Outputs:
    results/tables/faithfulness_permutation.csv
    results/tables/faithfulness_permutation_summary.csv
    results/tables/faithfulness_placebo.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from m0_setup.config import load_config
from m2_features.features import feature_columns
from m3_walkforward.validation import walk_forward_folds
from m4_models.models import build_models
from m7_interpretation.interpretation import feature_group


def _fold_rows(table: pd.DataFrame, dates) -> pd.DataFrame:
    return table[table["date"].isin(pd.DatetimeIndex(dates))]


def permutation_test(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Permutation deltas for the most and least important features."""
    processed = Path(config["paths"]["processed"])
    tables_dir = Path(config["paths"]["tables"])
    models_dir = Path(config["paths"]["models"])

    table = pd.read_parquet(processed / "model_table.parquet")
    features = feature_columns(config)
    folds = walk_forward_folds(table["date"], config)
    k = config["faithfulness"]["top_k"]
    importance = pd.read_csv(tables_dir / "shap_global_importance.csv")
    rng = np.random.default_rng(config["seed"])

    rows = []
    for model in ("xgboost", "logistic"):
        ranking = importance[importance["model"] == model].sort_values("rank")
        selected = {
            "top": ranking.head(k)["feature"].tolist(),
            "bottom": ranking.tail(k)["feature"].tolist(),
        }
        for fold in folds:
            test = _fold_rows(table, fold.test)
            fitted = joblib.load(models_dir / f"{model}_fold{fold.index}.joblib")
            x_test = test[features]
            base = fitted.predict_proba(x_test)[:, 1]

            for kind, names in selected.items():
                for feature in names:
                    shuffled = x_test.copy()
                    shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
                    delta = np.abs(fitted.predict_proba(shuffled)[:, 1] - base).mean()
                    rows.append(
                        {
                            "model": model,
                            "fold": fold.index,
                            "feature": feature,
                            "group": feature_group(feature),
                            "kind": kind,
                            "mean_abs_prob_delta": float(delta),
                        }
                    )

    detail = pd.DataFrame(rows)
    detail.to_csv(tables_dir / "faithfulness_permutation.csv", index=False)

    summary = (
        detail.groupby(["model", "kind"])["mean_abs_prob_delta"]
        .mean()
        .unstack("kind")
        .assign(ratio_top_over_bottom=lambda d: d["top"] / d["bottom"])
        .reset_index()
    )
    summary.to_csv(tables_dir / "faithfulness_permutation_summary.csv", index=False)
    return detail, summary


def placebo_test(config: dict) -> pd.DataFrame:
    """Add a random feature, retrain, and rank its attribution among all features."""
    processed = Path(config["paths"]["processed"])
    tables_dir = Path(config["paths"]["tables"])

    table = pd.read_parquet(processed / "model_table.parquet")
    features = feature_columns(config)
    folds = walk_forward_folds(table["date"], config)
    fold = folds[config["faithfulness"]["placebo_fold"]]
    rng = np.random.default_rng(config["seed"])

    train = _fold_rows(table, fold.train)
    test = _fold_rows(table, fold.test)
    y_train = train["target"]

    x_train = train[features].copy()
    x_test = test[features].copy()
    x_train["placebo"] = rng.normal(size=len(train))
    x_test["placebo"] = rng.normal(size=len(test))

    rows = []
    for model, estimator in build_models(config).items():
        estimator.fit(x_train, y_train)
        if model == "xgboost":
            values = np.asarray(shap.TreeExplainer(estimator).shap_values(x_test))
        else:
            scaler = estimator.named_steps["scaler"]
            classifier = estimator.named_steps["clf"]
            values = np.asarray(
                shap.LinearExplainer(classifier, scaler.transform(x_train)).shap_values(
                    scaler.transform(x_test)
                )
            )

        mean_abs = pd.Series(np.abs(values).mean(axis=0), index=x_train.columns)
        ranking = mean_abs.sort_values(ascending=False)
        rows.append(
            {
                "model": model,
                "fold": fold.index,
                "placebo_rank": int(ranking.index.get_loc("placebo")) + 1,
                "n_features": len(ranking),
                "placebo_mean_abs_shap": float(mean_abs["placebo"]),
                "top_feature": ranking.index[0],
                "top_mean_abs_shap": float(ranking.iloc[0]),
            }
        )

    placebo = pd.DataFrame(rows)
    placebo.to_csv(tables_dir / "faithfulness_placebo.csv", index=False)
    return placebo


def run(config: dict) -> dict:
    detail, summary = permutation_test(config)
    placebo = placebo_test(config)
    print(summary.round(4).to_string(index=False))
    print()
    print(placebo.round(4).to_string(index=False))
    return {"permutation_detail": detail, "permutation_summary": summary, "placebo": placebo}


if __name__ == "__main__":
    run(load_config())

"""M6 — SP4 SHAP attribution.

TreeSHAP for XGBoost and LinearSHAP for the scaled logistic pipeline. SHAP values
are stored wide (one column per feature) alongside the prediction identifiers so
later stages can merge them with predictions, regimes and rolling windows.

The LSTM / DeepSHAP path and the LIME cross-check are documented extensions; the
MVP reports the two model families that answer RQ1-RQ4.

Outputs:
    results/explanations/shap_xgboost.parquet
    results/explanations/shap_logistic.parquet
    results/explanations/shap_base_values.csv
    results/tables/shap_global_importance.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from m0_setup.config import load_config
from m3_walkforward.validation import walk_forward_folds


def _shap_values(model, name: str, x_train: pd.DataFrame, x_test: pd.DataFrame):
    """Return (values, base_value) for one fitted model, by family."""
    if name == "xgboost":
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_test)
        base_value = explainer.expected_value
    else:
        scaler = model.named_steps["scaler"]
        classifier = model.named_steps["clf"]
        explainer = shap.LinearExplainer(classifier, scaler.transform(x_train))
        values = explainer.shap_values(scaler.transform(x_test))
        base_value = explainer.expected_value

    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if np.ndim(base_value) > 0:
        base_value = float(np.ravel(base_value)[-1])
    return values, float(base_value)


def run(config: dict) -> dict[str, pd.DataFrame]:
    """Explain a sample of test rows for every fold and model; save attributions."""
    processed = Path(config["paths"]["processed"])
    explanations_dir = Path(config["paths"]["explanations"])
    tables_dir = Path(config["paths"]["tables"])
    models_dir = Path(config["paths"]["models"])
    explanations_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_parquet(processed / "model_table.parquet")
    features = json.loads((processed / "feature_columns.json").read_text())["features"]
    folds = walk_forward_folds(table["date"], config)
    sample_size = config["shap"]["explain_sample_size"]
    rng = np.random.default_rng(config["seed"])

    frames: dict[str, list[pd.DataFrame]] = {"xgboost": [], "logistic": []}
    base_values: list[dict] = []

    for fold in folds:
        train = table[table["date"].isin(pd.DatetimeIndex(fold.train))]
        test = table[table["date"].isin(pd.DatetimeIndex(fold.test))]
        if len(test) > sample_size:
            keep = rng.choice(len(test), size=sample_size, replace=False)
            test = test.iloc[np.sort(keep)]

        x_train, x_test = train[features], test[features]
        for name in ("xgboost", "logistic"):
            model = joblib.load(models_dir / f"{name}_fold{fold.index}.joblib")
            values, base_value = _shap_values(model, name, x_train, x_test)

            frame = test[["date", "stock"]].copy()
            frame["fold"] = fold.index
            for column, index in zip(features, range(len(features))):
                frame[column] = values[:, index]
            frames[name].append(frame)
            base_values.append({"model": name, "fold": fold.index, "base_value": base_value})

        print(f"fold {fold.index}: explained {len(test)} test rows for both models")

    saved = {}
    importance = []
    for name, parts in frames.items():
        frame = pd.concat(parts, ignore_index=True)
        frame.to_parquet(explanations_dir / f"shap_{name}.parquet", index=False)
        saved[name] = frame

        mean_abs = frame[features].abs().mean().sort_values(ascending=False)
        for rank, (feature, value) in enumerate(mean_abs.items(), start=1):
            importance.append(
                {"model": name, "rank": rank, "feature": feature, "mean_abs_shap": float(value)}
            )

    pd.DataFrame(base_values).to_csv(explanations_dir / "shap_base_values.csv", index=False)
    pd.DataFrame(importance).to_csv(tables_dir / "shap_global_importance.csv", index=False)
    return saved


if __name__ == "__main__":
    result = run(load_config())
    for key, frame in result.items():
        print(f"\n{key}: {frame.shape[0]} explanations x {len(frame.columns)} columns")

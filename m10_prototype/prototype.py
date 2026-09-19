"""M10 — SP7 prototype logic, shared by the Streamlit app and the notebook.

Given a stock, this loads the latest fold's XGBoost model, predicts next-day
direction, produces the local SHAP drivers, maps them to financial themes and
reports how consistent that explanation is with the stock's recent reasoning.

Scope: research demonstration, not a trading system. No reliability claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from m2_features.features import feature_columns
from m7_interpretation.interpretation import GROUPS, feature_group


@dataclass
class Prediction:
    stock: str
    date: str
    probability_up: float
    label: str
    consistency: float
    drivers: pd.DataFrame
    group_shares: pd.Series


def _latest_model(models_dir: Path, model: str = "xgboost"):
    candidates = sorted(
        models_dir.glob(f"{model}_fold*.joblib"), key=lambda p: int(p.stem.split("fold")[-1])
    )
    if not candidates:
        raise FileNotFoundError(f"no {model} models found in {models_dir}")
    return joblib.load(candidates[-1])


def explain_stock(stock: str, config: dict) -> Prediction:
    """Predict and explain the most recent available day for one stock."""
    table = pd.read_parquet(Path(config["paths"]["processed"]) / "model_table.parquet")
    features = feature_columns(config)
    history = table[table["stock"] == stock].sort_values("date")
    if history.empty:
        raise ValueError(f"unknown stock: {stock}")

    row = history.iloc[-1]
    x_row = row[features].to_frame().T.astype(float)

    model = _latest_model(Path(config["paths"]["models"]))
    probability = float(model.predict_proba(x_row)[0, 1])

    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(x_row))[0]

    contrib = pd.DataFrame({"feature": features, "shap": values})
    contrib["group"] = contrib["feature"].map(feature_group)
    drivers = contrib.assign(abs=lambda d: d["shap"].abs()).sort_values("abs", ascending=False).head(8)

    window = config["consistency"]["diagnostic_window"]
    recent = history.iloc[:-1].tail(window)[features].to_numpy(dtype=float)
    if len(recent):
        recent_values = np.asarray(explainer.shap_values(pd.DataFrame(recent, columns=features)))
        mean_vector = recent_values.mean(axis=0)
        denominator = np.linalg.norm(values) * np.linalg.norm(mean_vector)
        consistency = float(np.dot(values, mean_vector) / denominator) if denominator > 0 else float("nan")
    else:
        consistency = float("nan")

    signed = pd.Series({group: contrib.loc[contrib["group"] == group, "shap"].sum() for group in GROUPS})
    group_shares = 100 * signed.abs() / signed.abs().sum()

    return Prediction(
        stock=stock,
        date=str(pd.Timestamp(row["date"]).date()),
        probability_up=probability,
        label="UP" if probability >= 0.5 else "DOWN",
        consistency=consistency,
        drivers=drivers[["feature", "group", "shap"]],
        group_shares=group_shares,
    )


def interpretation_text(prediction: Prediction) -> str:
    """Plain-English reading of a prediction, framed as plausibility not causality."""
    top_group = prediction.group_shares.sort_values(ascending=False).index[0].replace("_", " ")
    direction = "upward" if prediction.probability_up >= 0.5 else "downward"
    if np.isnan(prediction.consistency):
        consistency_text = "not yet available"
    elif prediction.consistency > 0.5:
        consistency_text = "in line with recent reasoning"
    else:
        consistency_text = "diverging from recent reasoning"

    return (
        f"The model leans {direction} on {prediction.stock} for the next trading day "
        f"({prediction.probability_up:.1%} probability of an up move, as of {prediction.date}). "
        f"The strongest financial theme behind this call is {top_group}. "
        f"Explanation consistency against the recent window is {prediction.consistency:.2f} "
        f"({consistency_text}). Research demonstration only — this is not trading advice."
    )

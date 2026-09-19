"""M7 — SP5 financial interpretation.

Maps the 29 features onto six financial themes and aggregates SHAP values by
group. Group-level attribution is the signed sum of member SHAP values, so a
group vector preserves the direction of its net contribution — this is what M8
uses to compare feature-level and group-level explanation stability.

Expected signs from finance theory are recorded for the features where the
literature direction is unambiguous; the narrative in the report compares them
with the empirical direction. SHAP values are plausibility signals, not causal
effects.

Outputs:
    results/explanations/group_shap_<model>.parquet
    results/tables/shap_group_importance.csv
    results/tables/shap_direction_of_effect.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from m0_setup.config import load_config
from m2_features.features import feature_columns

GROUPS = (
    "momentum",
    "trend",
    "volatility",
    "market_conditions",
    "trading_activity",
    "economic_conditions",
)

# +1 expected positive, -1 expected negative, assembled from standard finance
# reasoning: momentum persists; volatility, rates and VIX raise uncertainty.
EXPECTED_SIGN = {
    "market_ret": 1,
    "vix": -1,
    "vix_change_5": -1,
    "market_rv_21": -1,
    "rv_21": -1,
    "atr_14": -1,
    "bb_bandwidth": -1,
    "macd": 1,
    "macd_hist": 1,
    "roc_10": 1,
    "rsi_14": 1,
    "close_sma_10": 1,
    "close_sma_20": 1,
    "close_sma_50": 1,
    "close_ema_12": 1,
    "close_ema_26": 1,
    "cash_rate": -1,
    "bond_2y": -1,
    "bond_10y": -1,
    "term_spread": 1,
}


def feature_group(name: str) -> str:
    """Assign one feature to its financial theme."""
    if name in {"cash_rate", "bond_2y", "bond_10y", "term_spread"}:
        return "economic_conditions"
    if name.startswith("volume_"):
        return "trading_activity"
    if name in {"market_ret", "vix"} or name.startswith(("market_rv", "vix_change")):
        return "market_conditions"
    if name == "bb_bandwidth" or name.startswith(("atr_", "rv_")):
        return "volatility"
    if name.startswith(("close_sma", "close_ema")) or name == "bb_pctb":
        return "trend"
    if name.startswith(("ret_", "rsi_", "macd", "roc_")):
        return "momentum"
    raise ValueError(f"unmapped feature: {name}")


def aggregate_groups(shap_frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Signed sum of member SHAP values per financial group."""
    grouped = shap_frame[["date", "stock", "fold"]].copy()
    for group in GROUPS:
        members = [f for f in features if feature_group(f) == group]
        grouped[group] = shap_frame[members].sum(axis=1)
    return grouped


def run(config: dict) -> dict:
    """Build group attributions, group importance and direction-of-effect tables."""
    processed = Path(config["paths"]["processed"])
    explanations_dir = Path(config["paths"]["explanations"])
    tables_dir = Path(config["paths"]["tables"])

    table = pd.read_parquet(processed / "model_table.parquet")
    features = feature_columns(config)

    importance_rows = []
    direction_rows = []
    for model in ("xgboost", "logistic"):
        shap_frame = pd.read_parquet(explanations_dir / f"shap_{model}.parquet")

        grouped = aggregate_groups(shap_frame, features)
        grouped.to_parquet(explanations_dir / f"group_shap_{model}.parquet", index=False)

        total = sum(grouped[group].abs().mean() for group in GROUPS)
        for group in GROUPS:
            mean_abs = float(grouped[group].abs().mean())
            importance_rows.append(
                {
                    "model": model,
                    "group": group,
                    "mean_abs_group_shap": mean_abs,
                    "share_pct": round(100 * mean_abs / total, 2),
                }
            )

        merged = shap_frame.merge(
            table[["date", "stock"] + features], on=["date", "stock"], suffixes=("_shap", "_value")
        )
        for feature in features:
            rho, _ = spearmanr(merged[f"{feature}_value"], merged[f"{feature}_shap"])
            expected = EXPECTED_SIGN.get(feature)
            direction_rows.append(
                {
                    "model": model,
                    "feature": feature,
                    "group": feature_group(feature),
                    "spearman_value_shap": float(rho),
                    "empirical_direction": "positive" if rho >= 0 else "negative",
                    "expected_sign": expected,
                    "matches_theory": (expected is None) or (expected == (1 if rho >= 0 else -1)),
                }
            )

    group_importance = (
        pd.DataFrame(importance_rows)
        .assign(rank=lambda d: d.groupby("model")["mean_abs_group_shap"].rank(ascending=False).astype(int))
        .sort_values(["model", "rank"])
    )
    group_importance.to_csv(tables_dir / "shap_group_importance.csv", index=False)

    direction = pd.DataFrame(direction_rows).sort_values(["model", "group", "feature"])
    direction.to_csv(tables_dir / "shap_direction_of_effect.csv", index=False)

    print(group_importance.to_string(index=False))
    return {"group_importance": group_importance, "direction": direction}


if __name__ == "__main__":
    run(load_config())

"""M8b — SP6b diagnostic value of explanation instability and abstention rules.

For each test prediction the consistency score is the cosine similarity between
its SHAP vector and the mean SHAP vector of the same stock over the preceding
`diagnostic_window` days. The question (RQ4b) is whether lower consistency means
a prediction is less likely to be correct, and whether that can drive an
abstention rule that benchmarks against the confidence-driven baseline.

Outputs:
    results/tables/diagnostic_summary.csv
    results/tables/diagnostic_regression.csv
    results/tables/abstention_risk_coverage.csv
    results/explanations/diagnostic_<model>.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

from m0_setup.config import load_config
from m2_features.features import feature_columns


def consistency_scores(
    shap_frame: pd.DataFrame, features: list[str], window: int, min_periods: int = 5
) -> pd.DataFrame:
    """Cosine similarity between each prediction's SHAP vector and its stock's recent mean."""
    parts = []
    for stock, group in shap_frame.groupby("stock", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        vectors = group[features].to_numpy(dtype=float)
        prior = (
            pd.DataFrame(vectors).rolling(window, min_periods=min_periods).mean().shift(1).to_numpy()
        )
        dot = np.einsum("ij,ij->i", vectors, prior)
        norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(prior, axis=1)
        cosine = np.where(norms > 0, dot / norms, np.nan)

        frame = group[["date", "stock", "fold"]].copy()
        frame["consistency"] = cosine
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def risk_coverage(score, correct, grid) -> pd.DataFrame:
    """Selective accuracy when the most reliable `coverage` share is kept."""
    score = np.asarray(score, dtype=float)
    correct = np.asarray(correct)
    order = np.argsort(-score)

    rows = []
    for coverage in grid:
        keep = max(1, int(round(coverage * len(correct))))
        rows.append({"coverage": coverage, "accuracy": float(correct[order[:keep]].mean())})
    return pd.DataFrame(rows)


def _minmax(values: np.ndarray) -> np.ndarray:
    spread = values.max() - values.min()
    return (values - values.min()) / spread if spread > 0 else np.zeros_like(values)


def run(config: dict) -> dict:
    processed = Path(config["paths"]["processed"])
    explanations_dir = Path(config["paths"]["explanations"])
    tables_dir = Path(config["paths"]["tables"])

    predictions = pd.read_parquet(explanations_dir / "predictions.parquet")
    features = feature_columns(config)
    window = config["consistency"]["diagnostic_window"]
    grid = config["abstention"]["coverage_grid"]

    summary_rows = []
    regression_rows = []
    coverage_rows = []
    for model in ("xgboost", "logistic"):
        shap_frame = pd.read_parquet(explanations_dir / f"shap_{model}.parquet")
        scores = consistency_scores(shap_frame, features, window)

        merged = (
            predictions[predictions["model"] == model]
            .merge(scores, on=["date", "stock", "fold"], how="left")
            .dropna(subset=["consistency"])
            .reset_index(drop=True)
        )
        merged["y_pred"] = (merged["y_prob"] >= 0.5).astype(int)
        merged["correct"] = (merged["y_pred"] == merged["y_true"]).astype(int)
        merged["confidence"] = (merged["y_prob"] - 0.5).abs()
        merged.to_parquet(explanations_dir / f"diagnostic_{model}.parquet", index=False)

        auc_consistency = roc_auc_score(merged["correct"], merged["consistency"])
        auc_confidence = roc_auc_score(merged["correct"], merged["confidence"])

        median = merged["consistency"].median()
        low = merged[merged["consistency"] <= median]
        high = merged[merged["consistency"] > median]
        summary_rows.append(
            {
                "model": model,
                "n": len(merged),
                "auc_consistency_vs_correct": float(auc_consistency),
                "auc_confidence_vs_correct": float(auc_confidence),
                "accuracy_low_consistency": float(low["correct"].mean()),
                "accuracy_high_consistency": float(high["correct"].mean()),
                "accuracy_all": float(merged["correct"].mean()),
            }
        )

        model_frame = merged.assign(
            consistency_z=lambda d: (d["consistency"] - d["consistency"].mean()) / d["consistency"].std(),
            confidence_z=lambda d: (d["confidence"] - d["confidence"].mean()) / d["confidence"].std(),
        )
        try:
            fit = smf.logit("correct ~ consistency_z + confidence_z", data=model_frame).fit(disp=0)
            for term in ("consistency_z", "confidence_z"):
                regression_rows.append(
                    {
                        "model": model,
                        "term": term,
                        "coefficient": float(fit.params[term]),
                        "p_value": float(fit.pvalues[term]),
                        "pseudo_r2": float(fit.prsquared),
                    }
                )
        except Exception as error:  # separation or degenerate data
            regression_rows.append(
                {"model": model, "term": "regression_failed", "coefficient": np.nan,
                 "p_value": np.nan, "pseudo_r2": np.nan}
            )

        combined = (_minmax(merged["confidence"].to_numpy()) + _minmax(merged["consistency"].to_numpy())) / 2
        for rule, score in (
            ("confidence", merged["confidence"].to_numpy()),
            ("consistency", merged["consistency"].to_numpy()),
            ("combined", combined),
        ):
            table = risk_coverage(score, merged["correct"], grid)
            table["rule"] = rule
            table["model"] = model
            coverage_rows.append(table)

    summary = pd.DataFrame(summary_rows)
    regression = pd.DataFrame(regression_rows)
    coverage = pd.concat(coverage_rows, ignore_index=True)[["model", "rule", "coverage", "accuracy"]]

    summary.to_csv(tables_dir / "diagnostic_summary.csv", index=False)
    regression.to_csv(tables_dir / "diagnostic_regression.csv", index=False)
    coverage.to_csv(tables_dir / "abstention_risk_coverage.csv", index=False)

    print(summary.round(4).to_string(index=False))
    return {"summary": summary, "regression": regression, "coverage": coverage}


if __name__ == "__main__":
    run(load_config())

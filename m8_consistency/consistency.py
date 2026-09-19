"""M8a — SP6a explanation consistency at two levels of analysis.

The central measurement: stability is computed twice, on the same SHAP output —
once per feature (29 dimensions) and once per financial group (6 dimensions).
Under multicollinearity, per-feature instability partly reflects credit-splitting
between correlated indicators; the group level tests whether stability is
materially higher when the unit of analysis is financial-semantic.

Metrics between two importance vectors: Top-K overlap, Spearman rank correlation,
cosine similarity and MARC (mean absolute rank change). Comparisons are made
across adjacent rolling windows and across VIX regimes.

Outputs:
    results/tables/consistency_rolling.csv
    results/tables/consistency_regimes.csv
    results/tables/consistency_level_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from m0_setup.config import load_config
from m2_features.features import feature_columns
from m7_interpretation.interpretation import GROUPS, aggregate_groups


def _top_k_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    top_a = set(np.argsort(-a)[:k])
    top_b = set(np.argsort(-b)[:k])
    return len(top_a & top_b) / k


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).correlation)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0 else float("nan")


def _marc(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute rank change (lower is more stable)."""
    return float(np.abs(rankdata(-a) - rankdata(-b)).mean())


def _metrics(a: np.ndarray, b: np.ndarray, k: int) -> dict:
    return {
        "topk_overlap": _top_k_overlap(a, b, k),
        "spearman": _spearman(a, b),
        "cosine": _cosine(a, b),
        "marc": _marc(a, b),
    }


def _feature_importance(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return frame[features].abs().mean().to_numpy()


def _group_importance(grouped: pd.DataFrame) -> np.ndarray:
    return grouped[list(GROUPS)].abs().mean().to_numpy()


def run(config: dict) -> dict[str, pd.DataFrame]:
    """Compute rolling-window and regime consistency at both levels."""
    processed = Path(config["paths"]["processed"])
    explanations_dir = Path(config["paths"]["explanations"])
    tables_dir = Path(config["paths"]["tables"])

    table = pd.read_parquet(processed / "model_table.parquet")
    features = feature_columns(config)
    window = config["consistency"]["rolling_window"]
    feature_k = config["consistency"]["top_k"]
    group_k = max(1, round(len(GROUPS) / 3))
    regime_bounds = config["consistency"]["regimes"]

    unique_dates = np.sort(pd.unique(table["date"]))
    quantiles = np.quantile(table["vix"], [regime_bounds["low"][1], regime_bounds["medium"][1]])

    rolling_rows = []
    regime_rows = []
    for model in ("xgboost", "logistic"):
        shap_frame = pd.read_parquet(explanations_dir / f"shap_{model}.parquet")
        grouped = aggregate_groups(shap_frame, features)

        windows = [unique_dates[i : i + window] for i in range(0, len(unique_dates) - window + 1, window)]
        for index in range(len(windows) - 1):
            current = shap_frame["date"].isin(windows[index])
            following = shap_frame["date"].isin(windows[index + 1])
            rolling_rows.append(
                {
                    "model": model,
                    "level": "feature",
                    "comparison": f"window_{index}_vs_{index + 1}",
                    **_metrics(
                        _feature_importance(shap_frame[current], features),
                        _feature_importance(shap_frame[following], features),
                        feature_k,
                    ),
                }
            )
            rolling_rows.append(
                {
                    "model": model,
                    "level": "group",
                    "comparison": f"window_{index}_vs_{index + 1}",
                    **_metrics(
                        _group_importance(grouped[current]),
                        _group_importance(grouped[following]),
                        group_k,
                    ),
                }
            )

        vix = shap_frame["vix"]
        regime_masks = {
            "low": vix <= quantiles[0],
            "medium": (vix > quantiles[0]) & (vix <= quantiles[1]),
            "high": vix > quantiles[1],
        }
        for first, second in (("low", "medium"), ("medium", "high"), ("low", "high")):
            regime_rows.append(
                {
                    "model": model,
                    "level": "feature",
                    "comparison": f"{first}_vs_{second}",
                    **_metrics(
                        _feature_importance(shap_frame[regime_masks[first]], features),
                        _feature_importance(shap_frame[regime_masks[second]], features),
                        feature_k,
                    ),
                }
            )
            regime_rows.append(
                {
                    "model": model,
                    "level": "group",
                    "comparison": f"{first}_vs_{second}",
                    **_metrics(
                        _group_importance(grouped[regime_masks[first]]),
                        _group_importance(grouped[regime_masks[second]]),
                        group_k,
                    ),
                }
            )

    rolling = pd.DataFrame(rolling_rows).sort_values(["model", "level", "comparison"])
    regimes = pd.DataFrame(regime_rows).sort_values(["model", "level", "comparison"])
    rolling.to_csv(tables_dir / "consistency_rolling.csv", index=False)
    regimes.to_csv(tables_dir / "consistency_regimes.csv", index=False)

    combined = pd.concat([rolling.assign(section="rolling"), regimes.assign(section="regime")])
    summary = (
        combined.groupby(["model", "level"])[["topk_overlap", "spearman", "cosine", "marc"]]
        .mean()
        .round(4)
    )
    summary.to_csv(tables_dir / "consistency_level_summary.csv")

    print("Feature-level vs group-level stability (mean over comparisons)")
    print(summary.to_string())
    return {"rolling": rolling, "regimes": regimes, "summary": summary}


if __name__ == "__main__":
    run(load_config())

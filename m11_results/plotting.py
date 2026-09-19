"""M11 — results assembly.

Builds every figure referenced by A2 §4 from the saved tables and attributions,
and writes a manifest recording the config hash, git commit and seeds so the
results are reproducible.

Outputs: results/figures/*.png and results/manifest.json.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from m0_setup.config import load_config
from m2_features.features import feature_columns
from m4_models.evaluation import classification_metrics
from m7_interpretation.interpretation import GROUPS

COLORS = {"logistic": "#3b6ea5", "xgboost": "#c1553b", "majority_class": "#999999", "persistence": "#cccccc"}


def _regime_frame(predictions: pd.DataFrame, table: pd.DataFrame, config: dict) -> pd.DataFrame:
    bounds = config["consistency"]["regimes"]
    vix = table[["date", "stock", "vix"]]
    merged = predictions.merge(vix, on=["date", "stock"], how="left")
    quantiles = np.quantile(merged["vix"], [bounds["low"][1], bounds["medium"][1]])
    merged["regime"] = np.where(
        merged["vix"] <= quantiles[0], "low", np.where(merged["vix"] <= quantiles[1], "medium", "high")
    )
    return merged


def fig_performance_comparison(pooled: pd.DataFrame, baselines: pd.DataFrame, out: Path) -> None:
    combined = pd.concat([pooled[["model", "accuracy", "roc_auc", "mcc"]],
                          baselines[["model", "accuracy", "roc_auc", "mcc"]]], ignore_index=True)
    metrics = ["accuracy", "roc_auc", "mcc"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric in zip(axes, metrics):
        axis.bar(combined["model"], combined[metric],
                 color=[COLORS.get(m, "#888") for m in combined["model"]])
        axis.axhline(0.5 if metric in {"accuracy", "roc_auc"} else 0.0, color="black", lw=0.8, ls="--")
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.3)
    fig.suptitle("RQ1 — pooled predictive performance vs naive baselines")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_performance_by_fold(per_fold: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for metric, axis in zip(["roc_auc", "accuracy"], axes):
        for model, group in per_fold.groupby("model"):
            axis.plot(group["fold"], group[metric], marker="o", label=model, color=COLORS.get(model))
        axis.axhline(0.5, color="black", lw=0.8, ls="--")
        axis.set_xlabel("walk-forward fold")
        axis.set_ylabel(metric)
        axis.set_title(f"{metric} per fold")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("RQ1 — performance stability across folds")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_performance_by_regime(predictions: pd.DataFrame, table: pd.DataFrame, config: dict, out: Path) -> pd.DataFrame:
    merged = _regime_frame(predictions, table, config)
    rows = []
    for (model, regime), group in merged.groupby(["model", "regime"]):
        rows.append({"model": model, "regime": regime, **classification_metrics(group["y_true"], group["y_prob"])})
    regime_metrics = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, axis in zip(["accuracy", "roc_auc"], axes):
        pivot = regime_metrics.pivot(index="regime", columns="model", values=metric).reindex(["low", "medium", "high"])
        pivot.plot(kind="bar", ax=axis, color=[COLORS.get(c) for c in pivot.columns])
        axis.axhline(0.5, color="black", lw=0.8, ls="--")
        axis.set_title(metric)
        axis.grid(axis="y", alpha=0.3)
    fig.suptitle("RQ1 — performance across VIX regimes")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return regime_metrics


def fig_shap_importance(importance: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for axis, model in zip(axes, ["xgboost", "logistic"]):
        top = importance[importance["model"] == model].nsmallest(15, "rank").sort_values("mean_abs_shap")
        axis.barh(top["feature"], top["mean_abs_shap"], color=COLORS.get(model))
        axis.set_title(f"{model}: top 15 mean |SHAP|")
        axis.grid(axis="x", alpha=0.3)
    fig.suptitle("RQ2 — global feature importance")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_shap_beeswarm(shap_frame: pd.DataFrame, table: pd.DataFrame, features: list[str], config: dict, out: Path) -> None:
    merged = shap_frame.merge(table[["date", "stock"] + features], on=["date", "stock"], suffixes=("_shap", "_value"))
    sample = merged.sample(min(1000, len(merged)), random_state=config["seed"])
    values = sample[[f"{f}_shap" for f in features]].to_numpy()
    data = sample[[f"{f}_value" for f in features]]
    data.columns = features

    np.random.seed(config["seed"])  # beeswarm jitter is otherwise non-deterministic
    shap.summary_plot(values, data, max_display=15, show=False)
    plt.title("RQ2 — XGBoost SHAP beeswarm (direction of effect)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def fig_group_importance(group_importance: pd.DataFrame, out: Path) -> None:
    pivot = group_importance.pivot(index="group", columns="model", values="share_pct").reindex(GROUPS)
    axis = pivot.plot(kind="barh", figsize=(9, 5), color=[COLORS.get(c) for c in pivot.columns])
    axis.set_xlabel("share of total mean |group SHAP| (%)")
    axis.set_title("RQ3 — SHAP importance by financial group")
    axis.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(axis.figure)


def fig_stability_levels(summary: pd.DataFrame, out: Path) -> None:
    metrics = ["topk_overlap", "spearman", "cosine"]
    labels = [f"{model}\n{level}" for model, level in summary.index]
    x = np.arange(len(summary))
    width = 0.25

    fig, axis = plt.subplots(figsize=(10, 5))
    for offset, metric in enumerate(metrics):
        axis.bar(x + (offset - 1) * width, summary[metric].to_numpy(), width, label=metric)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1.05)
    axis.axhline(0.5, color="black", lw=0.8, ls="--")
    axis.set_title("RQ4a — feature-level vs group-level explanation stability")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_group_by_regime(shap_frame: pd.DataFrame, group_shap: pd.DataFrame, config: dict, out: Path) -> None:
    merged = group_shap.merge(shap_frame[["date", "stock", "fold", "vix"]], on=["date", "stock", "fold"])
    bounds = config["consistency"]["regimes"]
    quantiles = np.quantile(merged["vix"], [bounds["low"][1], bounds["medium"][1]])
    merged["regime"] = np.where(
        merged["vix"] <= quantiles[0], "low", np.where(merged["vix"] <= quantiles[1], "medium", "high")
    )

    rows = []
    for regime in ("low", "medium", "high"):
        subset = merged[merged["regime"] == regime]
        total = sum(subset[group].abs().mean() for group in GROUPS)
        for group in GROUPS:
            rows.append(
                {"regime": regime, "group": group, "share_pct": 100 * subset[group].abs().mean() / total}
            )
    shares = pd.DataFrame(rows).pivot(index="group", columns="regime", values="share_pct").reindex(GROUPS)

    axis = shares.plot(kind="barh", figsize=(9, 5))
    axis.set_xlabel("share of total mean |group SHAP| (%)")
    axis.set_title("RQ4a — dominant financial group across VIX regimes (XGBoost)")
    axis.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(axis.figure)


def fig_risk_coverage(coverage: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, model in zip(axes, ["xgboost", "logistic"]):
        subset = coverage[coverage["model"] == model]
        for rule, group in subset.groupby("rule"):
            axis.plot(group["coverage"], group["accuracy"], marker="o", label=rule)
        axis.set_title(model)
        axis.set_xlabel("coverage (share of predictions kept)")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("selective accuracy")
    fig.suptitle("RQ4b — risk-coverage of confidence, consistency and combined abstention")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_manifest(config: dict, figures: list[Path]) -> None:
    results = Path(config["paths"]["results"])
    config_file = results.parent / "config.yaml"
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
            cwd=config_file.parent,
        ).stdout.strip()
    except Exception:
        git_commit = None

    tables = sorted(p.name for p in (results / "tables").glob("*.csv"))
    manifest = {
        "seed": config["seed"],
        "config_hash": hashlib.sha256(config_file.read_bytes()).hexdigest()[:16],
        "git_commit": git_commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "figures": sorted(p.name for p in figures),
        "tables": tables,
    }
    (results / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run(config: dict) -> list[Path]:
    results = Path(config["paths"]["results"])
    figures = results / "figures"
    tables = results / "tables"
    explanations = results / "explanations"
    figures.mkdir(parents=True, exist_ok=True)

    table = pd.read_parquet(Path(config["paths"]["processed"]) / "model_table.parquet")
    features = feature_columns(config)
    predictions = pd.read_parquet(explanations / "predictions.parquet")

    pooled = pd.read_csv(tables / "model_metrics_pooled.csv")
    baselines = pd.read_csv(tables / "baseline_metrics.csv")
    per_fold = pd.read_csv(tables / "model_metrics_per_fold.csv")
    importance = pd.read_csv(tables / "shap_global_importance.csv")
    group_importance = pd.read_csv(tables / "shap_group_importance.csv")
    stability = pd.read_csv(tables / "consistency_level_summary.csv", index_col=[0, 1])
    coverage = pd.read_csv(tables / "abstention_risk_coverage.csv")
    shap_xgboost = pd.read_parquet(explanations / "shap_xgboost.parquet")
    group_xgboost = pd.read_parquet(explanations / "group_shap_xgboost.parquet")

    created = [
        figures / "rq1_performance_vs_baselines.png",
        figures / "rq1_performance_by_fold.png",
        figures / "rq1_performance_by_regime.png",
        figures / "rq2_shap_importance.png",
        figures / "rq2_shap_beeswarm_xgboost.png",
        figures / "rq3_group_importance.png",
        figures / "rq4a_stability_levels.png",
        figures / "rq4a_group_by_regime_xgboost.png",
        figures / "rq4b_risk_coverage.png",
    ]

    fig_performance_comparison(pooled, baselines, created[0])
    fig_performance_by_fold(per_fold, created[1])
    regime_metrics = fig_performance_by_regime(predictions, table, config, created[2])
    regime_metrics.to_csv(tables / "model_metrics_by_regime.csv", index=False)
    fig_shap_importance(importance, created[3])
    fig_shap_beeswarm(shap_xgboost, table, features, config, created[4])
    fig_group_importance(group_importance, created[5])
    fig_stability_levels(stability, created[6])
    fig_group_by_regime(shap_xgboost, group_xgboost, config, created[7])
    fig_risk_coverage(coverage, created[8])

    write_manifest(config, created)
    for path in created:
        print(f"wrote {path.name}")
    return created


if __name__ == "__main__":
    run(load_config())

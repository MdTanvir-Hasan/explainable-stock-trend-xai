# An Explainable AI Framework for Short-Term Stock Trend Prediction

Financial interpretation and explanation-consistency analysis for ASX equities.

## Summary

The framework predicts next-day UP/DOWN for ten ASX large-caps, explains each
prediction with SHAP, interprets the explanations in financial terms, and tests
whether explanation stability carries information. The question is a deliberate
shift in framing: prior work measures explanation stability as a model property;
this work asks whether that stability indicates reliability.

## Research questions

- **RQ1** Predictive performance of machine learning on short-term trend.
- **RQ2** Which technical, market and economic factors drive predictions.
- **RQ3** Whether explanations correspond to established financial knowledge.
- **RQ4a** Explanation stability over time and across market regimes, measured per
  feature and per financial-semantic group.
- **RQ4b** Whether instability signals reduced reliability and supports abstention.

## Data and method

Ten ASX large-caps plus the S&P/ASX 200 index, the S&P/ASX 200 VIX, and RBA cash
rate and 2y/10y bond yields; daily from 2015. Macro is shifted by a one-business-day
publication lag. About 29 features are grouped into six financial themes. The target
is next-day direction on the adjusted close. Models are Logistic Regression and
XGBoost, evaluated with eight expanding walk-forward folds and no random splits.
Explanations use TreeSHAP and LinearSHAP. Consistency is measured with Top-K
overlap, Spearman correlation, cosine similarity and MARC at both the feature and
group level, across rolling windows and VIX regimes. A per-prediction consistency
score feeds a risk-coverage comparison of confidence-based, consistency-based and
combined abstention. Faithfulness is tested by permutation and placebo.

## Key findings

- **Prediction is at chance.** ROC-AUC 0.496 (logistic) and 0.498 (XGBoost), MCC
  near zero (−0.006 / 0.002), no edge over a majority-class baseline; per-fold
  XGBoost ROC-AUC ranges 0.421–0.538 and changes sign between folds.
- **The two families reason differently.** XGBoost leans on macro and market
  factors (`bond_10y` mean |SHAP| 0.123, `term_spread` 0.099, `vix` 0.088);
  logistic regression leans on trend ratios (`close_sma_20` 0.252, `close_ema_12`
  0.149). By theme, XGBoost attributes 35.4% to economic and 27.0% to market
  conditions; logistic regression 28.2% to momentum and 27.5% to trend.
- **Stability depends on the unit of analysis.** Group-level explanation stability
  exceeds feature-level on every metric for XGBoost (MARC 3.62 → 0.48; Spearman
  0.830 → 0.910) and on three of four for logistic regression. Correlated indicators
  split attribution at the feature level; grouping by financial theme absorbs it.
- **Instability is not a reliability signal.** Consistency does not separate correct
  from incorrect predictions (AUC 0.488 / 0.510) and does not beat confidence as an
  abstention driver. Reported as a null result.
- **The explanations are faithful.** Shuffling top features changes predictions
  about 3 to 24 times more than shuffling bottom features; a random placebo feature
  ranks 21st to 23rd of 30.

## Repository structure

```
pipeline.ipynb        the cumulative notebook; run this
config.yaml           default parameters (the notebook's Parameters cell overrides them)
requirements.txt
m0_setup/             configuration loading
m1_data/              data acquisition (Yahoo Finance and RBA)
m2_features/          alignment, feature engineering, next-day target
m3_walkforward/       chronological walk-forward splits
m4_models/            Logistic Regression and XGBoost with predictive metrics
m6_shap/              TreeSHAP and LinearSHAP attributions
m7_interpretation/    six financial groups and direction of effect
m8_consistency/       two-level consistency, diagnostic and abstention
m9_faithfulness/      permutation and placebo tests
m10_prototype/        Streamlit decision-support app and prototype logic
m11_results/          report figures and run manifest
tests/                29 leakage and correctness tests
results/              generated figures, tables and attributions (committed)
data/                 downloaded on first run (not committed)
```

Milestones appear one per commit in the history: setup and data, features,
walk-forward, models, SHAP, interpretation, consistency, results.

## Run it

Python 3.13, fixed seed, CPU only. The dataset downloads on first run.

```powershell
py -3.13 -m venv "$env:USERPROFILE\.venvs\stockxai"
& "$env:USERPROFILE\.venvs\stockxai\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
python -m ipykernel install --user --name stockxai --display-name "Python (stockxai)"
jupyter lab
```

Open `pipeline.ipynb`, edit the **Parameters** cell (download date range, stock
list, seed, walk-forward sizes, model tuning, feature windows, SHAP sample size),
then Run All. It runs every stage in order and finishes by running the tests.

Individual stages also run from the command line, for example:

```powershell
python -m m1_data.data_collection
python -m m4_models.models
python -m m11_results.plotting
pytest -q
```

The Streamlit prototype runs separately:

```powershell
streamlit run m10_prototype/streamlit_app.py
```

## Results

Generated outputs are committed under `results/`:

- `results/figures/` — nine report figures (performance, SHAP importance and
  beeswarm, group importance, stability, regime, risk-coverage).
- `results/tables/` — metrics, consistency, diagnostic, abstention and
  faithfulness tables.
- `results/explanations/` — SHAP and group attributions, predictions.
- `results/manifest.json` — config hash, git commit and seed for the run.

## Scope

Ten large-cap ASX equities only. Explanations are associational, not causal. The
prototype is a research demonstration, not a trading system. The LSTM comparison,
DeepSHAP and a LIME cross-check are deferred and reported as outstanding.

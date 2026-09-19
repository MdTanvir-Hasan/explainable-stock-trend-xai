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
  near zero, no edge over a majority-class baseline.
- **Stability depends on the unit of analysis.** Group-level explanation stability
  exceeds feature-level on every metric for XGBoost (MARC 3.48 to 0.38) and on three
  of four for logistic regression. Correlated indicators split attribution at the
  feature level; grouping by financial theme absorbs that churn.
- **Instability is not a reliability signal.** Consistency does not separate correct
  from incorrect predictions (AUC 0.485 / 0.510) and does not beat confidence as an
  abstention driver. Reported as a null result.
- **The explanations are faithful.** Shuffling top features changes predictions about
  3 to 24 times more than shuffling bottom features; a random placebo feature ranks
  18th to 21st of 30.

## Reproduce

Python 3.13, fixed seed, CPU only. Open `pipeline.ipynb`, set the download date range
and any tuning in the **Parameters** cell, and Run All. Data downloads on first run.

```
pip install -r requirements.txt
jupyter lab        # open pipeline.ipynb, Run All
```

`commits/` holds cumulative per-milestone snapshots (each a runnable project at that
point); `_build/` regenerates and verifies them. The full write-up, tables and
figures are in [Research_Progress_Results_A2.md](Research_Progress_Results_A2.md).

## Scope

Ten large-cap ASX equities only. Explanations are associational, not causal. The
prototype is a research demonstration, not a trading system. The LSTM comparison,
DeepSHAP and a LIME cross-check are deferred and reported as outstanding.

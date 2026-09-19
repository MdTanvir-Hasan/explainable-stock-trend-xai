"""SHAP values must sum to the model output minus the base value."""
from __future__ import annotations

import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from m2_features.features import add_features, feature_columns


def _xy(synthetic_panel):
    panel, config = synthetic_panel
    table = add_features(panel, config, save=False)
    features = feature_columns(config)
    return table[features].to_numpy(), table["target"].to_numpy(), features


def test_tree_shap_additivity(synthetic_panel):
    x, y, _ = _xy(synthetic_panel)
    model = XGBClassifier(n_estimators=20, max_depth=3, random_state=0, tree_method="hist")
    model.fit(x, y)

    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(x))
    margin = model.predict(x, output_margin=True)

    reconstructed = values.sum(axis=1) + explainer.expected_value
    assert np.allclose(reconstructed, margin, atol=1e-4)


def test_linear_shap_additivity(synthetic_panel):
    x, y, _ = _xy(synthetic_panel)
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    model = LogisticRegression(max_iter=200, random_state=0).fit(x_scaled, y)

    explainer = shap.LinearExplainer(model, x_scaled)
    values = np.asarray(explainer.shap_values(x_scaled))
    decision = model.decision_function(x_scaled)

    reconstructed = values.sum(axis=1) + float(np.ravel(explainer.expected_value)[-1])
    assert np.allclose(reconstructed, decision, atol=1e-4)

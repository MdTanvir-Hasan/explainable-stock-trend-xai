"""A predictive feature must matter more to the model than a noise feature."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def test_permuting_predictive_feature_changes_predictions_more():
    rng = np.random.default_rng(0)
    informative = rng.normal(size=2000)
    noise = rng.normal(size=2000)
    y = (informative > 0).astype(int)
    x = np.column_stack([informative, noise])

    model = LogisticRegression().fit(x, y)
    base = model.predict_proba(x)[:, 1]

    delta_informative = np.abs(
        model.predict_proba(np.column_stack([rng.permutation(informative), noise]))[:, 1] - base
    ).mean()
    delta_noise = np.abs(
        model.predict_proba(np.column_stack([informative, rng.permutation(noise)]))[:, 1] - base
    ).mean()

    assert delta_informative > delta_noise

"""Feature matrix must be finite: zero-volume days must not create infinities."""
from __future__ import annotations

import numpy as np

from conftest import make_panel
from m2_features.features import add_features, feature_columns


def test_zero_volume_days_do_not_produce_infinities(synthetic_panel):
    _, config = synthetic_panel
    panel = make_panel()
    panel.loc[panel.index[:3], "Volume"] = 0.0

    table = add_features(panel, config, save=False)
    values = table[feature_columns(config)].to_numpy()

    assert np.isfinite(values).all()

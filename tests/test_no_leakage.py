"""Truncating the data after time T must not change any feature at or before T."""
from __future__ import annotations

import numpy as np

from m2_features.features import add_features, feature_columns


def test_features_ignore_future_rows(synthetic_panel):
    panel, config = synthetic_panel
    full = add_features(panel, config, save=False)

    dates = np.sort(panel["date"].unique())
    cutoff = dates[len(dates) // 2]
    truncated = add_features(panel[panel["date"] <= cutoff], config, save=False)

    columns = feature_columns(config)
    merged = full.merge(truncated, on=["stock", "date"], suffixes=("_full", "_trunc"))
    assert len(merged) > 0

    for column in columns:
        assert np.array_equal(
            merged[f"{column}_full"].to_numpy(),
            merged[f"{column}_trunc"].to_numpy(),
            equal_nan=True,
        ), f"feature {column} changed when future rows were removed"


def test_truncation_removes_only_future_rows(synthetic_panel):
    panel, config = synthetic_panel
    dates = np.sort(panel["date"].unique())
    cutoff = dates[len(dates) // 2]

    full = add_features(panel, config, save=False)
    truncated = add_features(panel[panel["date"] <= cutoff], config, save=False)

    assert truncated["date"].max() < full["date"].max()
    assert len(truncated) < len(full)

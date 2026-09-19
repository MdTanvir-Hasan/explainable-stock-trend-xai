"""Every feature must map to exactly one of the six financial groups."""
from __future__ import annotations

from m2_features.features import feature_columns
from m7_interpretation.interpretation import GROUPS, feature_group


def test_every_feature_maps_to_a_known_group(synthetic_panel):
    _, config = synthetic_panel
    features = feature_columns(config)
    mapping = {feature: feature_group(feature) for feature in features}

    assert set(mapping.values()) <= set(GROUPS)
    assert len(mapping) == len(features)


def test_every_group_is_used(synthetic_panel):
    _, config = synthetic_panel
    used = {feature_group(feature) for feature in feature_columns(config)}
    assert used == set(GROUPS)

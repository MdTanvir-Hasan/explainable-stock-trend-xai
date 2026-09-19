"""Interpretation text must reflect the prediction it is given."""
from __future__ import annotations

import numpy as np
import pandas as pd

from m10_prototype.prototype import Prediction, interpretation_text


def _prediction(probability_up: float, consistency: float) -> Prediction:
    return Prediction(
        stock="BHP",
        date="2026-09-17",
        probability_up=probability_up,
        label="UP" if probability_up >= 0.5 else "DOWN",
        consistency=consistency,
        drivers=pd.DataFrame({"feature": ["market_ret"], "group": ["market_conditions"], "shap": [0.2]}),
        group_shares=pd.Series(
            {"momentum": 10.0, "trend": 5.0, "volatility": 5.0,
             "market_conditions": 70.0, "trading_activity": 5.0, "economic_conditions": 5.0}
        ),
    )


def test_text_states_direction_and_top_group():
    text = interpretation_text(_prediction(0.7, 0.8))
    assert "upward" in text
    assert "market conditions" in text
    assert "in line with" in text


def test_downward_and_diverging_wording():
    text = interpretation_text(_prediction(0.3, 0.1))
    assert "downward" in text
    assert "diverging" in text


def test_missing_consistency_is_handled():
    text = interpretation_text(_prediction(0.6, float("nan")))
    assert "not yet available" in text

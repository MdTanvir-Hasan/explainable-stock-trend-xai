"""A macro value must never be visible before its release date plus the lag."""
from __future__ import annotations

import pandas as pd

from m2_features.preprocessing import _macro_frame


def test_macro_values_shifted_by_publication_lag(synthetic_panel):
    _, config = synthetic_panel
    index = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"])
    macro = pd.DataFrame(
        {"cash_rate": [1.0, 1.0, 1.25], "bond_2y": [1.0, 1.0, 1.0], "bond_10y": [2.0, 2.0, 2.0]},
        index=index,
    )

    shifted = _macro_frame(config, macro)
    lag = config["data"]["macro_lag_days"]

    assert (shifted.index == index + pd.tseries.offsets.BDay(lag)).all()
    assert shifted.index.min() > index.min()


def test_shift_is_at_least_one_business_day(synthetic_panel):
    _, config = synthetic_panel
    assert config["data"]["macro_lag_days"] >= 1

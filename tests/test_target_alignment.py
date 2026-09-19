"""The target must equal the sign of the move from t to t+1."""
from __future__ import annotations

import pandas as pd

from m2_features.features import add_features


def test_target_is_next_day_direction(synthetic_panel):
    panel, config = synthetic_panel
    table = add_features(panel, config, save=False)

    reference = panel.copy()
    reference["price"] = reference["Adj Close"]
    expected = (
        reference.groupby("stock")["price"].shift(-1) > reference["price"]
    ).astype(float)
    reference["expected"] = expected.to_numpy()

    merged = table.merge(reference[["date", "stock", "expected"]], on=["date", "stock"], how="left")
    assert merged["expected"].notna().all()
    assert (merged["target"] == merged["expected"]).all()


def test_final_row_per_stock_is_dropped(synthetic_panel):
    panel, config = synthetic_panel
    table = add_features(panel, config, save=False)

    last_dates = panel.groupby("stock")["date"].max()
    for stock, last_date in last_dates.items():
        present = ((table["stock"] == stock) & (table["date"] == last_date)).sum()
        assert present == 0, f"uncertain target kept for {stock} on {last_date}"


def test_no_missing_targets_remain(synthetic_panel):
    panel, config = synthetic_panel
    table = add_features(panel, config, save=False)
    assert table["target"].notna().all()
    assert set(pd.unique(table["target"])) <= {0.0, 1.0}

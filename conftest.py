"""Shared pytest fixtures and import path setup."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from m0_setup.config import load_config  # noqa: E402


def make_panel(n_days: int = 300, stocks: tuple[str, ...] = ("AAA", "BBB"), seed: int = 0) -> pd.DataFrame:
    """Synthetic aligned panel with the columns add_features expects."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)

    frames = []
    for stock in stocks:
        price = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n_days))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "stock": stock,
                    "Open": price,
                    "High": price * 1.01,
                    "Low": price * 0.99,
                    "Close": price,
                    "Adj Close": price,
                    "Volume": rng.integers(100_000, 1_000_000, n_days).astype(float),
                    "market_ret": rng.normal(0.0, 0.005, n_days),
                    "market_rv_21": 0.01,
                    "vix": 15.0,
                    "vix_change_5": 0.0,
                    "cash_rate": 0.5,
                    "bond_2y": 1.0,
                    "bond_10y": 2.0,
                    "term_spread": 1.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def synthetic_panel() -> tuple[pd.DataFrame, dict]:
    return make_panel(), load_config(ROOT / "config.yaml")

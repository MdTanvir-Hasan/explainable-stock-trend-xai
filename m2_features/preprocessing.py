"""M2 — SP2 alignment.

Reads the M1 raw series and builds one long-format panel keyed by (date, stock).
No indicators are computed here; this module only makes the information available
at time t correct.

Leakage rules enforced:
    * market data (index, VIX) is merged as-of the stock trading date, never
      taken from a later date;
    * macro values are shifted by `macro_lag_days` business days before use, so a
      value becomes visible only after its publication lag.

Output: data/processed/panel.parquet (date, stock, OHLCV, market and macro columns).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from m0_setup.config import load_config


def _read_raw(path: Path) -> pd.DataFrame:
    """Load one raw CSV saved by M1, indexed by trading date."""
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = "date"
    return frame.sort_index()


def _market_frame(config: dict, index_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Date-indexed market features: ASX 200 return, market volatility, VIX."""
    window = config["features"]["realised_vol_window"]
    vix_window = config["features"]["vix_change_window"]

    market = pd.DataFrame(index=index_df.index)
    market["market_ret"] = index_df["Adj Close"].pct_change()
    market[f"market_rv_{window}"] = market["market_ret"].rolling(window, min_periods=window).std()

    vix = vix_df["Adj Close"].reindex(market.index)
    market["vix"] = vix
    market[f"vix_change_{vix_window}"] = vix.pct_change(vix_window)
    return market


def _macro_frame(config: dict, macro_df: pd.DataFrame) -> pd.DataFrame:
    """Macro features shifted forward by the publication lag."""
    macro = pd.DataFrame(index=macro_df.index)
    macro["cash_rate"] = macro_df["cash_rate"]
    macro["bond_2y"] = macro_df["bond_2y"]
    macro["bond_10y"] = macro_df["bond_10y"]
    macro["term_spread"] = macro["bond_10y"] - macro["bond_2y"]

    lag = pd.tseries.offsets.BDay(config["data"]["macro_lag_days"])
    macro.index = macro.index + lag
    return macro


def build_panel(config: dict) -> pd.DataFrame:
    """Combine stocks with lag-safe market and macro columns and save the panel."""
    raw_dir = Path(config["paths"]["raw"])
    stocks = config["data"]["stocks"]

    parts = []
    for stock in stocks:
        frame = _read_raw(raw_dir / f"{stock}.AX.csv")
        frame["stock"] = stock
        parts.append(frame)
    panel = pd.concat(parts, ignore_index=False).sort_values("date").reset_index()

    index_df = _read_raw(raw_dir / "AXJO.csv")
    vix_df = _read_raw(raw_dir / "AXVI.csv")
    macro_df = _read_raw(raw_dir / "rba_macro.csv")

    market = _market_frame(config, index_df, vix_df).reset_index()
    macro = _macro_frame(config, macro_df).reset_index()

    panel = pd.merge_asof(panel, market, on="date", direction="backward")
    panel = pd.merge_asof(panel, macro, on="date", direction="backward")

    out_path = Path(config["paths"]["processed"]) / "panel.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)

    print(f"panel: {panel.shape[0]} rows x {panel.shape[1]} cols -> {out_path.name}")
    return panel


if __name__ == "__main__":
    build_panel(load_config())

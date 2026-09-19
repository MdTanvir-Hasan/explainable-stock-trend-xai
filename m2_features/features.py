"""M2 — SP2 feature engineering and target construction.

Every indicator at row t is computed per stock from information available at or
before t. The target is next-day direction on the adjusted close:

    target(t) = 1 if price(t+1) > price(t) else 0

The last row per stock has no target and is dropped, as are warm-up rows whose
rolling windows are incomplete.

Output: data/processed/model_table.parquet and feature_columns.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from m0_setup.config import load_config
from m2_features.preprocessing import build_panel


def feature_columns(config: dict) -> list[str]:
    """Ordered feature names, derived from the config so names never drift."""
    f = config["features"]
    columns = ["ret_1"]
    columns += [f"ret_lag_{lag}" for lag in f["return_lags"]]
    columns += [f"rsi_{f['rsi_window']}", "macd", "macd_signal", "macd_hist"]
    columns += [f"roc_{f['roc_window']}"]
    columns += [f"close_sma_{w}" for w in f["sma_windows"]]
    columns += [f"close_ema_{w}" for w in f["ema_windows"]]
    columns += ["bb_pctb", "bb_bandwidth"]
    columns += [f"atr_{f['atr_window']}", f"rv_{f['realised_vol_window']}"]
    columns += [f"volume_change_{f['volume_change_window']}"]
    columns += [f"volume_ratio_{f['volume_ratio_window']}"]
    columns += [
        "market_ret",
        f"market_rv_{f['realised_vol_window']}",
        "vix",
        f"vix_change_{f['vix_change_window']}",
    ]
    columns += ["cash_rate", "bond_2y", "bond_10y", "term_spread"]
    return columns


def _rma(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing (used by RSI and ATR)."""
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def add_features(panel: pd.DataFrame, config: dict, save: bool = True) -> pd.DataFrame:
    """Append indicators and the next-day target to an aligned panel."""
    f = config["features"]
    df = panel.sort_values(["stock", "date"]).reset_index(drop=True).copy()
    df["price"] = df[config["data"]["price_field"]]
    by_stock = df.groupby("stock", sort=False)

    ret = by_stock["price"].pct_change()
    df["ret_1"] = ret
    for lag in f["return_lags"]:
        df[f"ret_lag_{lag}"] = ret.groupby(df["stock"]).shift(lag)

    window = f["rsi_window"]
    delta = by_stock["price"].diff()
    avg_gain = delta.clip(lower=0.0).groupby(df["stock"]).transform(lambda s: _rma(s, window))
    avg_loss = (-delta.clip(upper=0.0)).groupby(df["stock"]).transform(lambda s: _rma(s, window))
    df[f"rsi_{window}"] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    ema_fast = by_stock["price"].transform(
        lambda s: s.ewm(span=f["macd_fast"], adjust=False).mean()
    )
    ema_slow = by_stock["price"].transform(
        lambda s: s.ewm(span=f["macd_slow"], adjust=False).mean()
    )
    macd = ema_fast - ema_slow
    signal_line = macd.groupby(df["stock"]).transform(
        lambda s: s.ewm(span=f["macd_signal"], adjust=False).mean()
    )
    df["macd"] = macd
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd - signal_line

    df[f"roc_{f['roc_window']}"] = by_stock["price"].pct_change(f["roc_window"])

    for sma_window in f["sma_windows"]:
        sma = by_stock["price"].transform(
            lambda s: s.rolling(sma_window, min_periods=sma_window).mean()
        )
        df[f"close_sma_{sma_window}"] = df["price"] / sma - 1.0

    for ema_window in f["ema_windows"]:
        ema = by_stock["price"].transform(
            lambda s: s.ewm(span=ema_window, adjust=False, min_periods=ema_window).mean()
        )
        df[f"close_ema_{ema_window}"] = df["price"] / ema - 1.0

    band = f["bollinger_window"]
    mid = by_stock["price"].transform(lambda s: s.rolling(band, min_periods=band).mean())
    std = by_stock["price"].transform(lambda s: s.rolling(band, min_periods=band).std())
    upper = mid + f["bollinger_std"] * std
    lower = mid - f["bollinger_std"] * std
    df["bb_pctb"] = (df["price"] - lower) / (upper - lower)
    df["bb_bandwidth"] = (upper - lower) / mid

    prev_close = by_stock["price"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.groupby(df["stock"]).transform(lambda s: _rma(s, f["atr_window"]))
    df[f"atr_{f['atr_window']}"] = atr / df["price"]

    rv_window = f["realised_vol_window"]
    df[f"rv_{rv_window}"] = ret.groupby(df["stock"]).transform(
        lambda s: s.rolling(rv_window, min_periods=rv_window).std()
    )

    df[f"volume_change_{f['volume_change_window']}"] = by_stock["Volume"].pct_change(
        f["volume_change_window"]
    )
    volume_mean = by_stock["Volume"].transform(
        lambda s: s.rolling(f["volume_ratio_window"], min_periods=f["volume_ratio_window"]).mean()
    )
    df[f"volume_ratio_{f['volume_ratio_window']}"] = df["Volume"] / volume_mean

    next_price = by_stock["price"].shift(-1)
    target = (next_price > df["price"]).astype(float)
    target[next_price.isna()] = np.nan
    df["target"] = target

    columns = feature_columns(config)
    # Zero-volume days make some ratios undefined; treat them as missing, not infinite.
    df[columns] = df[columns].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=columns + ["target"]).reset_index(drop=True)

    if save:
        processed = Path(config["paths"]["processed"])
        df.to_parquet(processed / "model_table.parquet", index=False)
        (processed / "feature_columns.json").write_text(
            json.dumps({"features": columns}, indent=2), encoding="utf-8"
        )

        print(f"model table: {df.shape[0]} rows x {len(columns)} features")
        print(f"target balance: {df['target'].mean():.3f} up")
        print(f"date range: {df['date'].min().date()} .. {df['date'].max().date()}")
    return df


def run(config: dict) -> pd.DataFrame:
    """Full SP2: align then engineer."""
    return add_features(build_panel(config), config)


if __name__ == "__main__":
    run(load_config())

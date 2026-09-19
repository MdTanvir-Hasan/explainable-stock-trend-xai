"""M1 — SP1 data acquisition.

Downloads the raw inputs and reports coverage. Nothing is aligned or cleaned here;
that is M2. Raw files land in data/raw/ so later stages never re-download.

Sources:
    equities / index / VIX : Yahoo Finance (yfinance)
    RBA cash rate          : RBA F1  (daily, series FIRMMCRTD)
    AU government bonds    : RBA F2  (daily, 2y and 10y)
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from m0_setup.config import load_config

RBA_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _yahoo_symbol(symbol: str) -> str:
    """Append the ASX suffix unless the symbol is already an index/vix code."""
    return symbol if symbol.startswith("^") else f"{symbol}.AX"


def _raw_name(symbol: str) -> str:
    return f"{symbol.lstrip('^')}.csv"


def download_equities(config: dict) -> dict[str, pd.DataFrame]:
    """Download OHLCV for the stocks, index and VIX; save one CSV each."""
    raw_dir = Path(config["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    symbols = [_yahoo_symbol(s) for s in config["data"]["stocks"]]
    symbols += [config["data"]["index_symbol"], config["data"]["vix_symbol"]]

    meta_path = raw_dir / "_download_meta.json"
    expected = {
        "start_date": config["data"]["start_date"],
        "end_date": config["data"]["end_date"],
        "stocks": list(config["data"]["stocks"]),
        "index_symbol": config["data"]["index_symbol"],
        "vix_symbol": config["data"]["vix_symbol"],
    }
    cached_meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    stale = cached_meta is not None and cached_meta != expected
    if stale:
        print("download parameters changed: refreshing the raw snapshot")
    force = config["data"].get("force_download", False) or stale

    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = raw_dir / _raw_name(symbol)
        if path.exists() and not force:
            frames[symbol] = pd.read_csv(path, index_col=0, parse_dates=True)
            print(f"{symbol}: {len(frames[symbol])} rows cached")
            continue

        frame = yf.download(
            symbol,
            start=config["data"]["start_date"],
            end=config["data"]["end_date"],
            auto_adjust=False,
            progress=False,
        )
        if frame.empty:
            raise RuntimeError(f"no data returned for {symbol}")

        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        frame.to_csv(path)
        frames[symbol] = frame
        print(f"{symbol}: {len(frame)} rows saved")

    meta_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return frames


def _read_rba_csv(url: str) -> pd.DataFrame:
    """Parse an RBA statistical-table CSV (metadata rows, then date-indexed data)."""
    response = requests.get(url, timeout=60, headers=RBA_HEADERS)
    response.raise_for_status()
    lines = response.text.splitlines()

    titles = lines[1].split(",")
    first_data_row = next(
        i for i, line in enumerate(lines) if line.startswith("Series ID")
    ) + 1

    frame = pd.read_csv(
        io.StringIO("\n".join(lines[first_data_row:])),
        header=None,
        names=titles,
        on_bad_lines="skip",
    )
    frame = frame.rename(columns={titles[0]: "date"})
    frame["date"] = pd.to_datetime(frame["date"], format="%d-%b-%Y", errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date")

    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def download_macro(config: dict) -> pd.DataFrame:
    """Download the RBA cash rate and AU 2y/10y bond yields; save one CSV."""
    raw_dir = Path(config["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    cached = raw_dir / "rba_macro.csv"
    if cached.exists() and not config["data"].get("force_download", False):
        macro = pd.read_csv(cached, index_col=0, parse_dates=True)
        print(f"RBA macro: {len(macro)} rows cached")
        return macro

    cash = _read_rba_csv(config["data"]["rba_cash_rate_csv"])
    bonds = _read_rba_csv(config["data"]["rba_bond_yields_csv"])

    macro = pd.DataFrame(index=cash.index.union(bonds.index)).sort_index()
    macro["cash_rate"] = cash["Cash Rate Target"]
    macro["bond_2y"] = bonds["Australian Government 2 year bond"]
    macro["bond_10y"] = bonds["Australian Government 10 year bond"]
    macro.to_csv(cached)

    print(f"RBA macro: {len(macro)} rows saved")
    return macro


def coverage_report(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rows, date range and missing-value share for every raw series."""
    records = []
    for name, frame in frames.items():
        records.append(
            {
                "series": name,
                "rows": len(frame),
                "start": frame.index.min().date(),
                "end": frame.index.max().date(),
                "missing_pct": round(float(frame.isna().to_numpy().mean()) * 100, 2),
            }
        )
    return pd.DataFrame(records)


def run(config: dict) -> pd.DataFrame:
    """Download everything and write the coverage report to results/tables/."""
    equities = download_equities(config)
    macro = download_macro(config)

    report = coverage_report({**equities, "RBA_macro": macro})
    tables_dir = Path(config["paths"]["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(tables_dir / "data_coverage.csv", index=False)

    print()
    print(report.to_string(index=False))
    return report


if __name__ == "__main__":
    run(load_config())

"""M0 — configuration loading. The single source of runtime parameters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_KEYS = ("seed", "data", "paths", "features", "walk_forward", "models")


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`; scalar and list values are replaced."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path = "config.yaml", overrides: dict | None = None) -> dict[str, Any]:
    """Return the parsed config, failing loudly on missing required sections.

    `overrides` lets a notebook expose editable parameters without editing
    config.yaml; nested sections are merged so only the changed keys are touched.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if overrides:
        _merge(config, overrides)

    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")

    # Anchor output paths to the config file so notebooks and the CLI agree
    # regardless of the working directory the interpreter was launched from.
    base = config_path.resolve().parent
    config["paths"] = {
        key: str((base / value).resolve()) for key, value in config["paths"].items()
    }
    return config

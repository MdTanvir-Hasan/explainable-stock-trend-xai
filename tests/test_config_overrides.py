"""Notebook parameter overrides must merge without clobbering sibling keys."""
from __future__ import annotations

from pathlib import Path

from m0_setup.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_overrides_replace_values():
    config = load_config(ROOT / "config.yaml", {"seed": 7, "walk_forward": {"n_folds": 3}})
    assert config["seed"] == 7
    assert config["walk_forward"]["n_folds"] == 3


def test_overrides_do_not_clobber_siblings():
    config = load_config(ROOT / "config.yaml", {"walk_forward": {"n_folds": 3}})
    assert config["walk_forward"]["test_size"] == 120
    assert config["walk_forward"]["val_size"] == 60


def test_list_override_replaces_whole_list():
    config = load_config(ROOT / "config.yaml", {"data": {"stocks": ["BHP", "CBA"]}})
    assert config["data"]["stocks"] == ["BHP", "CBA"]
    assert config["data"]["index_symbol"] == "^AXJO"


def test_paths_remain_absolute_after_override():
    config = load_config(ROOT / "config.yaml", {"seed": 1})
    assert Path(config["paths"]["raw"]).is_absolute()

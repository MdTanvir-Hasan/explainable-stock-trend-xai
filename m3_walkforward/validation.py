"""M3 — SP3a chronological walk-forward splits.

Expanding-window walk-forward: for each fold the training set is everything before
a validation block, the validation block is immediately before a non-overlapping
test block. Fold k's test block comes after fold k-1's, and the final fold ends on
the last available date.

No random splitting is performed anywhere in the project.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from m0_setup.config import load_config


@dataclass(frozen=True)
class Fold:
    """Date boundaries for one walk-forward fold."""

    index: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def walk_forward_folds(dates, config: dict) -> list[Fold]:
    """Return the configured number of expanding-window folds over `dates`."""
    unique = np.sort(pd.unique(pd.to_datetime(pd.Series(dates))))
    settings = config["walk_forward"]
    n_folds = settings["n_folds"]
    test_size = settings["test_size"]
    val_size = settings["val_size"]
    min_train = settings["min_train_size"]

    required = min_train + val_size + n_folds * test_size
    if len(unique) < required:
        raise ValueError(
            f"need at least {required} dates for {n_folds} folds, got {len(unique)}"
        )

    first_test_start = len(unique) - n_folds * test_size
    folds = []
    for k in range(n_folds):
        test_start = first_test_start + k * test_size
        folds.append(
            Fold(
                index=k,
                train=unique[: test_start - val_size],
                val=unique[test_start - val_size : test_start],
                test=unique[test_start : test_start + test_size],
            )
        )
    return folds


if __name__ == "__main__":
    config = load_config()
    panel = pd.read_parquet(f"{config['paths']['processed']}/model_table.parquet")
    for fold in walk_forward_folds(panel["date"], config):
        print(
            f"fold {fold.index}: train {len(fold.train)} "
            f"({pd.Timestamp(fold.train.min()).date()}..{pd.Timestamp(fold.train.max()).date()})  "
            f"val {len(fold.val)}  test {len(fold.test)} "
            f"({pd.Timestamp(fold.test.min()).date()}..{pd.Timestamp(fold.test.max()).date()})"
        )

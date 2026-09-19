"""Walk-forward folds must be chronological, disjoint and cover the final date."""
from __future__ import annotations

import numpy as np

from m3_walkforward.validation import walk_forward_folds

FOLD_SETTINGS = {"n_folds": 4, "test_size": 30, "val_size": 20, "min_train_size": 100}


def _config(synthetic_panel):
    _, config = synthetic_panel
    return {**config, "walk_forward": FOLD_SETTINGS}


def test_train_precedes_validation_precedes_test(synthetic_panel):
    panel, _ = synthetic_panel
    folds = walk_forward_folds(panel["date"], _config(synthetic_panel))

    for fold in folds:
        assert fold.train.max() < fold.val.min()
        assert fold.val.max() < fold.test.min()


def test_test_blocks_do_not_overlap_and_move_forward(synthetic_panel):
    panel, _ = synthetic_panel
    folds = walk_forward_folds(panel["date"], _config(synthetic_panel))

    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test.max() < later.test.min()
        assert later.index == earlier.index + 1


def test_training_window_expands(synthetic_panel):
    panel, _ = synthetic_panel
    folds = walk_forward_folds(panel["date"], _config(synthetic_panel))

    sizes = [len(fold.train) for fold in folds]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_last_fold_ends_on_final_date(synthetic_panel):
    panel, _ = synthetic_panel
    folds = walk_forward_folds(panel["date"], _config(synthetic_panel))

    assert folds[-1].test.max() == np.max(panel["date"].to_numpy())


def test_disjoint_sets(synthetic_panel):
    panel, _ = synthetic_panel
    folds = walk_forward_folds(panel["date"], _config(synthetic_panel))

    for fold in folds:
        assert set(fold.train).isdisjoint(fold.val)
        assert set(fold.train).isdisjoint(fold.test)
        assert set(fold.val).isdisjoint(fold.test)

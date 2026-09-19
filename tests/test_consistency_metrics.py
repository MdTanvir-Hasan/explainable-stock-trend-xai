"""Unit checks for the consistency and risk-coverage helpers."""
from __future__ import annotations

import numpy as np

from m8_consistency.consistency import _cosine, _marc, _top_k_overlap
from m8_consistency.diagnostic import risk_coverage


def test_identical_vectors_are_perfectly_consistent():
    a = np.array([0.1, 0.5, 0.4, 0.0])
    assert _top_k_overlap(a, a, 2) == 1.0
    assert np.isclose(_cosine(a, a), 1.0)
    assert _marc(a, a) == 0.0


def test_disjoint_top_k_overlap_is_zero():
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 1.0])
    assert _top_k_overlap(a, b, 1) == 0.0


def test_risk_coverage_at_full_coverage_is_overall_accuracy():
    correct = np.array([1, 1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    table = risk_coverage(score, correct, [0.2, 0.5, 1.0])

    assert np.isclose(table.loc[table["coverage"] == 1.0, "accuracy"].iloc[0], correct.mean())
    assert table["coverage"].is_monotonic_increasing


def test_risk_coverage_keeps_most_reliable_first():
    correct = np.array([1, 1, 0, 0])
    score = np.array([0.9, 0.8, 0.1, 0.0])
    table = risk_coverage(score, correct, [0.5])

    assert np.isclose(table["accuracy"].iloc[0], 1.0)

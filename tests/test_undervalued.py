"""Tests for undervalued.py — Phase 6 undervalued zone detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_module_imports():
    """Smoke test: src.undervalued imports without error."""
    from src import undervalued
    assert hasattr(undervalued, "PROJECT_ROOT")
    assert hasattr(undervalued, "DEFAULT_INPUT")
    assert hasattr(undervalued, "DEFAULT_OUTPUT")
    assert hasattr(undervalued, "DEFAULT_SUMMARY_OUTPUT")
    assert hasattr(undervalued, "DEFAULT_THRESHOLD")
    assert hasattr(undervalued, "DEFAULT_MIN_LINE_N")
    assert undervalued.DEFAULT_THRESHOLD == -1.5
    assert undervalued.DEFAULT_MIN_LINE_N == 30


def test_compute_mad_zscore_basic():
    """Known values produce expected z-scores."""
    from src.undervalued import compute_mad_zscore
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = compute_mad_zscore(values)
    assert len(z) == 5
    median = values.median()  # 3.0
    abs_dev = (values - median).abs()
    mad = abs_dev.median()  # median([2,1,0,1,2]) = 1.0
    expected = (values - median) / (1.4826 * mad)
    pd.testing.assert_series_equal(z, expected, check_names=False)


def test_compute_mad_zscore_zero_mad():
    """All-identical values produce z-scores of 0.0."""
    from src.undervalued import compute_mad_zscore
    values = pd.Series([5.0, 5.0, 5.0, 5.0])
    z = compute_mad_zscore(values)
    assert (z == 0.0).all()


def test_compute_mad_zscore_median_centered():
    """Median of z-scores is approximately 0."""
    from src.undervalued import compute_mad_zscore
    rng = np.random.default_rng(42)
    values = pd.Series(rng.normal(0, 1, 200))
    z = compute_mad_zscore(values)
    assert abs(z.median()) < 0.05


def test_compute_mad_zscore_preserves_index():
    """Output has same index as input."""
    from src.undervalued import compute_mad_zscore
    values = pd.Series([1.0, 2.0, 3.0], index=[10, 20, 30])
    z = compute_mad_zscore(values)
    assert list(z.index) == [10, 20, 30]

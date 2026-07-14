"""Tests for undervalued.py — Phase 6 undervalued zone detection."""
from __future__ import annotations

import json

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
    assert undervalued.MAD_CONSISTENCY == 1.4826


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


@pytest.fixture
def modeled_df():
    """Synthetic modeled DataFrame with 3 lines of varying size."""
    rng = np.random.default_rng(99)
    rows = []
    for line, n in [("Line A", 50), ("Line B", 40), ("Line C", 10)]:
        for i in range(n):
            rows.append({
                "listing_id": f"{line}_{i}",
                "residual_log": rng.normal(0, 0.3),
                "primary_line": line,
            })
    return pd.DataFrame(rows)


def test_detect_undervalued_adds_columns(modeled_df):
    """detect_undervalued adds residual_zscore and used_global_stats."""
    from src.undervalued import detect_undervalued
    result = detect_undervalued(modeled_df)
    assert "residual_zscore" in result.columns
    assert "used_global_stats" in result.columns


def test_detect_undervalued_per_line_large_groups(modeled_df):
    """Lines with n >= 30 use per-line stats (not global)."""
    from src.undervalued import detect_undervalued
    result = detect_undervalued(modeled_df, min_line_n=30)
    large_lines = result[result["primary_line"].isin(["Line A", "Line B"])]
    assert (large_lines["used_global_stats"] == False).all()


def test_detect_undervalued_sparse_fallback(modeled_df):
    """Lines with n < 30 use global stats."""
    from src.undervalued import detect_undervalued
    result = detect_undervalued(modeled_df, min_line_n=30)
    small_line = result[result["primary_line"] == "Line C"]
    assert (small_line["used_global_stats"] == True).all()


def test_detect_undervalued_preserves_row_count(modeled_df):
    """Output has same number of rows as input."""
    from src.undervalued import detect_undervalued
    result = detect_undervalued(modeled_df)
    assert len(result) == len(modeled_df)


@pytest.fixture
def zscore_df():
    """DataFrame with known z-scores for tier testing."""
    return pd.DataFrame({
        "listing_id": ["L1", "L2", "L3", "L4", "L5"],
        "residual_zscore": [-2.5, -1.8, -1.2, -0.5, 1.0],
    })


def test_assign_tiers_boundaries(zscore_df):
    """Correct tier for known z-score values."""
    from src.undervalued import assign_tiers
    result = assign_tiers(zscore_df, threshold=-1.5)
    expected = ["strong", "good", "borderline", "fair", "fair"]
    assert result["value_tier"].tolist() == expected


def test_assign_tiers_is_undervalued(zscore_df):
    """is_undervalued = True iff z <= threshold."""
    from src.undervalued import assign_tiers
    result = assign_tiers(zscore_df, threshold=-1.5)
    expected = [True, True, False, False, False]
    assert result["is_undervalued"].tolist() == expected


def test_assign_tiers_exact_boundary():
    """z exactly at threshold is 'good', z exactly at threshold+0.5 is 'borderline'."""
    from src.undervalued import assign_tiers
    df = pd.DataFrame({"residual_zscore": [-2.0, -1.5, -1.0]})
    result = assign_tiers(df, threshold=-1.5)
    assert result["value_tier"].tolist() == ["strong", "good", "borderline"]


def test_assign_tiers_adds_columns(zscore_df):
    """assign_tiers adds value_tier and is_undervalued columns."""
    from src.undervalued import assign_tiers
    result = assign_tiers(zscore_df)
    assert "value_tier" in result.columns
    assert "is_undervalued" in result.columns
    assert result["value_tier"].dtype == object


def test_compute_undervalued_by_pct():
    """Formula correct for undervalued listings."""
    from src.undervalued import compute_undervalued_by
    df = pd.DataFrame({
        "residual_log": [-0.3, -0.5],
        "is_undervalued": [True, True],
    })
    result = compute_undervalued_by(df)
    expected = (1 - np.exp(df["residual_log"])) * 100
    np.testing.assert_allclose(result["undervalued_by_pct"], expected, rtol=1e-6)


def test_compute_undervalued_by_zero_for_fair():
    """0.0 for non-undervalued listings."""
    from src.undervalued import compute_undervalued_by
    df = pd.DataFrame({
        "residual_log": [0.1, 0.5, -0.01],
        "is_undervalued": [False, False, False],
    })
    result = compute_undervalued_by(df)
    assert (result["undervalued_by_pct"] == 0.0).all()


def test_compute_undervalued_by_mixed():
    """Mixed undervalued and fair listings."""
    from src.undervalued import compute_undervalued_by
    df = pd.DataFrame({
        "residual_log": [-0.3, 0.2],
        "is_undervalued": [True, False],
    })
    result = compute_undervalued_by(df)
    assert result["undervalued_by_pct"].iloc[0] > 0
    assert result["undervalued_by_pct"].iloc[1] == 0.0


@pytest.fixture
def full_flagged_df():
    """DataFrame with all Phase 6 columns for summary testing."""
    return pd.DataFrame({
        "listing_id": [f"L{i}" for i in range(10)],
        "residual_log": [-0.5, -0.3, 0.1, 0.2, -0.4, 0.05, -0.6, 0.3, -0.1, 0.15],
        "residual_zscore": [-2.1, -1.6, 0.1, 0.3, -1.8, -0.1, -2.5, 0.5, -0.5, 0.2],
        "value_tier": ["strong", "good", "fair", "fair", "good", "fair", "strong", "borderline", "borderline", "fair"],
        "is_undervalued": [True, True, False, False, True, False, True, False, False, False],
        "undervalued_by_pct": [39.3, 25.9, 0.0, 0.0, 33.0, 0.0, 45.1, 0.0, 0.0, 0.0],
        "used_global_stats": [False]*5 + [True]*5,
        "primary_line": ["Line A"]*5 + ["Line B"]*5,
    })


def test_compute_summary_structure(full_flagged_df):
    """Output has global, lines, threshold keys."""
    from src.undervalued import compute_summary
    summary = compute_summary(full_flagged_df, threshold=-1.5, min_line_n=30)
    assert "global" in summary
    assert "lines" in summary
    assert "threshold" in summary
    assert summary["threshold"] == -1.5
    g = summary["global"]
    assert "n" in g
    assert "n_undervalued" in g
    assert "pct_undervalued" in g
    assert "median_residual_log" in g
    assert "mad_residual_log" in g


def test_compute_summary_per_line(full_flagged_df):
    """Per-line stats include n, n_undervalued, pct_undervalued."""
    from src.undervalued import compute_summary
    summary = compute_summary(full_flagged_df, threshold=-1.5, min_line_n=30)
    assert "Line A" in summary["lines"]
    assert "Line B" in summary["lines"]
    la = summary["lines"]["Line A"]
    assert la["n"] == 5
    assert la["n_undervalued"] == 3
    assert "used_global_stats" in la


def test_write_summary_creates_json(tmp_path, full_flagged_df):
    """write_summary saves valid JSON."""
    from src.undervalued import compute_summary, write_summary
    summary = compute_summary(full_flagged_df, threshold=-1.5, min_line_n=30)
    out_path = tmp_path / "summary.json"
    write_summary(summary, out_path)
    assert out_path.exists()
    with open(out_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["threshold"] == -1.5
    assert "lines" in loaded


def test_main_end_to_end(tmp_path):
    """End-to-end: main() reads parquet, writes enriched parquet + summary JSON."""
    from src.undervalued import main

    rng = np.random.default_rng(7)
    rows = []
    for line, n in [("Line A", 40), ("Line B", 35), ("Line C", 8)]:
        for i in range(n):
            rows.append({
                "listing_id": f"{line}_{i}",
                "residual_log": rng.normal(0, 0.3),
                "primary_line": line,
                "price_per_sqm": rng.uniform(50000, 200000),
            })
    df = pd.DataFrame(rows)

    input_path = tmp_path / "input.parquet"
    df.to_parquet(input_path, index=False)

    output_path = tmp_path / "output.parquet"
    summary_path = tmp_path / "summary.json"

    main([
        "--input", str(input_path),
        "--output", str(output_path),
        "--summary-output", str(summary_path),
    ])

    assert output_path.exists()
    assert summary_path.exists()

    result_df = pd.read_parquet(output_path)
    assert "residual_zscore" in result_df.columns
    assert "value_tier" in result_df.columns
    assert "is_undervalued" in result_df.columns
    assert "undervalued_by_pct" in result_df.columns
    assert len(result_df) == len(df)

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    assert "global" in summary
    assert "lines" in summary

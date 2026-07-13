"""Tests for model.py — Phase 5 price-decay model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_model_module_imports():
    """Smoke test: src.model imports without error."""
    from src import model
    assert hasattr(model, "PROJECT_ROOT")
    assert hasattr(model, "DEFAULT_INPUT")
    assert hasattr(model, "DEFAULT_CURVES_OUTPUT")
    assert hasattr(model, "DEFAULT_MODELED_OUTPUT")


@pytest.fixture
def sample_df():
    """Synthetic enriched DataFrame for model tests."""
    return pd.DataFrame({
        "listing_id": ["L1", "L2", "L3", "L4"],
        "name": ["Condo A", "Condo B", "Condo C", "Condo D"],
        "price_per_sqm": [100000.0, 150000.0, 80000.0, 120000.0],
        "nearest_station_km": [0.5, 0.3, 2.0, 1.0],
        "nearest_station_line": [
            "BTS Sukhumvit Line",
            "BTS Silom Line",
            "BTS Sukhumvit Line",
            "MRT Blue Line",
        ],
        "area_sqm_num": [35.0, 50.0, 30.0, 45.0],
        "bedrooms": [1, 2, 0, 1],
        "latitude": [13.75, 13.72, 13.74, 13.73],
        "longitude": [100.56, 100.53, 100.55, 100.54],
    })


@pytest.fixture
def sample_with_nulls():
    """DataFrame with some null values to test filtering."""
    return pd.DataFrame({
        "listing_id": ["L1", "L2", "L3"],
        "name": ["A", "B", "C"],
        "price_per_sqm": [100000.0, None, 80000.0],
        "nearest_station_km": [0.5, 0.3, None],
        "nearest_station_line": ["BTS Sukhumvit Line", "BTS Silom Line", None],
        "area_sqm_num": [35.0, 50.0, 30.0],
        "bedrooms": [1, 2, 0],
        "latitude": [13.75, 13.72, 13.74],
        "longitude": [100.56, 100.53, 100.55],
    })


def test_prepare_data_filters_nulls(sample_with_nulls):
    """Rows with null price/distance/line are dropped."""
    from src.model import prepare_data
    df_expanded, df_original = prepare_data(sample_with_nulls)
    assert len(df_original) == 1
    assert df_original["listing_id"].iloc[0] == "L1"


def test_prepare_data_log_transforms(sample_df):
    """log_price_per_sqm and log_area columns are created correctly."""
    from src.model import prepare_data
    df_expanded, df_original = prepare_data(sample_df)
    assert "log_price_per_sqm" in df_original.columns
    assert "log_area" in df_original.columns
    assert "distance_km" in df_original.columns
    np.testing.assert_allclose(
        df_original["log_price_per_sqm"].iloc[0], np.log(100000.0), rtol=1e-6
    )
    np.testing.assert_allclose(
        df_original["log_area"].iloc[0], np.log(35.0), rtol=1e-6
    )
    assert df_original["distance_km"].iloc[0] == 0.5


@pytest.fixture
def sample_with_interchange():
    """DataFrame with an interchange listing (two lines)."""
    return pd.DataFrame({
        "listing_id": ["L1", "L2", "L3"],
        "name": ["Normal", "Interchange", "Normal2"],
        "price_per_sqm": [100000.0, 150000.0, 80000.0],
        "nearest_station_km": [0.5, 0.3, 2.0],
        "nearest_station_line": [
            "BTS Sukhumvit Line",
            "Airport Rail Link, BTS Sukhumvit Line",
            "MRT Blue Line",
        ],
        "area_sqm_num": [35.0, 50.0, 30.0],
        "bedrooms": [1, 2, 0],
        "latitude": [13.75, 13.72, 13.74],
        "longitude": [100.56, 100.53, 100.55],
    })


def test_interchange_split_expanded(sample_with_interchange):
    """Interchange row becomes 2 rows in expanded df, non-interchange stays 1."""
    from src.model import prepare_data
    df_expanded, df_original = prepare_data(sample_with_interchange)
    assert len(df_expanded) == 4
    l2_rows = df_expanded[df_expanded["listing_id"] == "L2"]
    assert len(l2_rows) == 2
    lines = set(l2_rows["line"].tolist())
    assert lines == {"Airport Rail Link", "BTS Sukhumvit Line"}


def test_interchange_split_original(sample_with_interchange):
    """Original df has one row per listing with is_interchange flag."""
    from src.model import prepare_data
    df_expanded, df_original = prepare_data(sample_with_interchange)
    assert len(df_original) == 3
    assert df_original["is_interchange"].tolist() == [False, True, False]


def test_interchange_primary_line(sample_with_interchange):
    """primary_line is the first line in the comma-joined string."""
    from src.model import prepare_data
    df_expanded, df_original = prepare_data(sample_with_interchange)
    l2 = df_original[df_original["listing_id"] == "L2"].iloc[0]
    assert l2["primary_line"] == "Airport Rail Link"


def test_interchange_non_interchange_primary_line(sample_with_interchange):
    """Non-interchange listings have primary_line == their only line."""
    from src.model import prepare_data
    df_expanded, df_original = prepare_data(sample_with_interchange)
    l1 = df_original[df_original["listing_id"] == "L1"].iloc[0]
    assert l1["primary_line"] == "BTS Sukhumvit Line"
    assert l1["is_interchange"] == False


@pytest.fixture
def fittable_df():
    """DataFrame large enough for MixedLM to converge.
    30 rows across 3 lines with clear price-decay pattern."""
    rng = np.random.default_rng(42)
    lines = ["Line A"] * 10 + ["Line B"] * 10 + ["Line C"] * 10
    distances = np.concatenate([
        rng.uniform(0.1, 3.0, 10),
        rng.uniform(0.1, 3.0, 10),
        rng.uniform(0.1, 3.0, 10),
    ])
    log_price = 12.0 - 0.15 * distances + rng.normal(0, 0.1, 30)
    return pd.DataFrame({
        "listing_id": [f"L{i}" for i in range(30)],
        "log_price_per_sqm": log_price,
        "distance_km": distances,
        "log_area": np.log(rng.uniform(20, 60, 30)),
        "bedrooms": rng.choice([0, 1, 2], 30),
        "line": lines,
    })


def test_fit_mixedlm_converges(fittable_df):
    """fit_mixedlm returns a result object and method string."""
    from src.model import fit_mixedlm
    result, method = fit_mixedlm(
        "log_price_per_sqm ~ distance_km", fittable_df
    )
    assert result is not None
    assert method in ("lbfgs", "powell", "ols_fallback")
    assert hasattr(result, "fe_params")
    assert hasattr(result, "random_effects")


def test_fit_mixedlm_has_fixed_effects(fittable_df):
    """Fixed effects include intercept and distance_km."""
    from src.model import fit_mixedlm
    result, method = fit_mixedlm(
        "log_price_per_sqm ~ distance_km", fittable_df
    )
    assert "Intercept" in result.fe_params.index
    assert "distance_km" in result.fe_params.index
    assert result.fe_params["distance_km"] < 0

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

"""Tests for model.py — Phase 5 price-decay model."""
from __future__ import annotations

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


def test_fallback_ols(fittable_df):
    """When MixedLM constructor raises, falls back to OLS without crashing."""
    from unittest.mock import patch
    import statsmodels.formula.api as smf
    from src.model import fit_mixedlm

    original = smf.mixedlm
    def mock_constructor(*args, **kwargs):
        raise RuntimeError("Simulated MixedLM failure")
    try:
        smf.mixedlm = mock_constructor
        result, method = fit_mixedlm("log_price_per_sqm ~ distance_km", fittable_df)
    finally:
        smf.mixedlm = original
    assert method == "ols_fallback"
    assert hasattr(result, "fe_params")
    assert hasattr(result, "random_effects")
    assert hasattr(result, "fittedvalues")
    assert result.converged == True
    assert "Intercept" in result.fe_params.index
    assert "distance_km" in result.fe_params.index
    assert result.fe_params["distance_km"] < 0
    assert len(result.random_effects) == 3


def test_fit_model_a_output_structure(fittable_df):
    """Model A output has global, lines, converged, method keys."""
    from src.model import fit_model_a
    curves = fit_model_a(fittable_df)
    assert "model" in curves
    assert curves["model"] == "A"
    assert "method" in curves
    assert "converged" in curves
    assert "global" in curves
    assert "intercept" in curves["global"]
    assert "slope" in curves["global"]
    assert "r_squared" in curves["global"]
    assert "lines" in curves
    for line_name, line_data in curves["lines"].items():
        assert "n" in line_data
        assert "intercept" in line_data
        assert "slope" in line_data
        assert "decay_pct_per_km" in line_data


def test_fit_model_a_per_line_coefficients(fittable_df):
    """Per-line intercept = fixed_intercept + random_intercept."""
    from src.model import fit_model_a
    curves = fit_model_a(fittable_df)
    assert 11.0 < curves["global"]["intercept"] < 13.0
    assert curves["global"]["slope"] < 0
    for line_data in curves["lines"].values():
        assert line_data["n"] == 10


def test_fit_model_b_adds_prediction_columns(fittable_df):
    """Model B adds predicted and residual columns to original df."""
    from src.model import fit_model_b
    df_original = fittable_df.copy()
    df_original["primary_line"] = df_original["line"]
    df_original["is_interchange"] = False
    df_original["price_per_sqm"] = np.exp(df_original["log_price_per_sqm"])

    result_df = fit_model_b(fittable_df, df_original)
    assert "predicted_log_price_per_sqm" in result_df.columns
    assert "predicted_price_per_sqm" in result_df.columns
    assert "residual_log" in result_df.columns
    assert "residual_pct" in result_df.columns


def test_fit_model_b_residual_pct_formula(fittable_df):
    """residual_pct = (actual_price / predicted_price) - 1."""
    from src.model import fit_model_b
    df_original = fittable_df.copy()
    df_original["primary_line"] = df_original["line"]
    df_original["is_interchange"] = False
    df_original["price_per_sqm"] = np.exp(df_original["log_price_per_sqm"])

    result_df = fit_model_b(fittable_df, df_original)
    for _, row in result_df.iterrows():
        expected_pct = (row["price_per_sqm"] / row["predicted_price_per_sqm"]) - 1
        np.testing.assert_allclose(row["residual_pct"], expected_pct, rtol=1e-4)


def test_fit_model_b_residual_log_formula(fittable_df):
    """residual_log = actual_log - predicted_log."""
    from src.model import fit_model_b
    df_original = fittable_df.copy()
    df_original["primary_line"] = df_original["line"]
    df_original["is_interchange"] = False
    df_original["price_per_sqm"] = np.exp(df_original["log_price_per_sqm"])

    result_df = fit_model_b(fittable_df, df_original)
    for _, row in result_df.iterrows():
        expected = row["log_price_per_sqm"] - row["predicted_log_price_per_sqm"]
        np.testing.assert_allclose(row["residual_log"], expected, rtol=1e-4)


def test_fit_model_b_one_row_per_listing(fittable_df):
    """Output has one row per listing (no interchange duplicates)."""
    from src.model import fit_model_b
    df_original = fittable_df.copy()
    df_original["primary_line"] = df_original["line"]
    df_original["is_interchange"] = False
    df_original["price_per_sqm"] = np.exp(df_original["log_price_per_sqm"])

    result_df = fit_model_b(fittable_df, df_original)
    assert len(result_df) == len(df_original)


def test_write_decay_curves_creates_json(tmp_path, fittable_df):
    """write_decay_curves saves valid JSON with expected structure."""
    from src.model import fit_model_a, write_decay_curves
    import json

    curves = fit_model_a(fittable_df)
    out_path = tmp_path / "decay_curves.json"
    write_decay_curves(curves, out_path)

    assert out_path.exists()
    with open(out_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["model"] == "A"
    assert "lines" in loaded
    assert "global" in loaded


def test_main_end_to_end(tmp_path, fittable_df):
    """End-to-end: main() reads parquet, writes decay_curves.json + listings_modeled.parquet."""
    from src.model import main

    df_for_save = fittable_df.copy()
    df_for_save["price_per_sqm"] = np.exp(df_for_save["log_price_per_sqm"])
    df_for_save["nearest_station_km"] = df_for_save["distance_km"]
    df_for_save["nearest_station_line"] = df_for_save["line"]
    df_for_save["area_sqm_num"] = np.exp(df_for_save["log_area"])
    df_for_save["listing_id"] = [f"L{i}" for i in range(len(df_for_save))]
    df_for_save["name"] = "Test"

    input_path = tmp_path / "input.parquet"
    df_for_save.to_parquet(input_path, index=False)

    curves_path = tmp_path / "decay_curves.json"
    modeled_path = tmp_path / "listings_modeled.parquet"

    main([
        "--input", str(input_path),
        "--curves-output", str(curves_path),
        "--modeled-output", str(modeled_path),
    ])

    assert curves_path.exists()
    assert modeled_path.exists()

    import json
    with open(curves_path, encoding="utf-8") as f:
        curves = json.load(f)
    assert curves["model"] == "A"
    assert len(curves["lines"]) > 0

    result_df = pd.read_parquet(modeled_path)
    assert "predicted_price_per_sqm" in result_df.columns
    assert "residual_pct" in result_df.columns
    assert "primary_line" in result_df.columns
    assert "is_interchange" in result_df.columns
    assert len(result_df) == len(df_for_save)

"""Tests for clean.py — Phase 3 data cleaning and geocoding cache."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.clean import (
    _parse_relative_date,
    _resolve_from_cache,
    load_and_clean,
)


def test_parse_relative_date_minutes():
    ref = datetime(2026, 7, 22, 12, 0, 0)
    result = _parse_relative_date("listed 21 minutes ago", ref)
    assert result == datetime(2026, 7, 22, 11, 39, 0)


def test_parse_relative_date_hours():
    ref = datetime(2026, 7, 22, 12, 0, 0)
    result = _parse_relative_date("updated 3 hours ago", ref)
    assert result == datetime(2026, 7, 22, 9, 0, 0)


def test_parse_relative_date_days():
    ref = datetime(2026, 7, 22, 12, 0, 0)
    result = _parse_relative_date("listed 2 days ago", ref)
    assert result == datetime(2026, 7, 20, 12, 0, 0)


def test_load_and_clean_dedup(tmp_path):
    input_path = tmp_path / "listings.parquet"
    df = pd.DataFrame({
        "listing_id": ["L1", "L1", "L2"],
        "latitude": [13.7, 13.7, 13.8],
        "longitude": [100.5, 100.5, 100.6],
        "price": ["1,000,000", "1,000,000", "2,000,000"],
        "first_price": ["900,000", "900,000", "1,900,000"],
        "area_sqm": ["35 sqm", "35 sqm", "50 sqm"],
        "bathrooms": [1, 1, 2],
        "listed_date": ["listed 2 days ago", "listed 2 days ago", "listed 1 week ago"],
        "updated_date": [None, None, None],
    })
    df.to_parquet(input_path, index=False)
    cleaned = load_and_clean(input_path)
    assert len(cleaned) == 2
    assert cleaned["listing_id"].tolist() == ["L1", "L2"]


def test_load_and_clean_coord_bounds_nulls_bad(tmp_path):
    input_path = tmp_path / "listings.parquet"
    df = pd.DataFrame({
        "listing_id": ["L1", "L2", "L3"],
        "latitude": [3.74, 13.7, 13.75],
        "longitude": [100.56, 100.5, 100.56],
        "price": ["1,000,000"] * 3,
        "first_price": [None] * 3,
        "area_sqm": ["35 sqm"] * 3,
        "bathrooms": [1] * 3,
        "listed_date": [None] * 3,
        "updated_date": [None] * 3,
    })
    df.to_parquet(input_path, index=False)
    cleaned = load_and_clean(input_path)
    assert pd.isna(cleaned.loc[0, "latitude"])
    assert pd.isna(cleaned.loc[0, "longitude"])
    assert cleaned.loc[1, "latitude"] == 13.7
    assert cleaned.loc[2, "latitude"] == 13.75


def test_load_and_clean_price_zero_nulled(tmp_path):
    input_path = tmp_path / "listings.parquet"
    df = pd.DataFrame({
        "listing_id": ["L1"],
        "latitude": [13.7],
        "longitude": [100.5],
        "price": ["0"],
        "first_price": ["0"],
        "area_sqm": ["35 sqm"],
        "bathrooms": [1],
        "listed_date": [None],
        "updated_date": [None],
    })
    df.to_parquet(input_path, index=False)
    cleaned = load_and_clean(input_path)
    assert pd.isna(cleaned.loc[0, "price_thb"])
    assert pd.isna(cleaned.loc[0, "first_price_thb"])


def test_load_and_clean_bathrooms_fillna(tmp_path):
    input_path = tmp_path / "listings.parquet"
    df = pd.DataFrame({
        "listing_id": ["L1"],
        "latitude": [13.7],
        "longitude": [100.5],
        "price": ["1,000,000"],
        "first_price": [None],
        "area_sqm": ["35 sqm"],
        "bathrooms": [None],
        "listed_date": [None],
        "updated_date": [None],
    })
    df.to_parquet(input_path, index=False)
    cleaned = load_and_clean(input_path)
    assert cleaned.loc[0, "bathrooms"] == 0


def test_resolve_from_cache_hit():
    cache = {"Condo A": [13.7, 100.5]}
    lat, lng, key = _resolve_from_cache(cache, "Condo A", "Bangkok")
    assert lat == 13.7
    assert lng == 100.5
    assert key == "Condo A"


def test_resolve_from_cache_name_miss_address_hit():
    """The key fix: a cached name-miss should not block the address hit."""
    cache = {"Condo A": None, "Bangkok": [13.7, 100.5]}
    lat, lng, key = _resolve_from_cache(cache, "Condo A", "Bangkok")
    assert lat == 13.7
    assert lng == 100.5
    assert key == "Bangkok"


def test_resolve_from_cache_all_miss():
    cache = {"Condo A": None, "Bangkok": None}
    lat, lng, key = _resolve_from_cache(cache, "Condo A", "Bangkok")
    assert lat is None
    assert lng is None
    assert key == "Condo A"


def test_resolve_from_cache_not_cached():
    cache = {}
    lat, lng, key = _resolve_from_cache(cache, "Condo A", "Bangkok")
    assert lat is None
    assert lng is None
    assert key is None


def test_resolve_from_cache_only_name():
    cache = {"Condo A": None}
    lat, lng, key = _resolve_from_cache(cache, "Condo A", None)
    assert lat is None
    assert lng is None
    assert key == "Condo A"


def test_resolve_from_cache_only_name_not_cached():
    cache = {}
    lat, lng, key = _resolve_from_cache(cache, "Condo A", None)
    assert lat is None
    assert lng is None
    assert key is None

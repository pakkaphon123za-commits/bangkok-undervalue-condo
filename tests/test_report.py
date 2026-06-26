"""Tests for report.py data-loading functions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.report import load_listings, load_stations


@pytest.fixture
def sample_enriched(tmp_path):
    """Create a tiny enriched parquet for testing."""
    data = {
        "listing_id": ["1", "2"],
        "name": ["Test Condo A", "Test Condo B"],
        "price_thb": [5000000.0, 3000000.0],
        "first_price_thb": [5100000.0, None],
        "area_sqm_num": [35.0, 24.0],
        "price_per_sqm": [142857.0, 125000.0],
        "bedrooms": [1, 1],
        "bathrooms": [1, 1],
        "detail_url": ["https://example.com/1", "https://example.com/2"],
        "address": ["Sukhumvit, Bangkok", "Silom, Bangkok"],
        "latitude": [13.75, 13.72],
        "longitude": [100.56, 100.53],
        "thumbnail": [None, None],
        "year_built": [None, None],
        "listed_dt": pd.to_datetime(["2026-04-01", "2026-05-01"]),
        "updated_dt": pd.to_datetime(["2026-05-01", "2026-06-01"]),
        "nearest_station": ["Phaya Thai", "Sala Daeng"],
        "nearest_station_km": [0.5, 0.3],
        "nearest_station_line": ["BTS Sukhumvit Line", "BTS Silom Line"],
    }
    path = tmp_path / "test_enriched.parquet"
    pd.DataFrame(data).to_parquet(path, index=False)
    return path


@pytest.fixture
def sample_stations(tmp_path):
    """Create a tiny geojson for testing."""
    import json
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [100.53, 13.75]},
                "properties": {
                    "name": "Phaya Thai", "name_th": "พญาไท",
                    "ref": "N2", "lines": ["BTS Sukhumvit Line"], "operational": True,
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [100.53, 13.72]},
                "properties": {
                    "name": "Sala Daeng", "name_th": "ศาลาแดง",
                    "ref": "S2", "lines": ["BTS Silom Line"], "operational": True,
                },
            },
        ],
    }
    path = tmp_path / "test_stations.geojson"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_listings_basic(sample_enriched):
    df = load_listings(sample_enriched)
    assert len(df) == 2
    assert "price_per_sqm" in df.columns
    assert "price_bin" in df.columns
    assert "dist_bin" in df.columns


def test_load_listings_quantile_bins(sample_enriched):
    df = load_listings(sample_enriched)
    assert df["price_bin"].dtype in ["int64", "int32", "Int64"]
    assert df["price_bin"].min() >= 1
    assert df["price_bin"].max() <= 4


def test_load_stations_basic(sample_stations):
    stations = load_stations(sample_stations)
    assert len(stations) == 2
    assert stations[0]["name"] == "Phaya Thai"
    assert stations[0]["name_th"] == "พญาไท"
    assert stations[0]["lat"] == 13.75
    assert stations[0]["lon"] == 100.53
    assert stations[0]["lines"] == ["BTS Sukhumvit Line"]

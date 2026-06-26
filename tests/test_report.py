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


from src.report import sort_stations_by_line



def test_sort_stations_single_line():
    """MRT Blue Line sorts by ref number."""
    stations = [
        {"name": "BL03", "ref": "BL03", "lines": ["MRT Blue Line"], "lat": 13.75, "lon": 100.47},
        {"name": "BL01", "ref": "BL01", "lines": ["MRT Blue Line"], "lat": 13.72, "lon": 100.47},
        {"name": "BL02", "ref": "BL02", "lines": ["MRT Blue Line"], "lat": 13.74, "lon": 100.47},
    ]
    by_line = sort_stations_by_line(stations)
    assert "MRT Blue Line" in by_line
    refs = [s["ref"] for s in by_line["MRT Blue Line"][0]]
    assert refs == ["BL01", "BL02", "BL03"]


def test_sort_stations_sukhumvit_splits_at_cen():
    """BTS Sukhumvit splits into N and E branches at CEN."""
    stations = [
        {"name": "N2", "ref": "N2", "lines": ["BTS Sukhumvit Line"], "lat": 13.75, "lon": 100.53},
        {"name": "CEN", "ref": "CEN", "lines": ["BTS Sukhumvit Line"], "lat": 13.74, "lon": 100.53},
        {"name": "E1", "ref": "E1", "lines": ["BTS Sukhumvit Line"], "lat": 13.74, "lon": 100.54},
        {"name": "N1", "ref": "N1", "lines": ["BTS Sukhumvit Line"], "lat": 13.75, "lon": 100.53},
        {"name": "E2", "ref": "E2", "lines": ["BTS Sukhumvit Line"], "lat": 13.74, "lon": 100.55},
    ]
    by_line = sort_stations_by_line(stations)
    suv = by_line["BTS Sukhumvit Line"]
    # Should produce 2 sub-lists (branches)
    assert len(suv) == 2
    # One branch starts with CEN, has N refs
    refs_0 = [s["ref"] for s in suv[0]]
    refs_1 = [s["ref"] for s in suv[1]]
    assert "CEN" in refs_0 or "CEN" in refs_1
    # N refs grouped together, E refs grouped together
    all_refs = refs_0 + refs_1
    n_idx = [all_refs.index(r) for r in all_refs if r.startswith("N")]
    e_idx = [all_refs.index(r) for r in all_refs if r.startswith("E")]
    assert max(n_idx) < min(e_idx) or max(e_idx) < min(n_idx)


def test_sort_stations_no_ref_falls_back_to_lat():
    """SRT lines with no ref sort by latitude."""
    stations = [
        {"name": "South", "ref": "", "lines": ["SRT Dark Red Line"], "lat": 13.70, "lon": 100.50},
        {"name": "North", "ref": "", "lines": ["SRT Dark Red Line"], "lat": 13.90, "lon": 100.50},
        {"name": "Mid", "ref": "", "lines": ["SRT Dark Red Line"], "lat": 13.80, "lon": 100.50},
    ]
    by_line = sort_stations_by_line(stations)
    srt = by_line["SRT Dark Red Line"]
    assert len(srt) == 1  # single branch (no split)
    lats = [s["lat"] for s in srt[0]]
    assert lats == sorted(lats, reverse=True)  # north to south


def test_sort_stations_digit_only_refs_fall_back_to_lat():
    stations = [
        {"name": "Station 10", "ref": "10", "lines": ["SRT Dark Red Line"], "lat": 13.70, "lon": 100.50, "operational": True},
        {"name": "Station 3", "ref": "3", "lines": ["SRT Dark Red Line"], "lat": 13.90, "lon": 100.50, "operational": True},
        {"name": "Station 4", "ref": "4", "lines": ["SRT Dark Red Line"], "lat": 13.80, "lon": 100.50, "operational": True},
    ]
    by_line = sort_stations_by_line(stations)
    srt = by_line["SRT Dark Red Line"]
    lats = [s["lat"] for s in srt[0]]
    assert lats == sorted(lats, reverse=True)  # north-to-south

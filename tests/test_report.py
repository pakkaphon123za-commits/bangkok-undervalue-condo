"""Tests for report.py data-loading functions."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import folium
import pandas as pd
import pytest

from src.report import inject_color_toggle, inject_lang_toggle, load_listings, load_stations


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


from src.report import build_popup_html, build_station_popup, build_transit_layer, sort_stations_by_line, build_listing_markers, build_ghost_markers



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


def test_build_station_popup():
    station = {
        "name": "Phaya Thai", "name_th": "พญาไท", "ref": "N2",
        "lines": ["BTS Sukhumvit Line"], "operational": True,
        "lat": 13.75, "lon": 100.53,
    }
    html = build_station_popup(station)
    assert "Phaya Thai" in html
    assert "พญาไท" in html
    assert "BTS Sukhumvit Line" in html
    assert "N2" in html
    assert "data-en" in html
    assert "data-th" in html


def test_build_transit_layer_returns_feature_group():
    stations = [
        {"name": "BL01", "name_th": "BL01", "ref": "BL01", "lines": ["MRT Blue Line"],
         "operational": True, "lat": 13.72, "lon": 100.47},
        {"name": "BL02", "name_th": "BL02", "ref": "BL02", "lines": ["MRT Blue Line"],
         "operational": True, "lat": 13.74, "lon": 100.47},
    ]
    by_line = sort_stations_by_line(stations)
    line_fgs = build_transit_layer(by_line, stations)
    assert isinstance(line_fgs, dict)
    assert "MRT Blue Line" in line_fgs
    assert hasattr(line_fgs["MRT Blue Line"], "add_to")


def test_build_popup_html_basic():
    row = pd.Series({
        "name": "Chewathai Residence Asoke",
        "price_thb": 5000000.0,
        "area_sqm_num": 35.56,
        "price_per_sqm": 140607.42,
        "bedrooms": 1,
        "bathrooms": 1,
        "year_built": "Dec 2016",
        "nearest_station": "Makkasan",
        "nearest_station_km": 0.145,
        "nearest_station_line": "Airport Rail Link",
        "listed_dt": pd.Timestamp("2026-04-01"),
        "detail_url": "https://www.fazwaz.com/property/123",
        "thumbnail": "https://cdn.fazwaz.com/img.jpg",
        "is_ghost": False,
    })
    html = build_popup_html(row)
    assert "Chewathai Residence Asoke" in html
    assert "5,000,000" in html
    assert "35.6" in html
    assert "140,607" in html
    assert "Makkasan" in html
    assert "0.145" in html
    assert "fazwaz.com/property/123" in html
    assert "cdn.fazwaz.com/img.jpg" in html
    assert "data-en" in html
    assert "GHOST" not in html


def test_build_popup_html_ghost():
    row = pd.Series({
        "name": "Old Condo",
        "price_thb": 3000000.0,
        "area_sqm_num": 40.0,
        "price_per_sqm": 75000.0,
        "bedrooms": 2,
        "bathrooms": 1,
        "year_built": None,
        "nearest_station": "Bang Wa",
        "nearest_station_km": 1.2,
        "nearest_station_line": "BTS Silom Line",
        "listed_dt": pd.Timestamp("2025-06-01"),
        "detail_url": "https://www.fazwaz.com/property/456",
        "thumbnail": None,
        "is_ghost": True,
    })
    html = build_popup_html(row, is_ghost=True)
    assert "GHOST" in html
    assert "days on market" in html or "วันที่ค้าง" in html


def test_popup_includes_tier_badge():
    from src.report import build_popup_html
    row = pd.Series({
        "name": "Condo", "price_thb": 3000000.0, "area_sqm_num": 30.0,
        "price_per_sqm": 100000.0, "bedrooms": 1, "bathrooms": 1,
        "year_built": None, "nearest_station": "Asok", "nearest_station_km": 0.3,
        "nearest_station_line": "BTS Sukhumvit Line", "listed_dt": pd.Timestamp("2026-05-01"),
        "detail_url": "https://example.com", "thumbnail": None, "is_ghost": False,
        "value_tier": "strong",
    })
    html = build_popup_html(row)
    assert "Strong value" in html
    assert "data-th" in html


def test_popup_undervalued_by_pct():
    from src.report import build_popup_html
    row = pd.Series({
        "name": "Condo", "price_thb": 3000000.0, "area_sqm_num": 30.0,
        "price_per_sqm": 100000.0, "bedrooms": 1, "bathrooms": 1,
        "year_built": None, "nearest_station": "Asok", "nearest_station_km": 0.3,
        "nearest_station_line": "BTS Sukhumvit Line", "listed_dt": pd.Timestamp("2026-05-01"),
        "detail_url": "https://example.com", "thumbnail": None, "is_ghost": False,
        "value_tier": "good", "is_undervalued": True, "undervalued_by_pct": 12.5,
    })
    html = build_popup_html(row)
    assert "Undervalued by" in html
    assert "12.5%" in html


def test_popup_hides_undervalued_when_fair():
    from src.report import build_popup_html
    row = pd.Series({
        "name": "Condo", "price_thb": 3000000.0, "area_sqm_num": 30.0,
        "price_per_sqm": 100000.0, "bedrooms": 1, "bathrooms": 1,
        "year_built": None, "nearest_station": "Asok", "nearest_station_km": 0.3,
        "nearest_station_line": "BTS Sukhumvit Line", "listed_dt": pd.Timestamp("2026-05-01"),
        "detail_url": "https://example.com", "thumbnail": None, "is_ghost": False,
        "value_tier": "fair", "is_undervalued": False, "undervalued_by_pct": 0.0,
    })
    html = build_popup_html(row)
    assert "Undervalued by" not in html
    assert "Fair value" in html


def test_build_popup_html_no_thumbnail():
    row = pd.Series({
        "name": "No Image Condo",
        "price_thb": 2000000.0,
        "area_sqm_num": 30.0,
        "price_per_sqm": 66666.67,
        "bedrooms": 1,
        "bathrooms": 1,
        "year_built": None,
        "nearest_station": "Asok",
        "nearest_station_km": 0.3,
        "nearest_station_line": "BTS Sukhumvit Line",
        "listed_dt": pd.Timestamp("2026-05-01"),
        "detail_url": "https://www.fazwaz.com/property/789",
        "thumbnail": None,
        "is_ghost": False,
    })
    html = build_popup_html(row)
    assert "<img" not in html
    assert "No Image Condo" in html


def test_build_listing_markers_returns_feature_group(sample_enriched):
    from src.report import build_listing_markers
    df = load_listings(sample_enriched)
    groups, color_data = build_listing_markers(df)
    assert isinstance(groups, dict)
    assert "Undervalued: strong" in groups
    assert "Undervalued: good" in groups
    assert "Undervalued: borderline" in groups
    assert "Other listings" in groups
    for fg in groups.values():
        assert fg is not None
        assert hasattr(fg, "add_to")
    assert isinstance(color_data, list)
    assert len(color_data) > 0
    for entry in color_data:
        assert "price" in entry
        assert "dist" in entry
        assert "line" in entry


def test_tier_featuregroups_exist(sample_enriched):
    from src.report import build_listing_markers
    df = load_listings(sample_enriched)
    groups, _ = build_listing_markers(df)
    expected = ["Undervalued: strong", "Undervalued: good", "Undervalued: borderline", "Other listings"]
    assert list(groups.keys()) == expected


def test_build_ghost_markers_empty_when_no_ghosts(sample_enriched):
    df = load_listings(sample_enriched)
    fg = build_ghost_markers(df)
    assert fg is not None
    assert hasattr(fg, "add_to")


def test_build_ghost_markers_with_ghosts():
    data = {
        "listing_id": ["1"], "name": ["Ghost Condo"],
        "price_thb": [1000000.0], "first_price_thb": [None],
        "area_sqm_num": [30.0], "price_per_sqm": [33333.0],
        "bedrooms": [1], "bathrooms": [1],
        "detail_url": ["https://example.com"], "address": ["Bangkok"],
        "latitude": [13.75], "longitude": [100.56],
        "thumbnail": [None], "year_built": [None],
        "listed_dt": pd.to_datetime(["2025-01-01"]),
        "updated_dt": pd.to_datetime(["2025-06-01"]),
        "nearest_station": ["Asok"], "nearest_station_km": [0.5],
        "nearest_station_line": ["BTS Sukhumvit Line"],
        "is_ghost": [True],
    }
    df = pd.DataFrame(data)
    fg = build_ghost_markers(df)
    assert fg is not None


def test_inject_color_toggle_adds_elements(sample_enriched, sample_stations):
    df = load_listings(sample_enriched)
    unique_lines = sorted(df["nearest_station_line"].dropna().unique())
    color_data = [{"price": "#2ecc71", "dist": "#2ecc71", "line": "#77BB44"}]
    m = folium.Map(location=[13.75, 100.56], zoom_start=12)

    inject_color_toggle(m, color_data, list(unique_lines))

    html_str = m.get_root().render()
    assert "colorMode" in html_str
    assert "recolorMarkers" in html_str


def test_inject_lang_toggle_adds_button():
    m = folium.Map(location=[13.75, 100.56], zoom_start=12)
    inject_lang_toggle(m)

    html_str = m.get_root().render()
    assert "langToggle" in html_str
    assert "data-en" in html_str
    assert "data-th" in html_str
    assert "switchLang" in html_str


def test_main_generates_html(sample_enriched, sample_stations, tmp_path):
    from src.report import DEFAULT_INPUT

    output_path = tmp_path / "index.html"
    result = subprocess.run(
        [
            sys.executable, "src/report.py",
            "--input", str(sample_enriched),
            "--stations", str(sample_stations),
            "--output", str(output_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "folium" in content.lower() or "leaflet" in content.lower()
    assert "colorMode" in content
    assert "langToggle" in content
    assert "Test Condo A" in content
    assert "listings_modeled.parquet" in str(DEFAULT_INPUT)
    assert DEFAULT_INPUT != sample_enriched


def test_default_input_is_modeled_parquet():
    from src.report import DEFAULT_INPUT
    assert "listings_modeled.parquet" in str(DEFAULT_INPUT)


@pytest.fixture
def real_enriched():
    path = Path("data/interim/listings_sample100_enriched.parquet")
    if not path.exists():
        pytest.skip("Sample enriched parquet not found")
    return path


@pytest.fixture
def real_stations():
    path = Path("data/processed/stations.geojson")
    if not path.exists():
        pytest.skip("stations.geojson not found")
    return path


def test_integration_full_map(real_enriched, real_stations, tmp_path):
    """End-to-end: real data → HTML output with all features."""
    output_path = tmp_path / "index.html"
    result = subprocess.run(
        [
            sys.executable, "src/report.py",
            "--input", str(real_enriched),
            "--stations", str(real_stations),
            "--output", str(output_path),
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")
    assert "leaflet" in content.lower()
    assert "colorMode" in content
    assert "langToggle" in content
    assert "recolorMarkers" in content
    assert "switchLang" in content
    assert "BTS Sukhumvit" in content or "Sukhumvit" in content
    assert "data-th" in content
    assert "พญาไท" in content or "สุรศักดิ์" in content


def test_narrative_panel_when_files_present(tmp_path):
    from src.report import _markdown_to_html, inject_narrative_panel
    import folium
    m = folium.Map(location=[13.75, 100.56], zoom_start=12)
    narrative_md = "# Brief\n\nSummary text.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n"
    html = _markdown_to_html(narrative_md)
    assert "<h1>Brief</h1>" in html
    assert "<p>Summary text.</p>" in html
    assert "<table" in html
    assert "<strong>" not in html  # no bold in this input

    inject_narrative_panel(m, html, [{"name": "Line 1", "n": 100, "pct_undervalued": 10.0, "decay_pct_per_km": -15.0}])
    rendered = m.get_root().render()
    assert "narrativePanel" in rendered
    assert "narrativeToggle" in rendered
    assert "openNarrative" in rendered
    assert "closeNarrative" in rendered
    assert "Line 1" in rendered


def test_no_narrative_panel_when_files_absent():
    from src.report import inject_narrative_panel
    import folium
    m = folium.Map(location=[13.75, 100.56], zoom_start=12)
    inject_narrative_panel(m, "", [])
    rendered = m.get_root().render()
    assert "narrativePanel" not in rendered
    assert "narrativeToggle" not in rendered


def test_backward_compat_missing_value_tier():
    from src.report import build_listing_markers, build_popup_html
    df = pd.DataFrame({
        "listing_id": ["1"], "name": ["Old Condo"], "price_thb": [3000000.0],
        "first_price_thb": [None], "area_sqm_num": [30.0], "price_per_sqm": [100000.0],
        "bedrooms": [1], "bathrooms": [1], "detail_url": ["https://example.com"],
        "address": ["Bangkok"], "latitude": [13.75], "longitude": [100.56],
        "thumbnail": [None], "year_built": [None],
        "listed_dt": pd.to_datetime(["2026-05-01"]),
        "updated_dt": pd.to_datetime(["2026-06-01"]),
        "nearest_station": ["Asok"], "nearest_station_km": [0.3],
        "nearest_station_line": ["BTS Sukhumvit Line"], "is_ghost": [False],
    })
    groups, color_data = build_listing_markers(df)
    assert list(groups.keys()) == ["Undervalued: strong", "Undervalued: good", "Undervalued: borderline", "Other listings"]
    assert len(list(groups["Other listings"]._children.keys())) == 1
    assert len(color_data) == 1
    html = build_popup_html(df.iloc[0])
    assert "Undervalued by" not in html


def test_app_shell_elements_present(sample_enriched, sample_stations, tmp_path):
    output_path = tmp_path / "index.html"
    result = subprocess.run(
        [
            sys.executable, "src/report.py",
            "--input", str(sample_enriched),
            "--stations", str(sample_stations),
            "--output", str(output_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    content = output_path.read_text(encoding="utf-8")
    assert "appHeader" in content
    assert "filterBar" in content
    assert "appSidebar" in content
    assert "listingPanel" in content
    assert "analysisPanel" in content
    assert "appFooter" in content
    assert "langToggleContainer" in content
    assert "narrativeToggleContainer" in content


def test_price_bubble_markers_and_clustering(sample_enriched, sample_stations, tmp_path):
    output_path = tmp_path / "index.html"
    result = subprocess.run(
        [
            sys.executable, "src/report.py",
            "--input", str(sample_enriched),
            "--stations", str(sample_stations),
            "--output", str(output_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    content = output_path.read_text(encoding="utf-8")
    assert "listing-marker" in content
    assert "price-bubble" in content
    assert "MarkerCluster" in content


def test_fonts_loaded(sample_enriched, sample_stations, tmp_path):
    output_path = tmp_path / "index.html"
    result = subprocess.run(
        [
            sys.executable, "src/report.py",
            "--input", str(sample_enriched),
            "--stations", str(sample_stations),
            "--output", str(output_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    content = output_path.read_text(encoding="utf-8")
    assert "IBM+Plex" in content or "IBM%20Plex" in content
    assert "IBM Plex" in content

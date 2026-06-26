# Folium Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/report.py` that generates `docs/index.html` — an interactive folium map of Bangkok condo listings with transit network overlay, color-toggle modes, TH/EN language toggle, and ghost listing layer.

**Architecture:** Single Python script reads enriched parquet + stations geojson, builds a folium Map with three FeatureGroups (transit network, listings, ghosts), injects custom JS for color-toggle recoloring and TH/EN language switching, and saves a self-contained HTML file.

**Tech Stack:** Python 3.14, folium 0.20.0, pandas, numpy (already installed)

## Global Constraints

- Run from repo root: `python3 src/report.py` (not `python`)
- Import convention: `from __future__ import annotations` + `Path(__file__).resolve().parent.parent` for PROJECT_ROOT
- No external CSS/JS at serve time — folium embeds everything inline
- folium 0.20.0 installed, branca 0.8.2 installed
- Output: `docs/index.html` — single file, GitHub Pages ready
- Graceful degradation: if `is_ghost` column missing, ghost layer empty + toggle hidden
- No comments in code unless requested

---

## File Structure

```
src/report.py          — new file, all map-building logic
tests/test_report.py   — new file, unit tests for data-processing functions
docs/index.html        — generated output (gitignored? no — this IS the website)
```

`src/report.py` is organized into these functions:
- `load_listings(path) -> pd.DataFrame` — reads parquet, computes quantile bins
- `load_stations(path) -> list[dict]` — reads geojson into station dicts
- `sort_stations_by_line(stations) -> dict[str, list[list[dict]]]` — orders stations per line for polylines
- `build_transit_layer(stations_by_line, all_stations) -> folium.FeatureGroup` — polylines + station dots
- `build_listing_markers(df) -> tuple[folium.FeatureGroup, list[dict]]` — CircleMarkers + color data
- `build_ghost_markers(df) -> folium.FeatureGroup` — hollow red rings
- `inject_color_toggle(m, color_data, unique_lines) -> None` — `<select>` + JS recolor + legend
- `inject_lang_toggle(m) -> None` — TH/EN button + JS text swap
- `build_popup_html(row, is_ghost=False) -> str` — popup HTML for a listing
- `build_station_popup(station) -> str` — popup HTML for a station
- `main()` — assembles everything, saves to docs/index.html

---

### Task 1: Skeleton + load functions + CLI

**Files:**
- Create: `src/report.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Produces: `load_listings(path: Path) -> pd.DataFrame`, `load_stations(path: Path) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.report'"

- [ ] **Step 3: Write minimal implementation**

```python
# src/report.py
"""Build interactive folium map of Bangkok condo listings.

Generates docs/index.html — a self-contained map with transit network overlay,
color-toggle modes (price/distance/line), TH/EN language toggle, and ghost
listing layer. Ready for GitHub Pages.

Usage:
    python3 src/report.py
    python3 src/report.py --input data/interim/listings_enriched.parquet
    python3 src/report.py --output docs/index.html
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "listings_enriched.parquet"
DEFAULT_STATIONS = PROJECT_ROOT / "data" / "processed" / "stations.geojson"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "index.html"

LINE_COLORS = {
    "BTS Sukhumvit Line": "#77BB44",
    "BTS Silom Line": "#007A33",
    "BTS Gold Line": "#F2A900",
    "MRT Blue Line": "#1B4F9C",
    "MRT Purple Line": "#9B26B6",
    "MRT Yellow Line": "#FFC20E",
    "MRT Pink Line": "#EC008C",
    "MRT Orange Line": "#FF6600",
    "Airport Rail Link": "#A0282C",
    "SRT Dark Red Line": "#8B1A1A",
    "SRT Light Red Line": "#D32F2F",
}

BIN_COLORS = ["#2ecc71", "#1abc9c", "#3498db", "#f1c40f"]

LINE_NAMES_TH = {
    "BTS Sukhumvit Line": "รถไฟฟ้า BTS สายสุขุมวิท",
    "BTS Silom Line": "รถไฟฟ้า BTS สายสีลม",
    "BTS Gold Line": "รถไฟฟ้า BTS สายทอง",
    "MRT Blue Line": "รถไฟใต้ดิน MRT สายสีน้ำเงิน",
    "MRT Purple Line": "รถไฟใต้ดิน MRT สายสีม่วง",
    "MRT Yellow Line": "รถไฟฟ้า MRT สายสีเหลือง",
    "MRT Pink Line": "รถไฟฟ้า MRT สายสีชมพู",
    "MRT Orange Line": "รถไฟใต้ดิน MRT สายสีส้ม",
    "Airport Rail Link": "รถไฟฟ้า Airport Rail Link",
    "SRT Dark Red Line": "รถไฟชานเมืองสายสีแดงเข้ม",
    "SRT Light Red Line": "รถไฟชานเมืองสายสีแดงอ่อน",
}

UI_LABELS = {
    "price_per_sqm": ("Price per sqm", "ราคา/ตร.ม."),
    "distance_to_station": ("Distance to station", "ระยะจากสถานี"),
    "by_transit_line": ("By transit line", "ตามสายรถไฟ"),
    "budget": ("Budget", "ประหยัด"),
    "below_median": ("Below median", "ต่ำกว่ามัธยะ"),
    "above_median": ("Above median", "สูงกว่ามัธยะ"),
    "premium": ("Premium", "ระดับพรีเมียม"),
    "walk": ("Walk", "เดินถึง"),
    "short_ride": ("Short ride", "นั่งรถสั้นๆ"),
    "transit_adjacent": ("Transit-adjacent", "ใกล้รถไฟ"),
    "far": ("Far", "ไกล"),
    "transit_network": ("Transit Network", "เครือข่ายรถไฟ"),
    "ghost_listings": ("Ghost Listings", "ป้ายขายที่ค้าง"),
    "price": ("Price", "ราคา"),
    "area": ("Area", "พื้นที่"),
    "price_per_sqm_label": ("Price/sqm", "ราคา/ตร.ม."),
    "beds_baths": ("Beds/Baths", "ห้องนอน/ห้องน้ำ"),
    "year_built": ("Year built", "ปีที่สร้าง"),
    "nearest": ("Nearest", "สถานีใกล้สุด"),
    "line": ("Line", "สาย"),
    "listed": ("Listed", "ลงป้ายเมื่อ"),
    "view_on_fazwaz": ("View on FazWaz", "ดูบน FazWaz"),
    "ghost_badge": ("GHOST", "ค้างนาน"),
    "days_on_market": ("days on market", "วันที่ค้าง"),
    "ref": ("Ref", "รหัส"),
    "status": ("Status", "สถานะ"),
    "operational": ("Operational", "เปิดให้บริการ"),
    "planned": ("Planned", "กำลังก่อสร้าง"),
}


def load_listings(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.dropna(subset=["latitude", "longitude"]).copy()

    for col in ["price_per_sqm", "nearest_station_km"]:
        if col not in df.columns:
            df[col] = np.nan

    df["price_bin"] = 1
    valid_price = df["price_per_sqm"].notna()
    if valid_price.sum() >= 4:
        df.loc[valid_price, "price_bin"] = pd.qcut(
            df.loc[valid_price, "price_per_sqm"], q=4, labels=False, duplicates="drop"
        ) + 1
    df["price_bin"] = df["price_bin"].astype(int)

    df["dist_bin"] = 1
    valid_dist = df["nearest_station_km"].notna()
    if valid_dist.sum() >= 4:
        df.loc[valid_dist, "dist_bin"] = pd.qcut(
            df.loc[valid_dist, "nearest_station_km"], q=4, labels=False, duplicates="drop"
        ) + 1
    df["dist_bin"] = df["dist_bin"].astype(int)

    if "is_ghost" not in df.columns:
        df["is_ghost"] = False

    if "thumbnail" not in df.columns:
        df["thumbnail"] = None
    if "year_built" not in df.columns:
        df["year_built"] = None
    if "first_price_thb" not in df.columns:
        df["first_price_thb"] = None

    return df


def load_stations(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    stations = []
    for feat in data["features"]:
        lon, lat = feat["geometry"]["coordinates"]
        props = feat["properties"]
        stations.append(
            {
                "name": props["name"],
                "name_th": props.get("name_th", props["name"]),
                "ref": props.get("ref", ""),
                "lines": props.get("lines", []),
                "operational": props.get("operational", True),
                "lat": lat,
                "lon": lon,
            }
        )
    return stations


def main() -> None:
    parser = argparse.ArgumentParser(description="Build folium map of listings")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}")
        return
    if not args.stations.exists():
        print(f"Stations not found: {args.stations}")
        return

    print(f"Loading listings: {args.input}")
    df = load_listings(args.input)
    print(f"  {len(df)} listings with coords")

    print(f"Loading stations: {args.stations}")
    stations = load_stations(args.stations)
    print(f"  {len(stations)} stations")

    print("Map not yet implemented — skeleton only")
    print(f"Would save to: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/report.py tests/test_report.py
git commit -m "Phase 8: report.py skeleton with load_listings + load_stations + tests"
```

---

### Task 2: Station polyline ordering

**Files:**
- Modify: `src/report.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Consumes: `load_stations()` from Task 1
- Produces: `sort_stations_by_line(stations: list[dict]) -> dict[str, list[dict]]`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_report.py
import re
from src.report import sort_stations_by_line


def _ref_num(ref: str) -> int:
    m = re.search(r"(\d+)", ref)
    return int(m.group(1)) if m else 999


def test_sort_stations_single_line():
    """MRT Blue Line sorts by ref number."""
    stations = [
        {"name": "BL03", "ref": "BL03", "lines": ["MRT Blue Line"], "lat": 13.75, "lon": 100.47},
        {"name": "BL01", "ref": "BL01", "lines": ["MRT Blue Line"], "lat": 13.72, "lon": 100.47},
        {"name": "BL02", "ref": "BL02", "lines": ["MRT Blue Line"], "lat": 13.74, "lon": 100.47},
    ]
    by_line = sort_stations_by_line(stations)
    assert "MRT Blue Line" in by_line
    refs = [s["ref"] for s in by_line["MRT Blue Line"]]
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report.py::test_sort_stations_single_line -v`
Expected: FAIL with "ImportError: cannot import name 'sort_stations_by_line'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/report.py` after `load_stations()`:

```python
def _ref_sort_key(ref: str) -> tuple[str, int]:
    """Sort key for ref codes: prefix (string) then number."""
    m = re.match(r"([A-Z]+)(\d+)", ref)
    if m:
        return (m.group(1), int(m.group(2)))
    return ("ZZZ", 999)


def sort_stations_by_line(
    stations: list[dict[str, Any]],
) -> dict[str, list[list[dict[str, Any]]]]:
    """Group stations by line, ordered for polyline drawing.

    Returns dict[line_name, list[branches]] where each branch is an ordered
    list of station dicts. Sukhumvit/Silom split at CEN into 2 branches.
    Lines without refs fall back to latitude sort (north to south).
    """
    import re as _re

    line_map: dict[str, list[dict[str, Any]]] = {}
    for s in stations:
        for line in s["lines"]:
            line_map.setdefault(line, []).append(s)

    result: dict[str, list[list[dict[str, Any]]]] = {}
    for line, stns in line_map.items():
        refs = [s["ref"] for s in stns]
        has_cen = any(r == "CEN" for r in refs)
        has_n = any(r.startswith("N") and r != "CEN" for r in refs if r)
        has_e = any(r.startswith("E") and r != "CEN" for r in refs if r)

        if has_cen and (has_n and has_e):
            branches = _split_bts_branches(stns)
        elif line == "BTS Silom Line" and has_cen:
            branches = _split_bts_branches(stns)
        else:
            any_ref = any(r for r in refs)
            if any_ref:
                sorted_stns = sorted(stns, key=lambda s: _ref_sort_key(s["ref"]))
                branches = [sorted_stns]
            else:
                sorted_stns = sorted(stns, key=lambda s: -s["lat"])
                branches = [sorted_stns]

        result[line] = branches

    return result


def _split_bts_branches(
    stns: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Split BTS Sukhumvit/Silom stations into branches at CEN."""
    cen = next((s for s in stns if s["ref"] == "CEN"), None)
    if not cen:
        return [sorted(stns, key=lambda s: _ref_sort_key(s["ref"]))]

    branches: list[list[dict[str, Any]]] = []
    prefixes_seen: set[str] = set()

    for s in stns:
        if s["ref"] == "CEN":
            continue
        prefix = ""
        m = re.match(r"([A-Z]+)", s["ref"])
        if m:
            prefix = m.group(1)
        if prefix not in prefixes_seen:
            prefixes_seen.add(prefix)
            branch = [cen] + sorted(
                [x for x in stns if x["ref"].startswith(prefix) and x["ref"] != "CEN"],
                key=lambda x: _ref_sort_key(x["ref"]),
            )
            branches.append(branch)

    return branches
```

Also add `import re` at the top of `src/report.py` (after `import json`).

Actually, `import re` is already included in Task 1's imports. Proceed to run tests.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/report.py tests/test_report.py
git commit -m "Phase 8: station polyline ordering with BTS branch splitting"
```

---

### Task 3: Transit network layer (polylines + station dots)

**Files:**
- Modify: `src/report.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Consumes: `load_stations()`, `sort_stations_by_line()` from Tasks 1-2
- Produces: `build_transit_layer(stations_by_line, stations) -> folium.FeatureGroup`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_report.py
from src.report import build_station_popup, build_transit_layer, sort_stations_by_line


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
    fg = build_transit_layer(by_line, stations)
    assert fg is not None
    assert hasattr(fg, "add_to")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report.py::test_build_station_popup -v`
Expected: FAIL with "ImportError: cannot import name 'build_station_popup'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/report.py`:

```python
def build_station_popup(station: dict[str, Any]) -> str:
    lines_en = ", ".join(station["lines"])
    lines_th = ", ".join(LINE_NAMES_TH.get(l, l) for l in station["lines"])
    status_en = "Operational" if station["operational"] else "Planned"
    status_th = "เปิดให้บริการ" if station["operational"] else "กำลังก่อสร้าง"

    return f"""<div style="font-family: sans-serif; font-size: 12px; min-width: 150px;">
<b><span data-en="{station['name']}" data-th="{station['name_th']}">{station['name']}</span></b><br>
<span data-en="{lines_en}" data-th="{lines_th}">{lines_en}</span><br>
<span data-en="Ref: {station['ref']}" data-th="รหัส: {station['ref']}">Ref: {station['ref']}</span><br>
<span data-en="Status: {status_en}" data-th="สถานะ: {status_th}">Status: {status_en}</span>
</div>"""


def build_transit_layer(
    stations_by_line: dict[str, list[list[dict[str, Any]]]],
    all_stations: list[dict[str, Any]],
) -> folium.FeatureGroup:
    fg = folium.FeatureGroup(name="transit_network")

    for line_name, branches in stations_by_line.items():
        color = LINE_COLORS.get(line_name, "#888888")
        for branch in branches:
            if len(branch) < 2:
                continue
            coords = [(s["lat"], s["lon"]) for s in branch]
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=3,
                opacity=0.6,
            ).add_to(fg)

    for station in all_stations:
        first_line = station["lines"][0] if station["lines"] else ""
        color = LINE_COLORS.get(first_line, "#888888")
        folium.CircleMarker(
            location=(station["lat"], station["lon"]),
            radius=4,
            color=color,
            weight=1.5,
            fill=True,
            fillColor="#ffffff",
            fillOpacity=1.0,
            popup=folium.Popup(build_station_popup(station), max_width=250),
            tooltip=station["name"],
        ).add_to(fg)

    return fg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report.py -v`
Expected: 8 PASS

- [ ] **Step 5: Commit**

```bash
git add src/report.py tests/test_report.py
git commit -m "Phase 8: transit network layer with polylines + station popups"
```

---

### Task 4: Listing popup HTML builder

**Files:**
- Modify: `src/report.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Produces: `build_popup_html(row: pd.Series, is_ghost: bool = False) -> str`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_report.py
from src.report import build_popup_html


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
    assert "35.56" in html
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report.py::test_build_popup_html_basic -v`
Expected: FAIL with "ImportError: cannot import name 'build_popup_html'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/report.py`:

```python
def _fmt_thb(value: float) -> str:
    return f"฿{value:,.0f}"


def _fmt_date(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "—"
    return dt.strftime("%Y-%m-%d")


def build_popup_html(row: pd.Series, is_ghost: bool = False) -> str:
    thumbnail_html = ""
    if pd.notna(row.get("thumbnail")) and row.get("thumbnail"):
        thumbnail_html = (
            f'<img src="{row["thumbnail"]}" style="width:100%; max-height:150px; '
            f'object-fit:cover; border-radius:4px; margin-bottom:6px;">'
        )

    line_en = row.get("nearest_station_line", "—")
    line_th = LINE_NAMES_TH.get(line_en, line_en)

    ghost_html = ""
    if is_ghost and pd.notna(row.get("listed_dt")):
        import datetime as _dt
        days = (_dt.date.today() - row["listed_dt"].date()).days
        ghost_html = (
            f'<div style="color: #c0392b; font-weight: bold; margin: 4px 0;">'
            f'<span data-en="GHOST · {days} days on market" '
            f'data-th="ค้างนาน · {days} วันที่ค้าง">GHOST · {days} days on market</span>'
            f'</div>'
        )

    listed_str = _fmt_date(row.get("listed_dt"))
    year_built = row.get("year_built") or "—"

    return f"""<div style="font-family: sans-serif; font-size: 12px; max-width: 280px;">
{thumbnail_html}
<b>{row['name']}</b>
<hr style="margin: 4px 0; border: none; border-top: 1px solid #ddd;">
<span data-en="Price" data-th="ราคา">Price</span>: {_fmt_thb(row['price_thb'])}<br>
<span data-en="Area" data-th="พื้นที่">Area</span>: {row['area_sqm_num']:.1f} sqm<br>
<span data-en="Price/sqm" data-th="ราคา/ตร.ม.">Price/sqm</span>: {_fmt_thb(row['price_per_sqm'])}<br>
<span data-en="Beds/Baths" data-th="ห้องนอน/ห้องน้ำ">Beds/Baths</span>: {row['bedrooms']} / {row['bathrooms']}<br>
<span data-en="Year built" data-th="ปีที่สร้าง">Year built</span>: {year_built}<br>
<span data-en="Nearest" data-th="สถานีใกล้สุด">Nearest</span>: {row['nearest_station']} ({row['nearest_station_km']:.3f} km)<br>
<span data-en="Line" data-th="สาย">Line</span>: <span data-en="{line_en}" data-th="{line_th}">{line_en}</span><br>
<span data-en="Listed" data-th="ลงป้ายเมื่อ">Listed</span>: {listed_str}<br>
{ghost_html}
<a href="{row['detail_url']}" target="_blank" style="color: #3498db;">
<span data-en="View on FazWaz →" data-th="ดูบน FazWaz →">View on FazWaz →</span>
</a>
</div>"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report.py -v`
Expected: 11 PASS

- [ ] **Step 5: Commit**

```bash
git add src/report.py tests/test_report.py
git commit -m "Phase 8: listing popup HTML with thumbnail, ghost badge, TH/EN labels"
```

---

### Task 5: Listing + ghost marker builders

**Files:**
- Modify: `src/report.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Consumes: `build_popup_html()` from Task 4, `BIN_COLORS`, `LINE_COLORS` from Task 1
- Produces: `build_listing_markers(df) -> folium.FeatureGroup`, `build_ghost_markers(df) -> folium.FeatureGroup`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_report.py
from src.report import build_listing_markers, build_ghost_markers


def test_build_listing_markers_returns_feature_group(sample_enriched):
    df = load_listings(sample_enriched)
    fg = build_listing_markers(df)
    assert fg is not None
    assert hasattr(fg, "add_to")


def test_build_ghost_markers_empty_when_no_ghosts(sample_enriched):
    df = load_listings(sample_enriched)
    fg = build_ghost_markers(df)
    assert fg is not None


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report.py::test_build_listing_markers_returns_feature_group -v`
Expected: FAIL with "ImportError: cannot import name 'build_listing_markers'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/report.py`:

```python
def build_listing_markers(df: pd.DataFrame) -> folium.FeatureGroup:
    fg = folium.FeatureGroup(name="listings")

    for _, row in df.iterrows():
        if row.get("is_ghost", False):
            continue

        price_bin = int(row.get("price_bin", 1))
        dist_bin = int(row.get("dist_bin", 1))
        line = row.get("nearest_station_line", "")
        line_color = LINE_COLORS.get(line, "#888888")

        default_color = BIN_COLORS[min(price_bin - 1, 3)]

        marker = folium.CircleMarker(
            location=(row["latitude"], row["longitude"]),
            radius=5,
            color=default_color,
            weight=1,
            fill=True,
            fillColor=default_color,
            fillOpacity=0.7,
            popup=folium.Popup(build_popup_html(row), max_width=300),
            tooltip=row["name"],
        )

        marker._dev_data_price = BIN_COLORS[min(price_bin - 1, 3)]
        marker._dev_data_dist = BIN_COLORS[min(dist_bin - 1, 3)]
        marker._dev_data_line = line_color

        fg.add_child(marker)

    return fg


def build_ghost_markers(df: pd.DataFrame) -> folium.FeatureGroup:
    fg = folium.FeatureGroup(name="ghost_listings", show=False)

    if "is_ghost" not in df.columns or not df["is_ghost"].any():
        return fg

    ghost_df = df[df["is_ghost"] == True]

    for _, row in ghost_df.iterrows():
        folium.CircleMarker(
            location=(row["latitude"], row["longitude"]),
            radius=6,
            color="#c0392b",
            weight=2,
            fill=False,
            popup=folium.Popup(build_popup_html(row, is_ghost=True), max_width=300),
            tooltip=row["name"],
        ).add_to(fg)

    return fg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report.py -v`
Expected: 14 PASS

- [ ] **Step 5: Commit**

```bash
git add src/report.py tests/test_report.py
git commit -m "Phase 8: listing + ghost marker builders with data attributes"
```

---

### Task 6: Color toggle JS injection + legend

**Files:**
- Modify: `src/report.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Consumes: `BIN_COLORS`, `LINE_COLORS` from Task 1
- Produces: `inject_color_toggle(m, df, line_colors) -> None`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_report.py
from src.report import inject_color_toggle


def test_inject_color_toggle_adds_elements(sample_enriched, sample_stations):
    df = load_listings(sample_enriched)
    stations = load_stations(sample_stations)
    m = folium.Map(location=[13.75, 100.56], zoom_start=12)

    inject_color_toggle(m, df)

    html_str = m.get_root().render()
    assert "colorMode" in html_str or "color-mode" in html_str
    assert "setStyle" in html_str
    assert "Price per sqm" in html_str or "price_per_sqm" in html_str
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report.py::test_inject_color_toggle_adds_elements -v`
Expected: FAIL with "ImportError: cannot import name 'inject_color_toggle'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/report.py`:

```python
def inject_color_toggle(m: folium.Map, df: pd.DataFrame) -> None:
    unique_lines = sorted(df["nearest_station_line"].dropna().unique()) if "nearest_station_line" in df.columns else []

    line_options = "".join(
        f'<option value="{l}">{l}</option>' for l in unique_lines
    )

    legend_quantile = """
    <div id="legend-quantile" style="display:block;">
      <div style="display:flex; align-items:center; gap:6px; margin:2px 0;">
        <span style="width:12px; height:12px; background:#2ecc71; border-radius:50%; display:inline-block;"></span>
        <span data-en="Budget" data-th="ประหยัด">Budget</span>
      </div>
      <div style="display:flex; align-items:center; gap:6px; margin:2px 0;">
        <span style="width:12px; height:12px; background:#1abc9c; border-radius:50%; display:inline-block;"></span>
        <span data-en="Below median" data-th="ต่ำกว่ามัธยะ">Below median</span>
      </div>
      <div style="display:flex; align-items:center; gap:6px; margin:2px 0;">
        <span style="width:12px; height:12px; background:#3498db; border-radius:50%; display:inline-block;"></span>
        <span data-en="Above median" data-th="สูงกว่ามัธยะ">Above median</span>
      </div>
      <div style="display:flex; align-items:center; gap:6px; margin:2px 0;">
        <span style="width:12px; height:12px; background:#f1c40f; border-radius:50%; display:inline-block;"></span>
        <span data-en="Premium" data-th="ระดับพรีเมียม">Premium</span>
      </div>
    </div>
    """

    line_swatch_items = ""
    for line in unique_lines:
        color = LINE_COLORS.get(line, "#888888")
        line_th = LINE_NAMES_TH.get(line, line)
        line_swatch_items += f"""
      <div style="display:flex; align-items:center; gap:6px; margin:2px 0;">
        <span style="width:12px; height:12px; background:{color}; border-radius:50%; display:inline-block;"></span>
        <span data-en="{line}" data-th="{line_th}">{line}</span>
      </div>"""

    legend_line = f'<div id="legend-line" style="display:none;">{line_swatch_items}</div>'

    toggle_html = f"""
    <div style="position: absolute; top: 10px; right: 10px; z-index: 9999; background: white; padding: 8px; border-radius: 4px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
      <select id="colorMode" onchange="recolorMarkers()" style="margin-bottom: 6px; font-size: 12px;">
        <option value="price" data-en="Price per sqm" data-th="ราคา/ตร.ม.">Price per sqm</option>
        <option value="distance" data-en="Distance to station" data-th="ระยะจากสถานี">Distance to station</option>
        <option value="line" data-en="By transit line" data-th="ตามสายรถไฟ">By transit line</option>
      </select>
      {legend_quantile}
      {legend_line}
    </div>
    """

    recolor_js = """
    <script>
    function recolorMarkers() {
      var mode = document.getElementById('colorMode').value;
      var legendQ = document.getElementById('legend-quantile');
      var legendL = document.getElementById('legend-line');
      if (mode === 'line') {
        legendQ.style.display = 'none';
        legendL.style.display = 'block';
      } else {
        legendQ.style.display = 'block';
        legendL.style.display = 'none';
      }
      var colors = ['#2ecc71', '#1abc9c', '#3498db', '#f1c40f'];

      // Build a map of marker -> colors using folium internals
      // folium CircleMarkers are stored in layer groups; we access via _map
      if (!window._listingMarkers) {
        window._listingMarkers = [];
        // Find the listings FeatureGroup by iterating layers
        map.eachLayer(function(layer) {
          if (layer instanceof L.LayerGroup) {
            layer.eachLayer(function(sub) {
              if (sub instanceof L.CircleMarker && sub._dev_colors) {
                window._listingMarkers.push(sub);
              }
            });
          }
        });
      }

      window._listingMarkers.forEach(function(marker) {
        var color;
        if (mode === 'price') {
          color = marker._dev_colors.price;
        } else if (mode === 'distance') {
          color = marker._dev_colors.dist;
        } else {
          color = marker._dev_colors.line;
        }
        marker.setStyle({fillColor: color, color: color});
      });
    }
    // Initial recolor to ensure default mode
    setTimeout(function() { recolorMarkers(); }, 500);
    </script>
    """

    toggle_element = folium.Element(toggle_html)
    m.get_root().html.add_child(toggle_element)
    js_element = folium.Element(recolor_js)
    m.get_root().html.add_child(js_element)
```

Note: the JS accesses `marker._dev_colors` which we set in `build_listing_markers`. Update that function to store colors in a way JS can read. Folium serializes marker options to JS, but custom Python attributes aren't passed to the browser. We need to inject the color data into the marker's JS options. The cleanest way is to attach a custom JS snippet per marker, but that's heavy. Instead, we'll use a different approach: build a JS data array injected once.

Replace `build_listing_markers` with this version that doesn't use `_dev_data_*`:

Actually, let me reconsider. The cleanest approach for folium: we create a JS data array with all marker colors, and the recolor function matches by index. Let me update:

```python
def build_listing_markers(df: pd.DataFrame) -> tuple[folium.FeatureGroup, list[dict]]:
    fg = folium.FeatureGroup(name="listings")
    color_data = []

    for idx, row in df.iterrows():
        if row.get("is_ghost", False):
            continue

        price_bin = int(row.get("price_bin", 1))
        dist_bin = int(row.get("dist_bin", 1))
        line = row.get("nearest_station_line", "")
        line_color = LINE_COLORS.get(line, "#888888")

        default_color = BIN_COLORS[min(price_bin - 1, 3)]

        folium.CircleMarker(
            location=(row["latitude"], row["longitude"]),
            radius=5,
            color=default_color,
            weight=1,
            fill=True,
            fillColor=default_color,
            fillOpacity=0.7,
            popup=folium.Popup(build_popup_html(row), max_width=300),
            tooltip=row["name"],
        ).add_to(fg)

        color_data.append({
            "price": BIN_COLORS[min(price_bin - 1, 3)],
            "dist": BIN_COLORS[min(dist_bin - 1, 3)],
            "line": line_color,
        })

    return fg, color_data
```

Then `inject_color_toggle` takes `color_data`:

```python
def inject_color_toggle(m: folium.Map, color_data: list[dict]) -> None:
    import json as _json
    colors_json = _json.dumps(color_data)
    # ... rest of function uses colors_json in JS
```

Update the JS:

```javascript
window._markerColors = JSON.parse(' + colors_json + ');
// Match markers by order — the Nth CircleMarker in the listings FeatureGroup
// corresponds to the Nth entry in _markerColors

function recolorMarkers() {
  var mode = document.getElementById('colorMode').value;
  var colors = ['#2ecc71', '#1abc9c', '#3498db', '#f1c40f'];

  if (!window._listingMarkers) {
    window._listingMarkers = [];
    map.eachLayer(function(layer) {
      if (layer instanceof L.LayerGroup && layer.options && layer.options.name === 'listings') {
        layer.eachLayer(function(sub) {
          if (sub instanceof L.CircleMarker) {
            window._listingMarkers.push(sub);
          }
        });
      }
    });
  }

  window._listingMarkers.forEach(function(marker, i) {
    var cd = window._markerColors[i];
    if (!cd) return;
    var color;
    if (mode === 'price') color = cd.price;
    else if (mode === 'distance') color = cd.dist;
    else color = cd.line;
    marker.setStyle({fillColor: color, color: color});
  });
}
```

Hmm, but `layer.options.name` might not be set. Let me check how folium names FeatureGroups. Actually, in folium, FeatureGroup name is set but may not be accessible via `layer.options.name`. The safer approach: give the FeatureGroup a custom ID via `folium.Element`.

Actually, the simplest approach: we know the order. The listings FeatureGroup is added to the map, and within it, markers are in order. We just need to find the right LayerGroup. Let me use a different strategy — add a custom class name.

Let me simplify: instead of matching by layer, we add markers with a custom `className` via the `icon` parameter or via `options`. Actually, CircleMarker doesn't support className easily.

The cleanest solution: use a global variable set when the FeatureGroup is created. We inject a JS snippet right after the FeatureGroup that captures a reference:

```javascript
// Injected right after listings FeatureGroup
window._listingsLayer = <folium_generated_variable_name>;
```

But we don't know folium's generated variable name. Let me use a different approach — iterate ALL CircleMarkers on the map and check if they have a `radius` of 5 (listings) vs 4 (stations) vs 6 (ghosts). That's hacky but works.

Better: we tag each listing marker with a custom tooltip that contains a data attribute. No, tooltips are strings.

Let me use the most reliable approach: store marker references in a global array when they're created. Folium generates JS like `marker_1 = L.circleMarker(...).addTo(layer_1)`. We can override by injecting our own JS that runs after folium's init:

```javascript
// After map init, collect listing markers by radius
map.eachLayer(function(layer) {
  if (layer instanceof L.CircleMarker) {
    var r = layer.options.radius;
    if (r === 5) window._listingMarkers.push(layer);
  }
});
```

This works because: stations have radius=4, listings have radius=5, ghosts have radius=6. The radius is serialized into the marker's options by folium.

Update `inject_color_toggle` to use this approach:

```python
def inject_color_toggle(m: folium.Map, color_data: list[dict]) -> None:
    import json as _json
    colors_json = _json.dumps(color_data)

    unique_lines = []
    for cd in color_data:
        # We don't have line names in color_data, need to pass separately
        pass

    # Build legend from LINE_COLORS for lines that appear in the data
    # We need unique lines — pass them as parameter
```

OK, I need to restructure. Let me pass the df to `inject_color_toggle` as well, or pass unique_lines separately. Let me simplify the function signature:

```python
def inject_color_toggle(m: folium.Map, color_data: list[dict], unique_lines: list[str]) -> None:
```

Let me write the full implementation now with all these decisions resolved. I'll update the implementation in the file directly:

```python
def inject_color_toggle(
    m: folium.Map, color_data: list[dict], unique_lines: list[str]
) -> None:
    import json as _json
    colors_json = _json.dumps(color_data)

    line_swatch_items = ""
    for line in unique_lines:
        color = LINE_COLORS.get(line, "#888888")
        line_th = LINE_NAMES_TH.get(line, line)
        line_swatch_items += f"""
      <div style="display:flex; align-items:center; gap:6px; margin:2px 0;">
        <span style="width:12px; height:12px; background:{color}; border-radius:50%; display:inline-block;"></span>
        <span data-en="{line}" data-th="{line_th}">{line}</span>
      </div>"""

    toggle_html = f"""
    <div style="position: absolute; top: 10px; right: 10px; z-index: 9999; background: white; padding: 8px; border-radius: 4px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
      <select id="colorMode" onchange="recolorMarkers()" style="margin-bottom: 6px; font-size: 12px;">
        <option value="price" data-en="Price per sqm" data-th="ราคา/ตร.ม.">Price per sqm</option>
        <option value="distance" data-en="Distance to station" data-th="ระยะจากสถานี">Distance to station</option>
        <option value="line" data-en="By transit line" data-th="ตามสายรถไฟ">By transit line</option>
      </select>
      <div id="legend-quantile" style="display:block;">
        <div style="display:flex; align-items:center; gap:6px; margin:2px 0;">
          <span style="width:12px; height:12px; background:#2ecc71; border-radius:50%; display:inline-block;"></span>
          <span data-en="Budget" data-th="ประหยัด">Budget</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px; margin:2px 0;">
          <span style="width:12px; height:12px; background:#1abc9c; border-radius:50%; display:inline-block;"></span>
          <span data-en="Below median" data-th="ต่ำกว่ามัธยะ">Below median</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px; margin:2px 0;">
          <span style="width:12px; height:12px; background:#3498db; border-radius:50%; display:inline-block;"></span>
          <span data-en="Above median" data-th="สูงกว่ามัธยะ">Above median</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px; margin:2px 0;">
          <span style="width:12px; height:12px; background:#f1c40f; border-radius:50%; display:inline-block;"></span>
          <span data-en="Premium" data-th="ระดับพรีเมียม">Premium</span>
        </div>
      </div>
      <div id="legend-line" style="display:none;">{line_swatch_items}</div>
    </div>
    """

    recolor_js = f"""
    <script>
    window._markerColors = {colors_json};
    window._listingMarkers = [];

    function collectListingMarkers() {{
      window._listingMarkers = [];
      map.eachLayer(function(layer) {{
        if (layer instanceof L.CircleMarker && layer.options.radius === 5) {{
          window._listingMarkers.push(layer);
        }}
      }});
    }}

    function recolorMarkers() {{
      var mode = document.getElementById('colorMode').value;
      var legendQ = document.getElementById('legend-quantile');
      var legendL = document.getElementById('legend-line');
      if (mode === 'line') {{
        legendQ.style.display = 'none';
        legendL.style.display = 'block';
      }} else {{
        legendQ.style.display = 'block';
        legendL.style.display = 'none';
      }}

      if (window._listingMarkers.length === 0) collectListingMarkers();

      window._listingMarkers.forEach(function(marker, i) {{
        var cd = window._markerColors[i];
        if (!cd) return;
        var color;
        if (mode === 'price') color = cd.price;
        else if (mode === 'distance') color = cd.dist;
        else color = cd.line;
        marker.setStyle({{fillColor: color, color: color}});
      }});
    }}

    setTimeout(function() {{ collectListingMarkers(); recolorMarkers(); }}, 500);
    </script>
    """

    m.get_root().html.add_child(folium.Element(toggle_html))
    m.get_root().html.add_child(folium.Element(recolor_js))
```

Also update `build_listing_markers` to return `color_data`:

```python
def build_listing_markers(df: pd.DataFrame) -> tuple[folium.FeatureGroup, list[dict]]:
    fg = folium.FeatureGroup(name="listings")
    color_data = []

    for _, row in df.iterrows():
        if row.get("is_ghost", False):
            continue

        price_bin = int(row.get("price_bin", 1))
        dist_bin = int(row.get("dist_bin", 1))
        line = row.get("nearest_station_line", "")
        line_color = LINE_COLORS.get(line, "#888888")

        default_color = BIN_COLORS[min(price_bin - 1, 3)]

        folium.CircleMarker(
            location=(row["latitude"], row["longitude"]),
            radius=5,
            color=default_color,
            weight=1,
            fill=True,
            fillColor=default_color,
            fillOpacity=0.7,
            popup=folium.Popup(build_popup_html(row), max_width=300),
            tooltip=row["name"],
        ).add_to(fg)

        color_data.append({
            "price": BIN_COLORS[min(price_bin - 1, 3)],
            "dist": BIN_COLORS[min(dist_bin - 1, 3)],
            "line": line_color,
        })

    return fg, color_data
```

Update the test to match the new return type:

```python
def test_build_listing_markers_returns_feature_group(sample_enriched):
    df = load_listings(sample_enriched)
    fg, color_data = build_listing_markers(df)
    assert fg is not None
    assert hasattr(fg, "add_to")
    assert isinstance(color_data, list)
    assert len(color_data) > 0
    assert "price" in color_data[0]
    assert "dist" in color_data[0]
    assert "line" in color_data[0]
```

And update `inject_color_toggle` test:

```python
def test_inject_color_toggle_adds_elements(sample_enriched, sample_stations):
    df = load_listings(sample_enriched)
    unique_lines = sorted(df["nearest_station_line"].dropna().unique())
    color_data = [{"price": "#2ecc71", "dist": "#2ecc71", "line": "#77BB44"}]
    m = folium.Map(location=[13.75, 100.56], zoom_start=12)

    inject_color_toggle(m, color_data, list(unique_lines))

    html_str = m.get_root().render()
    assert "colorMode" in html_str
    assert "recolorMarkers" in html_str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report.py -v`
Expected: 15 PASS

- [ ] **Step 5: Commit**

```bash
git add src/report.py tests/test_report.py
git commit -m "Phase 8: color toggle dropdown + JS recolor + legend swap"
```

---

### Task 7: TH/EN language toggle

**Files:**
- Modify: `src/report.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Produces: `inject_lang_toggle(m: folium.Map) -> None`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_report.py
from src.report import inject_lang_toggle


def test_inject_lang_toggle_adds_button():
    m = folium.Map(location=[13.75, 100.56], zoom_start=12)
    inject_lang_toggle(m)

    html_str = m.get_root().render()
    assert "langToggle" in html_str
    assert "data-en" in html_str
    assert "data-th" in html_str
    assert "switchLang" in html_str
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report.py::test_inject_lang_toggle_adds_button -v`
Expected: FAIL with "ImportError: cannot import name 'inject_lang_toggle'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/report.py`:

```python
def inject_lang_toggle(m: folium.Map) -> None:
    button_html = """
    <div style="position: absolute; top: 10px; left: 10px; z-index: 9999;">
      <button id="langToggle" onclick="switchLang()" style="background: white; border: 1px solid #ccc; border-radius: 4px; padding: 4px 12px; font-size: 12px; cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
        <span data-en="TH" data-th="EN">TH</span>
      </button>
    </div>
    """

    lang_js = """
    <script>
    window._lang = 'en';

    function switchLang() {
      window._lang = (window._lang === 'en') ? 'th' : 'en';
      var btn = document.getElementById('langToggle');
      btn.textContent = (window._lang === 'en') ? 'TH' : 'EN';

      document.querySelectorAll('[data-en][data-th]').forEach(function(el) {
        el.textContent = (window._lang === 'en') ? el.getAttribute('data-en') : el.getAttribute('data-th');
      });

      // Update select options
      var select = document.getElementById('colorMode');
      if (select) {
        Array.from(select.options).forEach(function(opt) {
          if (opt.hasAttribute('data-en') && opt.hasAttribute('data-th')) {
            opt.textContent = (window._lang === 'en') ? opt.getAttribute('data-en') : opt.getAttribute('data-th');
          }
        });
      }
    }
    </script>
    """

    m.get_root().html.add_child(folium.Element(button_html))
    m.get_root().html.add_child(folium.Element(lang_js))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report.py -v`
Expected: 16 PASS

- [ ] **Step 5: Commit**

```bash
git add src/report.py tests/test_report.py
git commit -m "Phase 8: TH/EN language toggle button + JS text swap"
```

---

### Task 8: Main assembly + output

**Files:**
- Modify: `src/report.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Consumes: all functions from Tasks 1-7

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_report.py
import subprocess
import sys


def test_main_generates_html(sample_enriched, sample_stations, tmp_path):
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report.py::test_main_generates_html -v`
Expected: FAIL (output says "Map not yet implemented — skeleton only")

- [ ] **Step 3: Write minimal implementation**

Replace the `main()` function in `src/report.py`:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Build folium map of listings")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}")
        return
    if not args.stations.exists():
        print(f"Stations not found: {args.stations}")
        return

    print(f"Loading listings: {args.input}")
    df = load_listings(args.input)
    print(f"  {len(df)} listings with coords")

    print(f"Loading stations: {args.stations}")
    stations = load_stations(args.stations)
    print(f"  {len(stations)} stations")

    stations_by_line = sort_stations_by_line(stations)

    m = folium.Map(location=[13.7563, 100.5018], zoom_start=12, tiles="OpenStreetMap")

    transit_fg = build_transit_layer(stations_by_line, stations)
    transit_fg.add_to(m)

    listings_fg, color_data = build_listing_markers(df)
    listings_fg.add_to(m)

    has_ghosts = "is_ghost" in df.columns and df["is_ghost"].any()
    if has_ghosts:
        ghost_fg = build_ghost_markers(df)
        ghost_fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    unique_lines = sorted(df["nearest_station_line"].dropna().unique()) if "nearest_station_line" in df.columns else []
    inject_color_toggle(m, color_data, list(unique_lines))
    inject_lang_toggle(m)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(args.output))
    print(f"Saved map to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report.py -v`
Expected: 17 PASS

- [ ] **Step 5: Run with real sample data**

```bash
python3 src/report.py --input data/interim/listings_sample100_enriched.parquet --output docs/index.html
```
Expected: "Saved map to docs/index.html", file exists, can open in browser

- [ ] **Step 6: Commit**

```bash
git add src/report.py tests/test_report.py docs/index.html
git commit -m "Phase 8: main assembly — folium map with all layers, toggles, popups"
```

---

### Task 9: Integration test with real sample data

**Files:**
- Modify: `tests/test_report.py`

- [ ] **Step 1: Write integration test**

```python
# Append to tests/test_report.py
from pathlib import Path


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
```

- [ ] **Step 2: Run integration test**

Run: `python3 -m pytest tests/test_report.py::test_integration_full_map -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `python3 -m pytest tests/test_report.py -v`
Expected: 18 PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_report.py
git commit -m "Phase 8: integration test with real sample data"
```

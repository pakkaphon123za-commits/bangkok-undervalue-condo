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


def _ref_sort_key(ref: str) -> tuple[str, int]:
    m = re.match(r"([A-Z]+)(\d+)", ref)
    if m:
        return (m.group(1), int(m.group(2)))
    m = re.match(r"^(\d+)$", ref)
    if m:
        return ("", int(m.group(1)))
    return ("ZZZ", 999)


def sort_stations_by_line(
    stations: list[dict[str, Any]],
) -> dict[str, list[list[dict[str, Any]]]]:
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
            any_ref = any(r and re.match(r"[A-Z]+\d+$", r) for r in refs)
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
<span data-en="Area" data-th="พื้นที่">Area</span>: {row['area_sqm_num']:.2f} sqm<br>
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


def build_listing_markers(df: pd.DataFrame) -> tuple[folium.FeatureGroup, list[dict]]:
    fg = folium.FeatureGroup(name="listings")
    color_data: list[dict] = []

    for _, row in df.iterrows():
        if row.get("is_ghost", False):
            continue

        price_bin = int(row.get("price_bin", 1))
        dist_bin = int(row.get("dist_bin", 1))
        line = row.get("nearest_station_line", "")
        line_color = LINE_COLORS.get(line, "#888888")

        price_color = BIN_COLORS[min(price_bin - 1, 3)]
        dist_color = BIN_COLORS[min(dist_bin - 1, 3)]

        marker = folium.CircleMarker(
            location=(row["latitude"], row["longitude"]),
            radius=5,
            color=price_color,
            weight=1,
            fill=True,
            fillColor=price_color,
            fillOpacity=0.7,
            popup=folium.Popup(build_popup_html(row), max_width=300),
            tooltip=row["name"],
        )

        fg.add_child(marker)
        color_data.append({"price": price_color, "dist": dist_color, "line": line_color})

    return fg, color_data


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

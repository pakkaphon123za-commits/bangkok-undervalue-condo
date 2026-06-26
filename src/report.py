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

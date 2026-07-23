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
from folium.plugins import MarkerCluster

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "listings_modeled.parquet"
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

TIER_COLORS = {
    "strong": "#2ecc71",
    "good": "#1abc9c",
    "borderline": "#f1c40f",
    "fair": None,
}

TIER_LABELS = {
    "strong": ("Strong value", "คุณค่าสูง"),
    "good": ("Good value", "คุณค่าดี"),
    "borderline": ("Borderline value", "คุณค่ารอง"),
    "fair": ("Fair value", "ตามตลาด"),
}

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
    "view_on_fazwaz": ("View on Fazwaz", "ดูบน Fazwaz"),
    "ghost_badge": ("GHOST", "ค้างนาน"),
    "days_on_market": ("days on market", "วันที่ค้าง"),
    "ref": ("Ref", "รหัส"),
    "status": ("Status", "สถานะ"),
    "operational": ("Operational", "เปิดให้บริการ"),
    "planned": ("Planned", "กำลังก่อสร้าง"),
    "show_all": ("Show all", "แสดงทั้งหมด"),
    "undervalued_only": ("Undervalued only", "เฉพาะต่ำกว่าราคา"),
    "color_by": ("Color by", "แสดงสีตาม"),
    "reset": ("Reset", "รีเซ็ต"),
    "listings": ("Listings", "ประกาศ"),
    "analysis": ("Analysis", "วิเคราะห์"),
    "no_cards": ("No listings match the current filters.", "ไม่มีประกาศที่ตรงตัวกรอง"),
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def inject_design_system(m: folium.Map) -> None:
    """Inject IBM Plex fonts, CSS tokens, and component styles."""
    font_link = """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Thai:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
    <title>Bangkok Undervalued Condo Listings</title>
    """
    m.get_root().header.add_child(folium.Element(font_link))

    line_css = "\n".join(
        f"    .line-{_slug(name)} {{ --line-color: {color}; }}"
        for name, color in LINE_COLORS.items()
    )

    style = f"""
    <style>
    :root {{
      --ink: #161b22;
      --paper: #f6f7f5;
      --surface: #ffffff;
      --line: #d8dcd5;
      --muted: #8b95a5;
      --accent: #007A33;
      --accent-ink: #ffffff;
      --strong: #16a34a;
      --good: #0ea5e9;
      --borderline: #f59e0b;
      --fair: #94a3b8;
      --r-sm: 6px;
      --r-md: 10px;
      --r-lg: 14px;
      --r-pill: 999px;
      --shadow-sm: 0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.1);
      --shadow-md: 0 4px 8px -2px rgba(16,24,40,.08), 0 2px 4px -2px rgba(16,24,40,.06);
      --shadow-lg: 0 12px 16px -4px rgba(16,24,40,.08), 0 4px 6px -2px rgba(16,24,40,.03);
    }}
    {line_css}
    html, body {{
      margin: 0;
      padding: 0;
      height: 100%;
      overflow: hidden;
      font-family: "IBM Plex Sans", "IBM Plex Sans Thai", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      color: var(--ink);
      background: var(--paper);
    }}
    .leaflet-container {{
      font-family: "IBM Plex Sans", "IBM Plex Sans Thai", system-ui, sans-serif;
    }}
    body:has(#appHeader) {{
      display: grid;
      grid-template-columns: 380px 1fr;
      grid-template-rows: 56px auto 1fr 28px;
      grid-template-areas:
        "header header"
        "filter filter"
        "sidebar map"
        "footer footer";
    }}
    #appHeader {{
      grid-area: header;
      background: var(--ink);
      color: var(--accent-ink);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      gap: 16px;
      box-shadow: var(--shadow-md);
      z-index: 1000;
      position: relative;
    }}
    .header-left {{
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }}
    .roundel {{
      width: 28px;
      height: 28px;
      border-radius: 50%;
      border: 3px solid var(--accent-ink);
      display: grid;
      place-items: center;
      flex-shrink: 0;
    }}
    .roundel::after {{
      content: "";
      width: 34px;
      height: 5px;
      background: var(--accent-ink);
      border-radius: 2px;
    }}
    .brand-title {{
      font-weight: 600;
      font-size: 15px;
      letter-spacing: .2px;
      white-space: nowrap;
    }}
    .header-stats {{
      display: flex;
      align-items: center;
      gap: 18px;
      font-family: "IBM Plex Mono", ui-monospace, "SF Mono", monospace;
      font-size: 13px;
      flex-shrink: 0;
    }}
    .stat {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .stat-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      flex-shrink: 0;
    }}
    .stat-num {{
      font-weight: 500;
      color: #ffffff;
    }}
    .stat-label {{
      color: rgba(255,255,255,.75);
    }}
    .header-right {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-shrink: 0;
    }}
    #filterBar {{
      position: fixed;
      top: 66px;
      left: 10px;
      z-index: 999;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--r-md);
      padding: 8px 12px;
      box-shadow: var(--shadow-md);
      display: flex;
      align-items: center;
      gap: 16px;
      font-size: 12px;
    }}
    body:has(#appHeader) #filterBar {{
      position: static;
      grid-area: filter;
      border: none;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
      box-shadow: none;
    }}
    .filter-group {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .filter-group label {{
      color: var(--muted);
      font-weight: 500;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .4px;
    }}
    .chip-group, .chip-segment {{
      display: inline-flex;
      align-items: center;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--r-pill);
      padding: 2px;
    }}
    .chip-btn {{
      border: none;
      background: transparent;
      padding: 5px 12px;
      border-radius: var(--r-pill);
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      color: var(--ink);
      transition: background .15s, color .15s;
    }}
    .chip-btn:hover {{
      background: rgba(0,0,0,.04);
    }}
    .chip-btn.active {{
      background: var(--accent);
      color: var(--accent-ink);
    }}
    .chip-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 10px;
      border: 1px solid var(--line);
      border-radius: var(--r-pill);
      background: var(--surface);
      cursor: pointer;
      font-size: 12px;
      font-weight: 500;
    }}
    .chip-toggle input {{
      margin: 0;
    }}
    select.chip-select, #lineFilter {{
      border: 1px solid var(--line);
      border-radius: var(--r-pill);
      background: var(--surface);
      padding: 5px 12px;
      font-size: 12px;
      font-family: inherit;
    }}
    #appSidebar {{
      grid-area: sidebar;
      background: var(--surface);
      border-right: 1px solid var(--line);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      z-index: 100;
    }}
    #map, .folium-map {{
      grid-area: map;
      width: 100%;
      height: 100%;
    }}
    .sidebar-tabs {{
      display: flex;
      border-bottom: 1px solid var(--line);
      background: var(--paper);
    }}
    .tab-btn {{
      flex: 1;
      border: none;
      background: transparent;
      padding: 12px;
      font-size: 13px;
      font-weight: 500;
      color: var(--muted);
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: color .15s;
    }}
    .tab-btn.active {{
      color: var(--ink);
      border-bottom-color: var(--accent);
      background: var(--surface);
    }}
    .sidebar-panels {{
      flex: 1;
      overflow: hidden;
      position: relative;
    }}
    .tab-panel {{
      position: absolute;
      inset: 0;
      overflow-y: auto;
      padding: 12px;
      display: none;
    }}
    .tab-panel.active {{
      display: block;
    }}
    .listing-card {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--r-md);
      overflow: hidden;
      margin-bottom: 12px;
      box-shadow: var(--shadow-sm);
      cursor: pointer;
      transition: transform .15s, box-shadow .15s;
      display: flex;
      flex-direction: row;
      min-height: 110px;
    }}
    .listing-card:hover {{
      transform: translateY(-1px);
      box-shadow: var(--shadow-md);
    }}
    .card-photo-wrap {{
      width: 110px;
      min-width: 110px;
      min-height: 110px;
      align-self: stretch;
      background: linear-gradient(135deg, #e0e4e8, #c8cdd3);
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 12px;
      position: relative;
      overflow: hidden;
    }}
    .card-photo-wrap img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .card-photo-wrap svg {{
      width: 28px;
      height: 28px;
      opacity: .5;
    }}
    .card-body {{
      padding: 12px;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 8px;
      margin-bottom: 8px;
    }}
    .card-title {{
      font-weight: 600;
      font-size: 14px;
      line-height: 1.3;
      margin: 0;
    }}
    .tier-pill {{
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .4px;
      padding: 3px 8px;
      border-radius: var(--r-pill);
      color: #fff;
      flex-shrink: 0;
      white-space: nowrap;
    }}
    .card-price-row {{
      display: flex;
      align-items: baseline;
      gap: 10px;
      margin-bottom: 8px;
    }}
    .card-price {{
      font-family: "IBM Plex Mono", ui-monospace, "SF Mono", monospace;
      font-size: 20px;
      font-weight: 500;
      letter-spacing: -0.5px;
    }}
    .card-price-unit {{
      font-size: 12px;
      color: var(--muted);
      font-family: "IBM Plex Mono", ui-monospace, "SF Mono", monospace;
    }}
    .card-stats {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .card-stat {{
      color: var(--muted);
    }}
    .card-stat strong {{
      display: block;
      color: var(--ink);
      font-weight: 500;
    }}
    .card-station {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .card-station .line-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      flex-shrink: 0;
    }}
    .card-station strong {{
      color: var(--ink);
    }}
    .undervalued-strip {{
      background: rgba(22,163,74,.08);
      border: 1px solid rgba(22,163,74,.25);
      color: #166534;
      border-radius: var(--r-sm);
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 500;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .ghost-strip {{
      background: rgba(220,38,38,.08);
      border: 1px solid rgba(220,38,38,.25);
      color: #991b1b;
      border-radius: var(--r-sm);
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 600;
      margin-bottom: 8px;
    }}
    .card-listed {{
      font-size: 11px;
      color: var(--muted);
    }}
    .listing-card.tier-fair {{
      display: none;
    }}
    .listing-card.tier-fair.visible {{
      display: block;
    }}
    .show-all-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 4px 0 12px;
      font-size: 12px;
      color: var(--muted);
    }}
    .show-all-bar label {{
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .analysis-section {{
      margin-bottom: 16px;
    }}
    .analysis-section h4 {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .4px;
      color: var(--muted);
      margin: 0 0 8px;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      margin: 4px 0;
    }}
    .legend-swatch {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      flex-shrink: 0;
    }}
    .legend-gradient {{
      height: 12px;
      border-radius: var(--r-pill);
      background: linear-gradient(to right, {" , ".join(BIN_COLORS)});
      margin: 6px 0 8px;
    }}
    .legend-labels {{
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      color: var(--muted);
      font-family: "IBM Plex Mono", ui-monospace, "SF Mono", monospace;
    }}
    #appFooter {{
      grid-area: footer;
      background: var(--paper);
      border-top: 1px solid var(--line);
      display: flex;
      align-items: center;
      padding: 0 16px;
      font-size: 11px;
      color: var(--muted);
      z-index: 100;
    }}
    #narrativePanel {{
      position: fixed;
      top: 0;
      right: -420px;
      width: 420px;
      max-width: 90vw;
      height: 100%;
      background: var(--surface);
      z-index: 10000;
      overflow-y: auto;
      box-shadow: var(--shadow-lg);
      transition: right .25s ease;
    }}
    #narrativePanel.open {{
      right: 0;
    }}
    .narrative-header {{
      position: sticky;
      top: 0;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
      padding: 12px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 2;
    }}
    .narrative-header h3 {{
      margin: 0;
      font-size: 16px;
    }}
    .narrative-close {{
      background: none;
      border: none;
      font-size: 22px;
      cursor: pointer;
      color: var(--muted);
      width: 32px;
      height: 32px;
      display: grid;
      place-items: center;
      border-radius: 50%;
    }}
    .narrative-close:hover {{
      background: var(--paper);
    }}
    .narrative-body {{
      padding: 16px;
      font-size: 13px;
      line-height: 1.6;
    }}
    .narrative-body h1 {{
      font-size: 18px;
      margin: 0 0 12px;
    }}
    .narrative-body h2 {{
      font-size: 15px;
      margin: 24px 0 10px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 6px;
    }}
    .narrative-body h3 {{
      font-size: 13px;
      margin: 18px 0 8px;
    }}
    .narrative-body p {{
      margin: 0 0 12px;
    }}
    .narrative-body table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      margin: 12px 0;
    }}
    .narrative-body th, .narrative-body td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      text-align: left;
    }}
    .narrative-body th {{
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      font-size: 10px;
      letter-spacing: .4px;
    }}
    .narrative-body td:nth-child(n+2), .narrative-body th:nth-child(n+2) {{
      text-align: right;
      font-family: "IBM Plex Mono", ui-monospace, "SF Mono", monospace;
    }}
    .narrative-body tr:nth-child(even) {{
      background: var(--paper);
    }}
    .narrative-body strong {{
      color: var(--ink);
    }}
    .leaflet-popup-content-wrapper {{
      border-radius: var(--r-md);
      box-shadow: var(--shadow-lg);
      padding: 0;
      overflow: hidden;
      max-width: 320px;
    }}
    .leaflet-popup-content {{
      margin: 0;
      width: 320px !important;
    }}
    .leaflet-container a.leaflet-popup-close-button {{
      top: 8px;
      right: 8px;
      color: var(--ink);
      font-size: 20px;
      width: 24px;
      height: 24px;
      display: grid;
      place-items: center;
      border-radius: 50%;
      background: rgba(255,255,255,.8);
      padding: 0;
      line-height: 24px;
    }}
    .leaflet-control-zoom {{
      border: none !important;
      box-shadow: var(--shadow-md) !important;
      border-radius: var(--r-md) !important;
      overflow: hidden;
      top: auto !important;
      bottom: 80px !important;
      left: auto !important;
      right: 12px !important;
    }}
    .leaflet-control-zoom a {{
      width: 32px !important;
      height: 32px !important;
      line-height: 32px !important;
      font-size: 18px !important;
      color: var(--ink) !important;
      border-color: var(--line) !important;
    }}
    .leaflet-control-layers {{
      border: none !important;
      box-shadow: var(--shadow-md) !important;
      border-radius: var(--r-md) !important;
      overflow: hidden;
      font-size: 12px !important;
      background: var(--surface) !important;
    }}
    .leaflet-control-layers-toggle {{
      width: 32px !important;
      height: 32px !important;
    }}
    .leaflet-control-layers-expanded {{
      padding: 10px 12px !important;
    }}
    .leaflet-control-layers-list {{
      line-height: 1.5;
    }}
    .leaflet-control-layers-base label, .leaflet-control-layers-overlays label {{
      display: flex;
      align-items: center;
      gap: 6px;
      margin: 4px 0;
    }}
    .leaflet-control-layers-separator {{
      border-top-color: var(--line) !important;
      margin: 8px 0 !important;
    }}
    .leaflet-control-attribution {{
      display: none;
    }}
    .price-bubble {{
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 4px 10px;
      border-radius: var(--r-pill);
      font-family: "IBM Plex Mono", ui-monospace, "SF Mono", monospace;
      font-size: 12px;
      font-weight: 600;
      color: #fff;
      text-shadow: 0 1px 2px rgba(0,0,0,.6), 0 0 1px rgba(0,0,0,.9);
      white-space: nowrap;
      box-shadow: var(--shadow-sm), 0 0 0 1px rgba(0,0,0,.15);
      border: 1px solid rgba(0,0,0,.2);
      cursor: pointer;
      transition: transform .15s;
    }}
    .price-bubble:hover {{
      transform: scale(1.05);
      z-index: 1000 !important;
    }}
    .price-bubble.ghost {{
      background: #fff;
      color: #991b1b;
      border: 2px solid #991b1b;
    }}
    .listing-marker .leaflet-marker-icon {{
      border: none;
      background: transparent;
    }}
    .mobile-toggle {{
      display: none;
    }}
    @media (max-width: 900px) {{
      body:has(#appHeader) {{
        grid-template-columns: 1fr;
        grid-template-rows: 56px auto 1fr 28px;
        grid-template-areas:
          "header"
          "filter"
          "map"
          "footer";
      }}
      #appSidebar {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 60vh;
        z-index: 1000;
        transform: translateY(calc(100% - 44px));
        transition: transform .25s ease;
        border-right: none;
        border-top: 1px solid var(--line);
        border-radius: var(--r-lg) var(--r-lg) 0 0;
        box-shadow: var(--shadow-lg);
      }}
      #appSidebar.open {{
        transform: translateY(0);
      }}
      .mobile-toggle {{
        display: flex;
        position: fixed;
        bottom: 12px;
        right: 12px;
        z-index: 1001;
        gap: 8px;
      }}
      .mobile-toggle button {{
        background: var(--ink);
        color: #fff;
        border: none;
        border-radius: var(--r-pill);
        padding: 10px 16px;
        font-size: 13px;
        font-weight: 500;
        box-shadow: var(--shadow-md);
        cursor: pointer;
      }}
      .header-stats {{
        display: none;
      }}
    }}
    .listing-popup-card {{
      background: var(--surface);
      display: block;
    }}
    .listing-popup-card .card-photo-wrap {{
      width: 100%;
      height: 140px;
    }}
    .listing-popup-card .card-body {{
      padding: 14px;
    }}
    .listing-popup-card .card-title {{
      font-size: 15px;
    }}
    .listing-popup-card .card-price {{
      font-size: 22px;
    }}
    .card-body {{
      padding: 10px 12px;
      flex: 1;
      min-width: 0;
    }}
    .card-cta {{
      display: block;
      margin-top: 12px;
      background: var(--accent);
      color: #000 !important;
      text-align: center;
      padding: 10px;
      border-radius: var(--r-sm);
      text-decoration: none !important;
      font-weight: 600;
      font-size: 13px;
    }}
    .card-cta:hover {{
      filter: brightness(1.05);
    }}
    .station-popup-card {{
      padding: 12px;
      min-width: 180px;
    }}
    .station-popup-name {{
      font-weight: 600;
      font-size: 15px;
      margin-bottom: 6px;
    }}
    .station-popup-lines {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      margin-bottom: 8px;
      color: var(--muted);
    }}
    .station-popup-meta {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .ref-chip {{
      font-family: "IBM Plex Mono", ui-monospace, "SF Mono", monospace;
      font-size: 11px;
      background: var(--paper);
      padding: 2px 8px;
      border-radius: var(--r-sm);
      border: 1px solid var(--line);
    }}
    .status-pill {{
      font-size: 11px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: var(--r-pill);
      color: #fff;
    }}
    .status-operational {{
      background: var(--strong);
    }}
    .status-planned {{
      background: var(--borderline);
    }}
    .line-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
      flex-shrink: 0;
    }}
    .leaflet-marker-icon .price-bubble {{
      position: relative;
      top: -14px;
      left: 50%;
      transform: translateX(-50%);
    }}
    .listing-card.selected {{
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(0,122,51,.18), var(--shadow-md);
      background: rgba(0,122,51,.04);
    }}
    #refineBar {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--paper);
      display: flex;
      flex-direction: column;
      gap: 8px;
      flex-shrink: 0;
    }}
    #refineSearch {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: var(--r-sm);
      padding: 7px 10px 7px 30px;
      font-size: 13px;
      font-family: inherit;
      background: var(--surface) url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%238b95a5' stroke-width='2'><circle cx='11' cy='11' r='7'/><path d='M21 21l-4.3-4.3'/></svg>") no-repeat 9px center;
    }}
    #refineSearch:focus {{
      outline: none;
      border-color: var(--accent);
    }}
    .refine-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 11px;
    }}
    .refine-row label {{
      color: var(--muted);
      font-weight: 500;
      min-width: 52px;
      text-transform: uppercase;
      letter-spacing: .3px;
    }}
    #sortSelect, #refineBar select {{
      border: 1px solid var(--line);
      border-radius: var(--r-sm);
      background: var(--surface);
      padding: 5px 8px;
      font-size: 12px;
      font-family: inherit;
      flex: 1;
    }}
    .slider-group {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex: 1;
    }}
    .slider-group input[type=range] {{
      flex: 1;
      accent-color: var(--accent);
      height: 4px;
    }}
    .slider-input {{
      width: 90px;
      border: 1px solid var(--line);
      border-radius: var(--r-sm);
      background: var(--surface);
      padding: 4px 8px;
      font-size: 12px;
      font-family: "IBM Plex Mono", ui-monospace, monospace;
      text-align: right;
    }}
    .slider-input::placeholder {{ color: var(--muted); }}
    .slider-input:focus {{ outline: none; border-color: var(--accent); }}
    .cluster-icon {{
      background: var(--ink);
      color: #fff;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: "IBM Plex Mono", ui-monospace, monospace;
      font-weight: 600;
      font-size: 13px;
      border: 2px solid rgba(255,255,255,.7);
      box-shadow: var(--shadow-md);
    }}
    .cluster-icon.sz-s {{ width: 32px; height: 32px; font-size: 11px; }}
    .cluster-icon.sz-m {{ width: 40px; height: 40px; font-size: 13px; }}
    .cluster-icon.sz-l {{ width: 48px; height: 48px; font-size: 15px; }}
    #filterStats {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 14px;
    }}
    .stat-card {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--r-sm);
      padding: 10px;
    }}
    .stat-card .stat-card-val {{
      font-family: "IBM Plex Mono", ui-monospace, monospace;
      font-size: 18px;
      font-weight: 600;
      color: var(--ink);
      line-height: 1.2;
    }}
    .stat-card .stat-card-lbl {{
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: .3px;
      color: var(--muted);
      margin-top: 2px;
    }}
    #scatterChart {{
      width: 100%;
      height: 180px;
      margin-bottom: 14px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--r-sm);
    }}
    #scatterChart .axis {{ stroke: var(--line); stroke-width: 1; }}
    #scatterChart .axis-label {{ fill: var(--muted); font-size: 9px; font-family: "IBM Plex Mono", monospace; }}
    #scatterChart .dot {{ stroke: #fff; stroke-width: .5; }}
    #topPicks {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .top-pick {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: var(--r-sm);
      background: var(--surface);
      cursor: pointer;
      transition: border-color .15s, background .15s;
    }}
    .top-pick:hover {{
      border-color: var(--accent);
      background: rgba(0,122,51,.04);
    }}
    .top-pick-rank {{
      font-family: "IBM Plex Mono", monospace;
      font-size: 11px;
      color: var(--muted);
      width: 16px;
      flex-shrink: 0;
    }}
    .top-pick-name {{
      flex: 1;
      font-size: 12px;
      font-weight: 500;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .top-pick-pct {{
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      font-weight: 600;
      color: var(--strong);
      flex-shrink: 0;
    }}
    #emptyState {{
      text-align: center;
      padding: 40px 20px;
      color: var(--muted);
    }}
    #emptyState svg {{
      width: 40px;
      height: 40px;
      opacity: .4;
      margin-bottom: 12px;
    }}
    #emptyState p {{
      margin: 0 0 14px;
      font-size: 13px;
    }}
    #emptyState button {{
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: var(--r-sm);
      padding: 8px 18px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
    }}
    #fitAllBtn {{
      position: fixed;
      bottom: 80px;
      left: 12px;
      z-index: 800;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--r-md);
      padding: 8px 12px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      box-shadow: var(--shadow-md);
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    #fitAllBtn:hover {{
      border-color: var(--accent);
    }}
    #darkToggle {{
      background: transparent;
      border: 1px solid rgba(255,255,255,.3);
      color: #fff;
      border-radius: var(--r-pill);
      padding: 5px 12px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
    }}
    html[data-theme="dark"] {{
      --ink: #e6edf3;
      --paper: #0d1117;
      --surface: #161b22;
      --line: #30363d;
      --muted: #7d8590;
      --accent: #3fb950;
      --accent-ink: #0d1117;
      --shadow-sm: 0 1px 2px rgba(0,0,0,.4);
      --shadow-md: 0 4px 8px -2px rgba(0,0,0,.5);
      --shadow-lg: 0 12px 16px -4px rgba(0,0,0,.6);
    }}
    html[data-theme="dark"] .leaflet-tile-pane {{
      filter: brightness(.55) invert(1) contrast(.92) hue-rotate(180deg) saturate(.7);
    }}
    html[data-theme="dark"] .leaflet-container {{
      background: #0d1117;
    }}
    html[data-theme="dark"] .price-bubble {{
      border-color: rgba(255,255,255,.15);
    }}
    html[data-theme="dark"] #refineSearch {{
      background-color: var(--surface);
    }}
    @media (prefers-reduced-motion: reduce) {{
      *, *::before, *::after {{
        animation-duration: .01ms !important;
        transition-duration: .01ms !important;
      }}
    }}
    </style>
    """
    m.get_root().html.add_child(folium.Element(style))


def build_listing_cards_html(df: pd.DataFrame) -> str:
    """Generate Airbnb-style listing cards for the left panel."""
    cards_html: list[str] = []
    for idx, row in df.iterrows():
        if row.get("is_ghost", False):
            continue
        cards_html.append(build_listing_card_html(row, idx))

    if not cards_html:
        return (
            '<div class="no-cards">'
            '<span data-en="No listings match the current filters." data-th="ไม่มีประกาศที่ตรงตัวกรอง">'
            'No listings match the current filters.'
            '</span></div>'
        )

    show_all_bar = (
        '<div class="show-all-bar">'
        '<span id="resultCount">0</span>'
        '<label><input type="checkbox" id="showAllCards"> '
        '<span data-en="Show all" data-th="แสดงทั้งหมด">Show all</span></label>'
        '</div>'
    )
    return show_all_bar + "\n".join(cards_html)


def inject_app_shell(
    m: folium.Map,
    stats: dict[str, Any],
) -> None:
    """Inject the app shell: header, sidebar with tabs, footer."""
    total = stats.get("total", 0)
    pct_uv = stats.get("pct_undervalued", 0.0)
    decay = stats.get("top_decay", 0.0)
    updated = stats.get("updated", "—")

    header = f"""
    <header id="appHeader">
      <div class="header-left">
        <div class="roundel" aria-hidden="true"></div>
        <div class="brand">
          <div class="brand-title">
            <span data-en="Bangkok Undervalued Condo Listings" data-th="Bangkok Undervalued Condo Listings">Bangkok Undervalued Condo Listings</span>
          </div>
        </div>
      </div>
      <div class="header-stats">
        <div class="stat">
          <span class="stat-dot" style="background:#77BB44"></span>
          <span class="stat-num">{total:,}</span>
          <span class="stat-label" data-en="listings" data-th="ประกาศ">listings</span>
        </div>
        <div class="stat">
          <span class="stat-dot" style="background:#EC008C"></span>
          <span class="stat-num">{pct_uv:.2f}%</span>
          <span class="stat-label" data-en="undervalued" data-th="ต่ำกว่าราคา">undervalued</span>
        </div>
        <div class="stat">
          <span class="stat-dot" style="background:#007A33"></span>
          <span class="stat-num">{decay:.1f}%</span>
          <span class="stat-label" data-en="/km" data-th="/กม">/km</span>
        </div>
      </div>
      <div class="header-right">
        <button id="darkToggle" type="button">Dark</button>
        <div id="langToggleContainer"></div>
      </div>
    </header>
    """

    sidebar = f"""
    <aside id="appSidebar">
      <div class="sidebar-tabs" role="tablist">
        <button class="tab-btn active" data-tab="listings" role="tab" aria-selected="true">
          <span data-en="Listings" data-th="ประกาศ">Listings</span>
        </button>
        <button class="tab-btn" data-tab="analysis" role="tab" aria-selected="false">
          <span data-en="Analysis" data-th="วิเคราะห์">Analysis</span>
        </button>
      </div>
      <div class="sidebar-panels">
        <div id="listingPanel" class="tab-panel active" role="tabpanel">
          <div id="refineBar">
            <input type="text" id="refineSearch" placeholder="Condo, station, area...">
            <div class="refine-row">
              <label>Sort</label>
              <select id="sortSelect">
                <option value="value">Best value</option>
                <option value="price_asc">Price: Low to High</option>
                <option value="price_desc">Price: High to Low</option>
                <option value="distance">Nearest to transit</option>
                <option value="deal">Best deal %</option>
                <option value="newest">Newest</option>
              </select>
            </div>
            <div class="refine-row">
              <label>Price</label>
              <div class="slider-group">
                <input type="range" id="priceSlider" min="0" max="100" value="100">
                <input type="number" id="priceInput" class="slider-input" placeholder="max baht" min="0">
              </div>
            </div>
            <div class="refine-row">
              <label>Dist</label>
              <div class="slider-group">
                <input type="range" id="distSlider" min="0" max="100" value="100">
                <input type="number" id="distInput" class="slider-input" placeholder="max km" min="0" step="0.1">
              </div>
            </div>
          </div>
          <div class="show-all-bar">
            <span id="resultCount">0 listings</span>
            <label><input type="checkbox" id="showAllCards"> <span data-en="Show all" data-th="แสดงทั้งหมด">Show all</span></label>
          </div>
          <div id="listingScroll"></div>
          <div id="listSentinel" style="height:1px;"></div>
        </div>
        <div id="analysisPanel" class="tab-panel" role="tabpanel">
          <div class="analysis-section">
            <h4><span data-en="Current view" data-th="มุมมองปัจจุบัน">Current view</span></h4>
            <div id="filterStats"></div>
          </div>
          <div class="analysis-section">
            <h4><span data-en="Price vs distance" data-th="ราคาเทียบระยะทาง">Price vs distance</span></h4>
            <svg id="scatterChart" viewBox="0 0 240 180"></svg>
          </div>
          <div class="analysis-section">
            <h4><span data-en="Top undervalued picks" data-th="ตัวเลือกต่ำกว่าราคายอดนิยม">Top undervalued picks</span></h4>
            <div id="topPicks"></div>
          </div>
          <div class="analysis-section">
            <h4><span data-en="Legend" data-th="คำอธิบายสี">Legend</span></h4>
            <div id="analysisLegend"></div>
          </div>
          <div class="analysis-section">
            <h4><span data-en="Layers" data-th="ชั้นข้อมูล">Layers</span></h4>
            <p class="legend-row" style="color:var(--muted);">
              <span data-en="Use the map layer control to toggle transit lines and listing groups." data-th="ใช้ปุ่มควบคุมชั้นข้อมูลบนแผนที่เพื่อเปิด/ปิดสายรถไฟและกลุ่มประกาศ">Use the map layer control to toggle transit lines and listing groups.</span>
            </p>
          </div>
          <div id="narrativeToggleContainer"></div>
        </div>
      </div>
    </aside>
    """

    footer = f"""
    <footer id="appFooter">
      <span>
        <span data-en="Data: FazWaz · Stations: OpenStreetMap (ODbL) · Model: hedonic price-decay · Updated" data-th="ข้อมูล: FazWaz · สถานี: OpenStreetMap (ODbL) · โมเดล: hedonic price-decay · อัปเดต">Data: FazWaz · Stations: OpenStreetMap (ODbL) · Model: hedonic price-decay · Updated</span>
        {updated}
      </span>
    </footer>
    """

    mobile_toggle = """
    <div class="mobile-toggle">
      <button id="mobileListBtn" onclick="switchMobileView('list')">List</button>
      <button id="mobileMapBtn" onclick="switchMobileView('map')">Map</button>
    </div>
    <button id="fitAllBtn" type="button">
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 9V3h6M21 9V3h-6M3 15v6h6M21 15v6h-6"/></svg>
      <span data-en="Fit all" data-th="มองทั้งหมด">Fit all</span>
    </button>
    """

    m.get_root().html.add_child(folium.Element(header))
    m.get_root().html.add_child(folium.Element(sidebar))
    m.get_root().html.add_child(folium.Element(footer))
    m.get_root().html.add_child(folium.Element(mobile_toggle))


def load_listings(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.dropna(subset=["latitude", "longitude"]).copy()

    # Drop unrealistic outlier listings above 100M baht
    if "price_thb" in df.columns:
        df = df[df["price_thb"].fillna(0) <= 100_000_000].copy()

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



def _markdown_to_html(md: str) -> str:
    lines = md.strip().splitlines()
    out: list[str] = []
    in_table = False
    table_rows: list[str] = []

    def inline(text: str) -> str:
        text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        return text

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if in_table and len(table_rows) >= 2:
            out.append("<table style='border-collapse:collapse;width:100%;font-size:12px;'>")
            for idx, row in enumerate(table_rows):
                if idx == 1:
                    continue
                cells = [c.strip() for c in row.strip("|").split("|")]
                tag = "th" if idx == 0 else "td"
                out.append("<tr>")
                for c in cells:
                    out.append(f"<{tag} style='border:1px solid #ddd;padding:4px;'>{inline(c)}</{tag}>")
                out.append("</tr>")
            out.append("</table>")
        in_table = False
        table_rows = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            if not in_table:
                flush_table()
                in_table = True
            table_rows.append(stripped)
            continue

        flush_table()

        if stripped.startswith("### "):
            out.append(f"<h3>{inline(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            out.append(f"<h2>{inline(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            out.append(f"<h1>{inline(stripped[2:])}</h1>")
        elif stripped == "":
            if out and out[-1] != "":
                out.append("")
        else:
            out.append(f"<p>{inline(stripped)}</p>")

    flush_table()
    return "\n".join(out)


def inject_narrative_panel(m: folium.Map, narrative_html: str, meta_lines: list[dict]) -> None:
    if not narrative_html and not meta_lines:
        return

    rows = ""
    for line in meta_lines:
        name_th = LINE_NAMES_TH.get(line["name"], "")
        decay = line.get("decay_pct_per_km", "—")
        line_color = LINE_COLORS.get(line["name"], "#888888")
        rows += (
            f"<tr>"
            f"<td><span class='legend-swatch' style='background:{line_color}; margin-right:6px;'></span>"
            f"<span data-en='{line['name']}' data-th='{name_th}'>{line['name']}</span></td>"
            f"<td>{line['n']}</td>"
            f"<td>{line['pct_undervalued']:.2f}%</td>"
            f"<td>{decay}<span data-en='%/km' data-th='%/กม'>%/km</span></td>"
            f"</tr>"
        )

    summary_table = ""
    if meta_lines:
        summary_table = f"""
        <h2><span data-en="Per-line summary" data-th="สรุปตามสาย">Per-line summary</span></h2>
        <table>
          <tr>
            <th><span data-en="Line" data-th="สาย">Line</span></th>
            <th><span data-en="N" data-th="N">N</span></th>
            <th><span data-en="Undervalued %" data-th="% ต่ำกว่าโมเดล">Undervalued %</span></th>
            <th><span data-en="Decay %/km" data-th="% ลด/กม">Decay %/km</span></th>
          </tr>
          {rows}
        </table>
        """

    panel_html = f"""
    <div id="narrativePanel">
      <div class="narrative-header">
        <h3><span data-en="Analysis" data-th="บทวิเคราะห์">Analysis</span></h3>
        <button class="narrative-close" onclick="closeNarrative()" aria-label="Close">×</button>
      </div>
      <div class="narrative-body">
        {narrative_html}
        {summary_table}
      </div>
    </div>
    <button id="narrativeToggle" class="chip-btn" onclick="openNarrative()" style="position:fixed; top:48px; right:10px; z-index:9999;">
      <span data-en="Narrative" data-th="บทวิเคราะห์">Narrative</span>
    </button>
    """

    panel_js = """
    <script>
    function openNarrative() {
      var panel = document.getElementById('narrativePanel');
      if (panel) panel.classList.add('open');
    }
    function closeNarrative() {
      var panel = document.getElementById('narrativePanel');
      if (panel) panel.classList.remove('open');
    }
    </script>
    """

    m.get_root().html.add_child(folium.Element(panel_html))
    m.get_root().html.add_child(folium.Element(panel_js))


def inject_bootstrap_js(m: folium.Map) -> None:
    """Wire up the app shell interactions after folium renders the map."""
    js = r"""
    <script>
    (function() {
      function ready(fn) {
        if (document.readyState !== 'loading') fn();
        else document.addEventListener('DOMContentLoaded', fn);
      }

      var TIER_LABELS = {strong: ['Strong value','คุณค่าสูง'], good: ['Good value','คุณค่าดี'], borderline: ['Borderline value','คุณค่ารอง'], fair: ['Fair value','ตามตลาด']};
      var TIER_COLORS = {strong: '#2ecc71', good: '#1abc9c', borderline: '#f1c40f', fair: '#94a3b8'};
      var RENDER_BATCH = 40;

      function slugify(text) {
        return (text || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
      }
      function esc(s) {
        var d = document.createElement('div');
        d.textContent = s == null ? '' : String(s);
        return d.innerHTML;
      }
      function fmtThb(v) {
        if (v == null) return '\u0e3f\u2014';
        return '\u0e3f' + Math.round(v).toLocaleString();
      }
      function fmtPpsm(v) {
        if (v == null) return '\u2014';
        return '\u0e3f' + Math.round(v).toLocaleString();
      }

      function cardInnerHtml(l) {
        var lang = window._lang || 'en';
        var tl = TIER_LABELS[l.tier] || TIER_LABELS.fair;
        var tc = TIER_COLORS[l.tier] || TIER_COLORS.fair;
        var lineTh = (window._lineNamesTh && window._lineNamesTh[l.line]) || l.line;
        var lineColor = (window._lineColors && window._lineColors[l.line]) || '#888';
        var lineSlug = slugify(l.line);

        var tierPill = '<span class="tier-pill" style="background:' + tc + ';" data-en="' + tl[0] + '" data-th="' + tl[1] + '">' + (lang === 'en' ? tl[0] : tl[1]) + '</span>';

        var uv = '';
        if (l.undervalued && l.undervaluedPct && l.undervaluedPct > 0) {
          var uvEn = 'Undervalued by', uvTh = '\u0e15\u0e48\u0e33\u0e01\u0e27\u0e48\u0e32\u0e42\u0e21\u0e40\u0e14\u0e25';
          uv = '<div class="undervalued-strip"><span data-en="' + uvEn + '" data-th="' + uvTh + '">' + (lang === 'en' ? uvEn : uvTh) + '</span> ' + l.undervaluedPct.toFixed(1) + '%</div>';
        }

        var kmStr = l.stationKm != null ? l.stationKm.toFixed(3) + ' km' : '';
        var listed = l.listed ? l.listed.split('T')[0] : '\u2014';
        var yearBuilt = l.yearBuilt || '\u2014';
        var sqmEn = 'sqm', sqmTh = '\u0e15\u0e23.\u0e21.';
        var bedsEn = 'beds', bedsTh = '\u0e2b\u0e49\u0e2d\u0e07\u0e19\u0e2d\u0e19';
        var bathsEn = 'baths', bathsTh = '\u0e2b\u0e49\u0e2d\u0e07\u0e19\u0e49\u0e33';
        var listedEn = 'Listed', listedTh = '\u0e25\u0e07\u0e1b\u0e49\u0e32\u0e22\u0e40\u0e21\u0e37\u0e48\u0e2d';
        var ybEn = 'Year built', ybTh = '\u0e1b\u0e35\u0e17\u0e35\u0e48\u0e2a\u0e23\u0e49\u0e32\u0e07';

        return '<div class="card-header"><h3 class="card-title">' + esc(l.name) + '</h3>' + tierPill + '</div>'
          + '<div class="card-price-row"><span class="card-price">' + fmtThb(l.price) + '</span><span class="card-price-unit">' + fmtPpsm(l.pricePerSqm) + '<span data-en="/sqm" data-th="/\u0e15\u0e23.\u0e21.">/' + (lang === 'en' ? 'sqm' : '\u0e15\u0e23.\u0e21.') + '</span></span></div>'
          + '<div class="card-stats">'
          + '<div class="card-stat"><strong>' + (l.area != null ? l.area.toFixed(1) : '\u2014') + '</strong><span data-en="' + sqmEn + '" data-th="' + sqmTh + '">' + (lang === 'en' ? sqmEn : sqmTh) + '</span></div>'
          + '<div class="card-stat"><strong>' + (l.beds != null ? l.beds : '\u2014') + '</strong><span data-en="' + bedsEn + '" data-th="' + bedsTh + '">' + (lang === 'en' ? bedsEn : bedsTh) + '</span></div>'
          + '<div class="card-stat"><strong>' + (l.baths != null ? l.baths : '\u2014') + '</strong><span data-en="' + bathsEn + '" data-th="' + bathsTh + '">' + (lang === 'en' ? bathsEn : bathsTh) + '</span></div>'
          + '</div>'
          + '<div class="card-station line-' + lineSlug + '"><span class="line-dot" style="background:' + lineColor + ';"></span><strong>' + esc(l.station) + '</strong> <span>(' + kmStr + ')</span></div>'
          + uv
          + '<div class="card-listed"><span data-en="' + listedEn + '" data-th="' + listedTh + '">' + (lang === 'en' ? listedEn : listedTh) + '</span>: ' + listed + ' \u00b7 <span data-en="' + ybEn + '" data-th="' + ybTh + '">' + (lang === 'en' ? ybEn : ybTh) + '</span>: ' + esc(yearBuilt) + '</div>';
      }

      function cardHtml(l) {
        var lineSlug = slugify(l.line);
        var photo = l.thumbnail
          ? '<div class="card-photo-wrap"><img src="' + esc(l.thumbnail) + '" alt="" loading="lazy"></div>'
          : '<div class="card-photo-wrap no-photo"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 21h18M5 21V7l8-4 8 4v14M8 21v-9a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v9"/></svg></div>';
        return '<article class="listing-card tier-' + l.tier + ' line-' + lineSlug + '" data-id="' + esc(l.id) + '" data-tier="' + l.tier + '" data-line="' + lineSlug + '" data-lat="' + l.lat + '" data-lng="' + l.lng + '">'
          + photo
          + '<div class="card-body">' + cardInnerHtml(l) + '</div></article>';
      }
      window._cardHtml = cardHtml;

      function popupHtml(l) {
        var photo = l.thumbnail
          ? '<div class="card-photo-wrap"><img src="' + esc(l.thumbnail) + '" alt="" loading="lazy"></div>'
          : '<div class="card-photo-wrap no-photo"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 21h18M5 21V7l8-4 8 4v14M8 21v-9a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v9"/></svg></div>';
        var lang = window._lang || 'en';
        var ctaEn = 'View on FazWaz \u2192', ctaTh = '\u0e14\u0e39\u0e1a\u0e19 FazWaz \u2192';
        return '<div class="listing-popup-card">' + photo + '<div class="card-body">' + cardInnerHtml(l)
          + '<a class="card-cta" href="' + esc(l.detailUrl) + '" target="_blank"><span data-en="' + ctaEn + '" data-th="' + ctaTh + '">' + (lang === 'en' ? ctaEn : ctaTh) + '</span></a>'
          + '</div></div>';
      }
      window._popupHtml = popupHtml;

      ready(function() {
        // --- Load listing data ---
        var dataEl = document.getElementById('listingsData');
        window._listings = dataEl ? JSON.parse(dataEl.textContent) : [];
        window._listingsById = {};
        window._listings.forEach(function(l) { window._listingsById[l.id] = l; });

        // --- Move lang toggle into header ---
        var langToggle = document.getElementById('langToggle');
        var langContainer = document.getElementById('langToggleContainer');
        if (langToggle && langContainer) {
          langToggle.style.position = 'static';
          langContainer.appendChild(langToggle);
        }

        // --- Move narrative toggle ---
        var narrativeToggle = document.getElementById('narrativeToggle');
        var narrativeContainer = document.getElementById('narrativeToggleContainer');
        if (narrativeToggle && narrativeContainer) {
          narrativeToggle.style.position = 'static';
          narrativeToggle.style.width = '100%';
          narrativeToggle.innerHTML = '<span data-en="Read the analysis" data-th="\u0e2d\u0e48\u0e32\u0e19\u0e1a\u0e17\u0e27\u0e34\u0e40\u0e04\u0e23\u0e32\u0e30\u0e2b\u0e4c">Read the analysis</span>';
          narrativeContainer.appendChild(narrativeToggle);
        }

        // --- Tab switching ---
        var tabBtns = document.querySelectorAll('.tab-btn');
        tabBtns.forEach(function(btn) {
          btn.addEventListener('click', function() {
            var tab = btn.getAttribute('data-tab');
            tabBtns.forEach(function(b) {
              b.classList.toggle('active', b === btn);
              b.setAttribute('aria-selected', b === btn);
            });
            document.querySelectorAll('.tab-panel').forEach(function(p) {
              p.classList.toggle('active', p.id === tab + 'Panel');
            });
            if (window.map) window.map.invalidateSize();
          });
        });

        // --- Virtualized sidebar ---
        window._filtered = [];
        window._renderIdx = 0;
        var lineFilter = document.getElementById('lineFilter');
        var uvOnly = document.getElementById('undervaluedOnly');
        var showAll = document.getElementById('showAllCards');
        var searchInput = document.getElementById('refineSearch');
        var sortSelect = document.getElementById('sortSelect');
        var priceSlider = document.getElementById('priceSlider');
        var distSlider = document.getElementById('distSlider');
        var priceInput = document.getElementById('priceInput');
        var distInput = document.getElementById('distInput');
        var _priceMax = 0, _distMax = 0;
        for (var i = 0; i < window._listings.length; i++) {
          if (window._listings[i].price && window._listings[i].price > _priceMax) _priceMax = window._listings[i].price;
          if (window._listings[i].stationKm != null && window._listings[i].stationKm > _distMax) _distMax = window._listings[i].stationKm;
        }
        if (priceSlider) { priceSlider.max = Math.ceil(_priceMax / 1000); priceSlider.step = 1000; priceSlider.value = priceSlider.max; }
        if (distSlider) { distSlider.max = Math.ceil(_distMax * 10) / 10; distSlider.step = 0.1; distSlider.value = distSlider.max; }
        syncPriceInputs();

        function syncPriceInputs() {
          var pv = priceSlider ? parseFloat(priceSlider.value) : Infinity;
          var dv = distSlider ? parseFloat(distSlider.value) : Infinity;
          var pMax = priceSlider ? parseFloat(priceSlider.max) : Infinity;
          var dMax = distSlider ? parseFloat(distSlider.max) : Infinity;
          if (priceInput) priceInput.value = (pv >= pMax) ? '' : Math.round(pv * 1000);
          if (distInput) distInput.value = (dv >= dMax) ? '' : dv.toFixed(1);
        }

        function applyFilters() {
          var selectedLine = lineFilter ? lineFilter.value : '';
          var onlyUv = uvOnly ? uvOnly.checked : false;
          var showFair = showAll ? showAll.checked : false;
          var q = searchInput ? searchInput.value.toLowerCase().trim() : '';
          var maxPrice = priceSlider ? parseFloat(priceSlider.value) * 1000 : Infinity;
          var maxDist = distSlider ? parseFloat(distSlider.value) : Infinity;
          var priceAny = priceSlider && parseFloat(priceSlider.value) >= parseFloat(priceSlider.max);
          var distAny = distSlider && parseFloat(distSlider.value) >= parseFloat(distSlider.max);

          window._filtered = window._listings.filter(function(l) {
            var lineMatch = !selectedLine || (l.lineSlugs && l.lineSlugs.indexOf(selectedLine) !== -1);
            var uvMatch = !onlyUv || l.tier !== 'fair';
            var fairMatch = l.tier !== 'fair' || showFair;
            var priceMatch = priceAny || (l.price != null && l.price <= maxPrice);
            var distMatch = distAny || (l.stationKm != null && l.stationKm <= maxDist);
            var searchMatch = !q || (l.name && l.name.toLowerCase().indexOf(q) !== -1) || (l.station && l.station.toLowerCase().indexOf(q) !== -1) || (l.line && l.line.toLowerCase().indexOf(q) !== -1);
            return lineMatch && uvMatch && fairMatch && priceMatch && distMatch && searchMatch;
          });

          sortFiltered();

          window._renderIdx = 0;
          var container = document.getElementById('listingScroll');
          if (container) container.innerHTML = '';
          if (window._filtered.length === 0) {
            renderEmptyState();
          } else {
            renderBatch();
          }
          updateResultCount();
          updateMarkerVisibility();
          updateFilterStats();
          renderScatter();
          renderTopPicks();
        }

        function sortFiltered() {
          var mode = sortSelect ? sortSelect.value : 'value';
          var tierOrder = {strong: 0, good: 1, borderline: 2, fair: 3};
          if (mode === 'value') {
            window._filtered.sort(function(a, b) {
              var td = (tierOrder[a.tier] || 3) - (tierOrder[b.tier] || 3);
              if (td !== 0) return td;
              return (a.price || Infinity) - (b.price || Infinity);
            });
          } else if (mode === 'price_asc') {
            window._filtered.sort(function(a, b) { return (a.price || Infinity) - (b.price || Infinity); });
          } else if (mode === 'price_desc') {
            window._filtered.sort(function(a, b) { return (b.price || 0) - (a.price || 0); });
          } else if (mode === 'distance') {
            window._filtered.sort(function(a, b) { return (a.stationKm || Infinity) - (b.stationKm || Infinity); });
          } else if (mode === 'deal') {
            window._filtered.sort(function(a, b) { return (b.undervaluedPct || 0) - (a.undervaluedPct || 0); });
          } else if (mode === 'newest') {
            window._filtered.sort(function(a, b) { return (b.listed || '').localeCompare(a.listed || ''); });
          }
        }

        function renderBatch() {
          var container = document.getElementById('listingScroll');
          if (!container) return;
          var end = Math.min(window._renderIdx + RENDER_BATCH, window._filtered.length);
          var frag = '';
          for (; window._renderIdx < end; window._renderIdx++) {
            frag += cardHtml(window._filtered[window._renderIdx]);
          }
          if (frag) container.insertAdjacentHTML('beforeend', frag);
        }

        function updateResultCount() {
          var el = document.getElementById('resultCount');
          if (el) el.textContent = window._filtered.length + ' listings';
        }

        function updateMarkerVisibility() {
          if (!window._allMarkerData || window._allMarkerData.length === 0) return;
          var selectedLine = lineFilter ? lineFilter.value : '';
          var onlyUv = uvOnly ? uvOnly.checked : false;
          var visibleIds = {};
          window._filtered.forEach(function(l) { visibleIds[l.id] = true; });

          window._allMarkerData.forEach(function(md) {
            var visible = !!visibleIds[md.id];
            var inCluster = md.clusterGroup.hasLayer(md.marker);
            if (visible && !inCluster) md.clusterGroup.addLayer(md.marker);
            else if (!visible && inCluster) md.clusterGroup.removeLayer(md.marker);
          });
        }

        // --- Card selection (highlight) ---
        function selectCard(id) {
          document.querySelectorAll('.listing-card.selected').forEach(function(c) { c.classList.remove('selected'); });
          if (!id) return;
          var card = document.querySelector('.listing-card[data-id="' + CSS.escape(id) + '"]');
          if (card) {
            card.classList.add('selected');
            card.scrollIntoView({behavior: 'smooth', block: 'nearest'});
          }
        }

        // --- Live filtered stats ---
        function updateFilterStats() {
          var el = document.getElementById('filterStats');
          if (!el) return;
          var arr = window._filtered;
          if (arr.length === 0) { el.innerHTML = '<div class="stat-card"><div class="stat-card-val">0</div><div class="stat-card-lbl">Listings</div></div>'; return; }
          var ppsmArr = arr.map(function(l) { return l.pricePerSqm; }).filter(function(v) { return v != null; }).sort(function(a,b){return a-b;});
          var distArr = arr.map(function(l) { return l.stationKm; }).filter(function(v) { return v != null; });
          var uvCount = arr.filter(function(l) { return l.tier !== 'fair'; }).length;
          var med = ppsmArr.length ? ppsmArr[Math.floor(ppsmArr.length / 2)] : null;
          var avgD = distArr.length ? distArr.reduce(function(s,v){return s+v;},0) / distArr.length : null;
          var pctUv = arr.length ? (uvCount / arr.length * 100) : 0;
          function fmtP(v) { return v == null ? '\u2014' : '\u0e3f' + Math.round(v).toLocaleString(); }
          function fmtKm(v) { return v == null ? '\u2014' : v.toFixed(2) + ' km'; }
          el.innerHTML =
            '<div class="stat-card"><div class="stat-card-val">' + arr.length + '</div><div class="stat-card-lbl">Listings</div></div>' +
            '<div class="stat-card"><div class="stat-card-val">' + pctUv.toFixed(0) + '%</div><div class="stat-card-lbl">Undervalued</div></div>' +
            '<div class="stat-card"><div class="stat-card-val">' + fmtP(med) + '</div><div class="stat-card-lbl">Median \u0e3f/sqm</div></div>' +
            '<div class="stat-card"><div class="stat-card-val">' + fmtKm(avgD) + '</div><div class="stat-card-lbl">Avg distance</div></div>';
        }

        // --- Scatter chart (price/sqm vs distance) ---
        function renderScatter() {
          var svg = document.getElementById('scatterChart');
          if (!svg) return;
          var arr = window._filtered.filter(function(l) { return l.pricePerSqm != null && l.stationKm != null; });
          if (arr.length === 0) { svg.innerHTML = '<text x="120" y="90" text-anchor="middle" fill="#8b95a5" font-size="11">No data</text>'; return; }
          var W = 240, H = 180, pad = 28;
          var maxP = 0, maxD = 0;
          arr.forEach(function(l) { if (l.pricePerSqm > maxP) maxP = l.pricePerSqm; if (l.stationKm > maxD) maxD = l.stationKm; });
          if (maxP === 0) maxP = 1; if (maxD === 0) maxD = 1;
          var dots = arr.slice(0, 500);
          var tCol = {strong: '#2ecc71', good: '#1abc9c', borderline: '#f1c40f', fair: '#94a3b8'};
          var html = '';
          html += '<line class="axis" x1="' + pad + '" y1="' + (H-pad) + '" x2="' + (W-4) + '" y2="' + (H-pad) + '"/>';
          html += '<line class="axis" x1="' + pad + '" y1="4" x2="' + pad + '" y2="' + (H-pad) + '"/>';
          html += '<text class="axis-label" x="' + (W/2) + '" y="' + (H-2) + '" text-anchor="middle">distance (km)</text>';
          html += '<text class="axis-label" x="10" y="' + (H/2) + '" transform="rotate(-90 10 ' + (H/2) + ')" text-anchor="middle">\u0e3f/sqm</text>';
          html += '<text class="axis-label" x="' + pad + '" y="' + (H-pad+10) + '">' + maxD.toFixed(1) + '</text>';
          dots.forEach(function(l) {
            var x = pad + (l.stationKm / maxD) * (W - pad - 4);
            var y = (H - pad) - (l.pricePerSqm / maxP) * (H - pad - 4);
            html += '<circle class="dot" cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="2.5" fill="' + (tCol[l.tier] || tCol.fair) + '"/>';
          });
          svg.innerHTML = html;
        }

        // --- Top undervalued picks ---
        function renderTopPicks() {
          var el = document.getElementById('topPicks');
          if (!el) return;
          var arr = window._filtered.filter(function(l) { return l.undervaluedPct && l.undervaluedPct > 0; });
          arr.sort(function(a, b) { return b.undervaluedPct - a.undervaluedPct; });
          arr = arr.slice(0, 5);
          if (arr.length === 0) { el.innerHTML = '<p style="font-size:12px;color:var(--muted);">No undervalued listings in current view.</p>'; return; }
          var html = '';
          arr.forEach(function(l, i) {
            html += '<div class="top-pick" data-id="' + esc(l.id) + '"><span class="top-pick-rank">' + (i+1) + '</span><span class="top-pick-name">' + esc(l.name) + '</span><span class="top-pick-pct">-' + l.undervaluedPct.toFixed(1) + '%</span></div>';
          });
          el.innerHTML = html;
          el.querySelectorAll('.top-pick').forEach(function(item) {
            item.addEventListener('click', function() { flyToListing(item.getAttribute('data-id')); });
          });
        }

        // --- Empty state ---
        function renderEmptyState() {
          var container = document.getElementById('listingScroll');
          if (!container) return;
          container.innerHTML = '<div id="emptyState"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg><p>No listings match the current filters.</p><button onclick="document.getElementById(\'resetFilters\').click()">Reset filters</button></div>';
        }

        // --- Fit all results ---
        function fitAllResults() {
          if (!window.map || window._filtered.length === 0) return;
          var bounds = L.latLngBounds(window._filtered.map(function(l) { return [l.lat, l.lng]; }));
          window.map.fitBounds(bounds, {padding: [40, 40]});
        }

        // --- Teleport: card click -> fly + open popup ---
        function flyToListing(id) {
          var l = window._listingsById[id];
          if (!l || !window.map) return;
          var marker = window._markersById ? window._markersById[id] : null;

          if (marker) {
            var cg = null;
            window.map.eachLayer(function(layer) {
              if (layer instanceof L.MarkerClusterGroup && layer.hasLayer(marker)) cg = layer;
            });
            if (cg) {
              cg.zoomToShowLayer(marker, function() {
                marker.openPopup();
              });
            } else {
              window.map.flyTo([l.lat, l.lng], 16, {duration: 0.4});
              setTimeout(function() { marker.openPopup(); }, 500);
            }
          } else {
            window.map.flyTo([l.lat, l.lng], 16, {duration: 0.4});
            L.popup({maxWidth: 320}).setLatLng([l.lat, l.lng]).setContent(popupHtml(l)).openOn(window.map);
          }
        }

        // Event delegation for card clicks
        var listingPanel = document.getElementById('listingPanel');
        if (listingPanel) {
          listingPanel.addEventListener('click', function(e) {
            var card = e.target.closest('.listing-card');
            if (!card) return;
            var id = card.getAttribute('data-id');
            if (id) { selectCard(id); flyToListing(id); }
            if (window.innerWidth <= 900) {
              document.getElementById('appSidebar').classList.remove('open');
            }
          });
        }

        // --- Marker click -> scroll to + highlight card ---
        function bindMarkerCardSync() {
          if (!window._markersById) return;
          for (var id in window._markersById) {
            var marker = window._markersById[id];
            marker.off('click').on('click', function(cid) {
              return function() { selectCard(cid); };
            }(id));
          }
        }

        // --- Bind popups to markers after collection ---
        function bindMarkerPopups() {
          if (!window._markersById) return;
          for (var id in window._markersById) {
            var marker = window._markersById[id];
            var l = window._listingsById[id];
            if (l && !marker.getPopup()) {
              marker.bindPopup(popupHtml(l), {maxWidth: 320});
            }
          }
        }

        // --- Show all toggle ---
        if (showAll) {
          showAll.addEventListener('change', function() {
            applyFilters();
          });
        }

        // --- Line filter ---
        if (lineFilter) {
          lineFilter.addEventListener('change', applyFilters);
        }

        // --- UV-only toggle ---
        if (uvOnly) {
          uvOnly.addEventListener('change', applyFilters);
        }

        // --- Search input ---
        if (searchInput) {
          var _searchTimer = null;
          searchInput.addEventListener('input', function() {
            clearTimeout(_searchTimer);
            _searchTimer = setTimeout(applyFilters, 250);
          });
        }

        // --- Sort select ---
        if (sortSelect) {
          sortSelect.addEventListener('change', function() {
            sortFiltered();
            window._renderIdx = 0;
            var container = document.getElementById('listingScroll');
            if (container) container.innerHTML = '';
            if (window._filtered.length === 0) renderEmptyState(); else renderBatch();
          });
        }

        // --- Price / distance sliders ---
        if (priceSlider) priceSlider.addEventListener('input', function() { syncPriceInputs(); clearTimeout(window._sliderTimer); window._sliderTimer = setTimeout(applyFilters, 200); });
        if (distSlider) distSlider.addEventListener('input', function() { syncPriceInputs(); clearTimeout(window._sliderTimer); window._sliderTimer = setTimeout(applyFilters, 200); });

        // --- Price / distance manual inputs ---
        if (priceInput) {
          priceInput.addEventListener('input', function() {
            var v = priceInput.value.trim();
            if (!v || isNaN(parseFloat(v))) {
              if (priceSlider) priceSlider.value = priceSlider.max;
            } else {
              if (priceSlider) priceSlider.value = Math.max(0, Math.min(parseFloat(v) / 1000, parseFloat(priceSlider.max)));
            }
            clearTimeout(window._sliderTimer);
            window._sliderTimer = setTimeout(applyFilters, 300);
          });
        }
        if (distInput) {
          distInput.addEventListener('input', function() {
            var v = distInput.value.trim();
            if (!v || isNaN(parseFloat(v))) {
              if (distSlider) distSlider.value = distSlider.max;
            } else {
              if (distSlider) distSlider.value = Math.max(0, Math.min(parseFloat(v), parseFloat(distSlider.max)));
            }
            clearTimeout(window._sliderTimer);
            window._sliderTimer = setTimeout(applyFilters, 300);
          });
        }

        // --- Fit all button ---
        var fitAllBtn = document.getElementById('fitAllBtn');
        if (fitAllBtn) fitAllBtn.addEventListener('click', fitAllResults);

        // --- Dark mode toggle ---
        var darkBtn = document.getElementById('darkToggle');
        if (darkBtn) {
          darkBtn.addEventListener('click', function() {
            var isDark = document.documentElement.getAttribute('data-theme') === 'dark';
            if (isDark) { document.documentElement.removeAttribute('data-theme'); darkBtn.textContent = 'Dark'; }
            else { document.documentElement.setAttribute('data-theme', 'dark'); darkBtn.textContent = 'Light'; }
            if (window.map) setTimeout(function() { window.map.invalidateSize(); }, 100);
          });
        }

        // --- Reset ---
        var resetBtn = document.getElementById('resetFilters');
        if (resetBtn) {
          resetBtn.addEventListener('click', function() {
            if (lineFilter) lineFilter.value = '';
            if (uvOnly) uvOnly.checked = true;
            if (showAll) showAll.checked = false;
            if (searchInput) searchInput.value = '';
            if (sortSelect) sortSelect.value = 'value';
            if (priceSlider) priceSlider.value = priceSlider.max;
            if (distSlider) distSlider.value = distSlider.max;
            if (priceInput) priceInput.value = '';
            if (distInput) distInput.value = '';
            setColorMode('price');
            applyFilters();
          });
        }

        // --- IntersectionObserver for progressive loading ---
        var sentinel = document.getElementById('listSentinel');
        if (sentinel && 'IntersectionObserver' in window) {
          var panel = document.getElementById('listingPanel');
          var obs = new IntersectionObserver(function(entries) {
            if (entries[0].isIntersecting && window._renderIdx < window._filtered.length) {
              renderBatch();
            }
          }, {root: panel, rootMargin: '300px'});
          obs.observe(sentinel);
        }

        // --- Mobile view ---
        window.switchMobileView = function(mode) {
          var sidebar = document.getElementById('appSidebar');
          if (mode === 'list') sidebar.classList.add('open');
          else sidebar.classList.remove('open');
          if (window.map) setTimeout(function() { window.map.invalidateSize(); }, 260);
        };

        var sidebar = document.getElementById('appSidebar');
        if (sidebar && window.innerWidth <= 900) {
          sidebar.classList.add('open');
        }

        // --- Init color mode ---
        if (typeof initColorMode === 'function') initColorMode();

        // --- Initial render ---
        applyFilters();

        // --- Bind popups after markers are collected ---
        setTimeout(function() {
          if (window._listingMarkers.length === 0 && typeof collectListingMarkers === 'function') {
            collectListingMarkers();
          }
          bindMarkerPopups();
          bindMarkerCardSync();
          if (typeof recolorMarkers === 'function') recolorMarkers();
        }, 700);

        if (window.map) setTimeout(function() { window.map.invalidateSize(); }, 300);
      });
    })();
    </script>
    """
    m.get_root().html.add_child(folium.Element(js))


def build_station_popup(station: dict[str, Any]) -> str:
    lines_html = ""
    for line in station["lines"]:
        color = LINE_COLORS.get(line, "#888888")
        line_th = LINE_NAMES_TH.get(line, line)
        lines_html += f'<span class="line-dot" style="background:{color};"></span><span data-en="{line}" data-th="{line_th}">{line}</span> '

    status_en = "Operational" if station["operational"] else "Planned"
    status_th = "เปิดให้บริการ" if station["operational"] else "กำลังก่อสร้าง"
    status_class = "status-operational" if station["operational"] else "status-planned"

    return f"""<div class="station-popup-card">
<div class="station-popup-name"><span data-en="{station['name']}" data-th="{station['name_th']}">{station['name']}</span></div>
<div class="station-popup-lines">{lines_html}</div>
<div class="station-popup-meta">
  <span class="ref-chip">{station['ref']}</span>
  <span class="status-pill {status_class}" data-en="{status_en}" data-th="{status_th}">{status_en}</span>
</div>
</div>"""


def build_transit_layer(
    stations_by_line: dict[str, list[list[dict[str, Any]]]],
    all_stations: list[dict[str, Any]],
) -> dict[str, folium.FeatureGroup]:
    line_fgs: dict[str, folium.FeatureGroup] = {}

    station_by_name: dict[str, list[dict[str, Any]]] = {}
    for s in all_stations:
        station_by_name.setdefault(s["name"], []).append(s)

    for line_name, branches in stations_by_line.items():
        color = LINE_COLORS.get(line_name, "#888888")
        fg = folium.FeatureGroup(name=line_name)

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

        for branch in branches:
            for station in branch:
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

        line_fgs[line_name] = fg

    return line_fgs


def _fmt_thb(value: float) -> str:
    return f"฿{value:,.0f}"


def _fmt_date(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "—"
    return dt.strftime("%Y-%m-%d")


def _card_photo_html(row: pd.Series) -> str:
    if pd.notna(row.get("thumbnail")) and row.get("thumbnail"):
        return f'<div class="card-photo-wrap"><img src="{row["thumbnail"]}" alt="" loading="lazy"></div>'
    return (
        '<div class="card-photo-wrap no-photo">'
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">'
        '<path d="M3 21h18M5 21V7l8-4 8 4v14M8 21v-9a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v9"/>'
        '</svg></div>'
    )


def _card_tier_html(row: pd.Series) -> str:
    if "value_tier" not in row or pd.isna(row.get("value_tier")):
        return ""
    tier = str(row.get("value_tier"))
    if tier not in TIER_LABELS:
        return ""
    en_label, th_label = TIER_LABELS[tier]
    color = TIER_COLORS.get(tier, "#888888") or "#888888"
    return (
        f'<span class="tier-pill" style="background:{color};" '
        f'data-en="{en_label}" data-th="{th_label}">{en_label}</span>'
    )


def _card_body_html(row: pd.Series, is_ghost: bool = False) -> str:
    line_en = row.get("nearest_station_line", "—")
    line_th = LINE_NAMES_TH.get(line_en, line_en)
    line_color = LINE_COLORS.get(line_en, "#888888")
    line_slug = _slug(line_en)

    tier_html = _card_tier_html(row)

    undervalued_html = ""
    if row.get("is_undervalued") and pd.notna(row.get("undervalued_by_pct")):
        pct = float(row["undervalued_by_pct"])
        if pct > 0:
            undervalued_html = (
                f'<div class="undervalued-strip">'
                f'<span data-en="Undervalued by" data-th="ต่ำกว่าโมเดล">Undervalued by</span> {pct:.1f}%'
                f'</div>'
            )

    ghost_html = ""
    if is_ghost and pd.notna(row.get("listed_dt")):
        import datetime as _dt

        days = (_dt.date.today() - row["listed_dt"].date()).days
        ghost_html = (
            f'<div class="ghost-strip">'
            f'<span data-en="GHOST · {days} days on market" '
            f'data-th="ค้างนาน · {days} วันที่ค้าง">GHOST · {days} days on market</span>'
            f'</div>'
        )

    listed_str = _fmt_date(row.get("listed_dt"))
    year_built = row.get("year_built") or "—"
    price_per_sqm = _fmt_thb(row["price_per_sqm"]) if pd.notna(row.get("price_per_sqm")) else "—"

    return f"""
<div class="card-header">
  <h3 class="card-title">{row['name']}</h3>
  {tier_html}
</div>
<div class="card-price-row">
  <span class="card-price">{_fmt_thb(row['price_thb'])}</span>
  <span class="card-price-unit">{price_per_sqm}<span data-en="/sqm" data-th="/ตร.ม.">/sqm</span></span>
</div>
<div class="card-stats">
  <div class="card-stat"><strong>{row['area_sqm_num']:.1f}</strong><span data-en="sqm" data-th="ตร.ม.">sqm</span></div>
  <div class="card-stat"><strong>{row['bedrooms']}</strong><span data-en="beds" data-th="ห้องนอน">beds</span></div>
  <div class="card-stat"><strong>{row['bathrooms']}</strong><span data-en="baths" data-th="ห้องน้ำ">baths</span></div>
</div>
<div class="card-station line-{line_slug}">
  <span class="line-dot" style="background:{line_color};"></span>
  <strong>{row['nearest_station']}</strong>
  <span>({row['nearest_station_km']:.3f} km)</span>
</div>
{undervalued_html}
{ghost_html}
<div class="card-listed"><span data-en="Listed" data-th="ลงป้ายเมื่อ">Listed</span>: {listed_str} · <span data-en="Year built" data-th="ปีที่สร้าง">Year built</span>: {year_built}</div>
"""


def build_listing_card_html(row: pd.Series, idx: int, is_ghost: bool = False) -> str:
    tier = str(row.get("value_tier", "fair")) if "value_tier" in row else "fair"
    if tier not in ("strong", "good", "borderline"):
        tier = "fair"
    line_slug = _slug(row.get("nearest_station_line", ""))
    popup_html = build_popup_html(row, is_ghost=is_ghost).replace('"', '&quot;')
    lat = row["latitude"]
    lng = row["longitude"]
    return (
        f'<article class="listing-card tier-{tier} line-{line_slug}" '
        f'data-idx="{idx}" data-tier="{tier}" data-line="{line_slug}" '
        f'data-lat="{lat}" data-lng="{lng}" data-popup="{popup_html}">'
        f'{_card_photo_html(row)}'
        f'<div class="card-body">{_card_body_html(row, is_ghost=is_ghost)}</div>'
        f'</article>'
    )


def build_popup_html(row: pd.Series, is_ghost: bool = False) -> str:
    return (
        '<div class="listing-popup-card">'
        f'{_card_photo_html(row)}'
        '<div class="card-body">'
        f'{_card_body_html(row, is_ghost=is_ghost)}'
        f'<a class="card-cta" href="{row["detail_url"]}" target="_blank">'
        '<span data-en="View on FazWaz →" data-th="ดูบน FazWaz →">View on FazWaz →</span>'
        '</a>'
        '</div></div>'
    )


def _fmt_price_bubble(value: float) -> str:
    if pd.isna(value):
        return "฿—"
    if value >= 1_000_000:
        return f"฿{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"฿{value / 1_000:.0f}K"
    return f"฿{value:,.0f}"


def _fmt_price_spelled(value: float) -> str:
    if pd.isna(value):
        return "—"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f} billion baht"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} million baht"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K baht"
    return f"฿{value:,.0f}"


def build_listing_payload(df: pd.DataFrame) -> str:
    """Build compact JSON payload of listing data for client-side rendering."""
    import json as _json

    records: list[dict] = []
    for idx, row in df.iterrows():
        if row.get("is_ghost", False):
            continue
        tier = str(row.get("value_tier", "fair"))
        if tier not in ("strong", "good", "borderline"):
            tier = "fair"
        rec = {
            "id": str(row.get("listing_id", idx)),
            "name": str(row.get("name", "")),
            "price": float(row["price_thb"]) if pd.notna(row.get("price_thb")) else None,
            "pricePerSqm": float(row["price_per_sqm"]) if pd.notna(row.get("price_per_sqm")) else None,
            "area": float(row["area_sqm_num"]) if pd.notna(row.get("area_sqm_num")) else None,
            "beds": int(row["bedrooms"]) if pd.notna(row.get("bedrooms")) else None,
            "baths": int(row["bathrooms"]) if pd.notna(row.get("bathrooms")) else None,
            "station": str(row.get("nearest_station", "")),
            "stationKm": float(row["nearest_station_km"]) if pd.notna(row.get("nearest_station_km")) else None,
            "line": str(row.get("nearest_station_line", "")),
            "lineSlugs": [_slug(s.strip()) for s in str(row.get("nearest_station_line", "")).split(",") if s.strip()],
            "tier": tier,
            "undervalued": bool(row.get("is_undervalued", False)),
            "undervaluedPct": float(row["undervalued_by_pct"]) if pd.notna(row.get("undervalued_by_pct")) else None,
            "thumbnail": str(row.get("thumbnail")) if pd.notna(row.get("thumbnail")) else "",
            "detailUrl": str(row.get("detail_url", "")),
            "listed": row["listed_dt"].isoformat() if pd.notna(row.get("listed_dt")) else "",
            "yearBuilt": str(row.get("year_built")) if pd.notna(row.get("year_built")) and row.get("year_built") else "",
            "lat": float(row["latitude"]),
            "lng": float(row["longitude"]),
        }
        records.append(rec)
    return _json.dumps(records, ensure_ascii=False)


def inject_listing_data(m: folium.Map, payload_json: str) -> None:
    """Inject listing data as a JSON script tag for client-side rendering."""
    script = f'<script type="application/json" id="listingsData">{payload_json}</script>'
    m.get_root().html.add_child(folium.Element(script))


def build_listing_markers(df: pd.DataFrame) -> tuple[dict[str, folium.FeatureGroup], list[dict]]:
    groups: dict[str, folium.FeatureGroup] = {
        "Undervalued: strong": folium.FeatureGroup(name="Undervalued: strong"),
        "Undervalued: good": folium.FeatureGroup(name="Undervalued: good"),
        "Undervalued: borderline": folium.FeatureGroup(name="Undervalued: borderline"),
        "Other listings": folium.FeatureGroup(name="Other listings"),
    }
    clusters: dict[str, MarkerCluster] = {
        name: MarkerCluster(name=name) for name in groups
    }
    color_data: list[dict] = []
    has_tier = "value_tier" in df.columns

    for _, row in df.iterrows():
        if row.get("is_ghost", False):
            continue

        tier = str(row.get("value_tier", "fair")) if has_tier else "fair"
        if tier not in ("strong", "good", "borderline"):
            tier = "fair"
        group_name = f"Undervalued: {tier}" if tier != "fair" else "Other listings"
        fg = groups[group_name]
        cluster = clusters[group_name]

        price_bin = int(row.get("price_bin", 1))
        dist_bin = int(row.get("dist_bin", 1))
        line = row.get("nearest_station_line", "")
        line_color = LINE_COLORS.get(line, "#888888")

        price_color = BIN_COLORS[min(price_bin - 1, 3)]
        dist_color = BIN_COLORS[min(dist_bin - 1, 3)]
        tier_color = TIER_COLORS.get(tier)
        default_fill = tier_color if tier_color is not None else price_color

        line_slug = _slug(line)
        listing_id = str(row.get("listing_id", _))
        price_text = _fmt_price_bubble(row["price_thb"])

        tooltip_text = f"{row['name']} · {_fmt_price_spelled(row['price_thb'])}"
        if pd.notna(row.get("nearest_station_km")):
            tooltip_text += f" · {row['nearest_station_km']:.2f} km to {row['nearest_station']}"

        icon = folium.DivIcon(
            class_name=f"listing-marker marker-{tier} line-{line_slug}",
            html=(
                f'<div class="price-bubble" style="background-color:{default_fill};" '
                f'data-id="{listing_id}" data-tier="{tier}" data-line="{line_slug}">{price_text}</div>'
            ),
        )

        marker = folium.Marker(
            location=(row["latitude"], row["longitude"]),
            icon=icon,
            tooltip=tooltip_text,
        )

        cluster.add_child(marker)
        color_data.append({"price": price_color, "dist": dist_color, "line": line_color})

    for name, fg in groups.items():
        fg.add_child(clusters[name])

    return groups, color_data


def build_ghost_markers(df: pd.DataFrame) -> folium.FeatureGroup:
    fg = folium.FeatureGroup(name="ghost_listings", show=False)

    if "is_ghost" not in df.columns or not df["is_ghost"].any():
        return fg

    ghost_df = df[df["is_ghost"] == True]

    for _, row in ghost_df.iterrows():
        price_text = _fmt_price_bubble(row["price_thb"])
        icon = folium.DivIcon(
            class_name="ghost-marker",
            html=f'<div class="price-bubble ghost">{price_text}</div>',
        )
        folium.Marker(
            location=(row["latitude"], row["longitude"]),
            icon=icon,
            popup=folium.Popup(build_popup_html(row, is_ghost=True), max_width=320),
            tooltip=row["name"],
        ).add_to(fg)

    return fg


def inject_color_toggle(
    m: folium.Map, color_data: list[dict], unique_lines: list[str]
) -> None:
    import json as _json
    colors_json = _json.dumps(color_data)

    line_options = '<option value="">All lines</option>'
    for line in unique_lines:
        line_th = LINE_NAMES_TH.get(line, line)
        line_options += f'<option value="{_slug(line)}" data-en="{line}" data-th="{line_th}">{line}</option>'

    line_swatch_items = ""
    for line in unique_lines:
        color = LINE_COLORS.get(line, "#888888")
        line_th = LINE_NAMES_TH.get(line, line)
        line_swatch_items += f"""
        <div class="legend-row">
          <span class="legend-swatch" style="background:{color};"></span>
          <span data-en="{line}" data-th="{line_th}">{line}</span>
        </div>"""

    quantile_legend = ""
    labels = [UI_LABELS["budget"][0], UI_LABELS["below_median"][0], UI_LABELS["above_median"][0], UI_LABELS["premium"][0]]
    for color, label in zip(BIN_COLORS, labels):
        quantile_legend += f"""
        <div class="legend-row">
          <span class="legend-swatch" style="background:{color};"></span>
          <span data-en="{label}" data-th="{UI_LABELS.get(_slug(label), (label, label))[1]}">{label}</span>
        </div>"""

    toggle_html = f"""
    <div id="filterBar">
      <div class="filter-group">
        <label data-en="Color by" data-th="{UI_LABELS['color_by'][1]}">Color by</label>
        <div class="chip-segment" id="colorMode" role="radiogroup" data-value="price">
          <button type="button" class="chip-btn active" data-value="price" aria-pressed="true">
            <span data-en="Price" data-th="ราคา">Price</span>
          </button>
          <button type="button" class="chip-btn" data-value="distance" aria-pressed="false">
            <span data-en="Distance" data-th="ระยะทาง">Distance</span>
          </button>
          <button type="button" class="chip-btn" data-value="line" aria-pressed="false">
            <span data-en="Line" data-th="สาย">Line</span>
          </button>
        </div>
      </div>
      <div class="filter-group">
        <label data-en="Line" data-th="สาย">Line</label>
        <select id="lineFilter">{line_options}</select>
      </div>
      <label class="chip-toggle">
        <input type="checkbox" id="undervaluedOnly" checked>
        <span data-en="Undervalued only" data-th="{UI_LABELS['undervalued_only'][1]}">Undervalued only</span>
      </label>
      <button type="button" class="chip-btn reset" id="resetFilters" data-en="Reset" data-th="{UI_LABELS['reset'][1]}">Reset</button>
    </div>
    """

    legend_html = f"""
    <div id="legend-templates" style="display:none;">
      <div id="legend-quantile">{quantile_legend}</div>
      <div id="legend-line">{line_swatch_items}</div>
    </div>
    """

    line_colors_json = _json.dumps(LINE_COLORS)
    line_names_th_json = _json.dumps(LINE_NAMES_TH)

    recolor_js = f"""
    <script>
    (function() {{
      var _origLMap = L.map;
      L.map = function() {{
        var _m = _origLMap.apply(this, arguments);
        if (!window.map) window.map = _m;
        return _m;
      }};
      var _origMCG = L.markerClusterGroup;
      L.markerClusterGroup = function(opts) {{
        opts = opts || {{}};
        opts.iconCreateFunction = function(cluster) {{
          var c = cluster.getChildCount();
          var sz = c < 10 ? 's' : c < 50 ? 'm' : 'l';
          return L.divIcon({{
            html: '<div class="cluster-icon sz-' + sz + '">' + c + '</div>',
            className: 'cluster-wrapper',
            iconSize: [40, 40]
          }});
        }};
        return _origMCG.call(this, opts);
      }};
    }})();
    window._markerColors = {colors_json};
    window._lineColors = {line_colors_json};
    window._lineNamesTh = {line_names_th_json};
    window._listingMarkers = [];
    window._markersById = {{}};
    window._allMarkerData = [];
    window._ghostMarkers = [];

    function _markerId(m) {{
      var htmlStr = m.options.icon && m.options.icon.options && m.options.icon.options.html;
      var match = htmlStr && htmlStr.match(/data-id="([^"]+)"/);
      return match ? match[1] : null;
    }}

    function collectListingMarkers() {{
      if (!window.map) return;
      window._listingMarkers = [];
      window._markersById = {{}};
      window._allMarkerData = [];
      window.map.eachLayer(function(layer) {{
        if (layer instanceof L.MarkerClusterGroup) {{
          layer.eachLayer(function(m) {{
            window._listingMarkers.push(m);
            var id = _markerId(m);
            if (id) {{
              window._markersById[id] = m;
              var htmlStr = m.options.icon.options.html;
              var tierMatch = htmlStr.match(/data-tier="([^"]+)"/);
              var lineMatch = htmlStr.match(/data-line="([^"]+)"/);
              window._allMarkerData.push({{
                marker: m,
                clusterGroup: layer,
                id: id,
                tier: tierMatch ? tierMatch[1] : 'fair',
                lineSlug: lineMatch ? lineMatch[1] : ''
              }});
            }}
          }});
        }}
      }});
      window._ghostMarkers = [];
      window.map.eachLayer(function(layer) {{
        if (layer instanceof L.Marker && layer.options.icon && layer.options.icon.options.className === 'ghost-marker') {{
          window._ghostMarkers.push(layer);
        }}
      }});
    }}

    function getColorMode() {{
      var group = document.getElementById('colorMode');
      return group ? group.getAttribute('data-value') : 'price';
    }}

    function setColorMode(mode) {{
      var group = document.getElementById('colorMode');
      if (!group) return;
      group.setAttribute('data-value', mode);
      group.querySelectorAll('.chip-btn').forEach(function(btn) {{
        var active = btn.getAttribute('data-value') === mode;
        btn.classList.toggle('active', active);
        btn.setAttribute('aria-pressed', active);
      }});
      updateLegend(mode);
      recolorMarkers();
    }}

    function updateLegend(mode) {{
      var container = document.getElementById('analysisLegend');
      if (!container) return;
      var src = document.getElementById(mode === 'line' ? 'legend-line' : 'legend-quantile');
      container.innerHTML = src ? src.innerHTML : '';
    }}

    function recolorMarkers() {{
      var mode = getColorMode();
      if (window._listingMarkers.length === 0) collectListingMarkers();

      window._listingMarkers.forEach(function(marker, i) {{
        var cd = window._markerColors[i];
        if (!cd) return;
        var color = (mode === 'price') ? cd.price : (mode === 'distance') ? cd.dist : cd.line;
        var bubble = marker.getElement() ? marker.getElement().querySelector('.price-bubble') : null;
        if (bubble) bubble.style.backgroundColor = color;
      }});
    }}

    function initColorMode() {{
      var group = document.getElementById('colorMode');
      if (!group) return;
      group.querySelectorAll('.chip-btn').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          setColorMode(btn.getAttribute('data-value'));
        }});
      }});
      updateLegend(getColorMode());
    }}

    setTimeout(function() {{ collectListingMarkers(); recolorMarkers(); }}, 600);
    </script>
    """

    m.get_root().html.add_child(folium.Element(toggle_html))
    m.get_root().html.add_child(folium.Element(legend_html))
    m.get_root().html.add_child(folium.Element(recolor_js))


def inject_lang_toggle(m: folium.Map) -> None:
    button_html = """
    <button id="langToggle" class="chip-btn" onclick="switchLang()" style="position:fixed; top:10px; right:10px; z-index:10001;">
      <span data-en="TH" data-th="EN">TH</span>
    </button>
    """

    lang_js = """
    <script>
    window._lang = 'en';

    function switchLang() {
      window._lang = (window._lang === 'en') ? 'th' : 'en';
      var btn = document.getElementById('langToggle');
      var label = (window._lang === 'en') ? 'TH' : 'EN';
      var span = btn.querySelector('span[data-en][data-th]');
      if (span) span.textContent = label;

      document.querySelectorAll('[data-en][data-th]').forEach(function(el) {
        if (el === span) return;
        var en = el.getAttribute('data-en');
        var th = el.getAttribute('data-th');
        if (en && th) el.textContent = (window._lang === 'en') ? en : th;
      });

      var select = document.getElementById('lineFilter');
      if (select) {
        Array.from(select.options).forEach(function(opt) {
          if (opt.hasAttribute('data-en') && opt.hasAttribute('data-th')) {
            opt.textContent = (window._lang === 'en') ? opt.getAttribute('data-en') : opt.getAttribute('data-th');
          }
        });
      }

      // Re-render visible listing cards in new language
      if (window._filtered && window._renderIdx != null) {
        var container = document.getElementById('listingScroll');
        if (container) {
          var oldIdx = window._renderIdx;
          window._renderIdx = 0;
          container.innerHTML = '';
          var end = Math.min(oldIdx, window._filtered.length);
          var frag = '';
          for (; window._renderIdx < end; window._renderIdx++) {
            frag += window._cardHtml(window._filtered[window._renderIdx]);
          }
          if (frag) container.insertAdjacentHTML('beforeend', frag);
        }
      }

      // Re-bind marker popups in new language
      if (window._markersById && window._listingsById && window._popupHtml) {
        for (var id in window._markersById) {
          var marker = window._markersById[id];
          var l = window._listingsById[id];
          if (l && marker.getPopup()) {
            marker.setPopupContent(window._popupHtml(l));
          }
        }
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

    stations_by_line = sort_stations_by_line(stations)

    m = folium.Map(
        location=[13.7563, 100.5018],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    inject_design_system(m)

    # Stats for header
    total = len(df)
    pct_undervalued = 0.0
    if "is_undervalued" in df.columns:
        pct_undervalued = df["is_undervalued"].mean() * 100

    narrative_path = PROJECT_ROOT / "docs" / "narrative.md"
    meta_path = PROJECT_ROOT / "data" / "processed" / "narrative_meta.json"
    narrative_html = ""
    meta_lines: list[dict] = []
    if narrative_path.exists() and meta_path.exists():
        narrative_html = _markdown_to_html(narrative_path.read_text(encoding="utf-8"))
        with open(meta_path, encoding="utf-8") as f:
            meta_lines = json.load(f).get("lines", [])
    elif narrative_path.exists():
        narrative_html = _markdown_to_html(narrative_path.read_text(encoding="utf-8"))
    elif meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta_lines = json.load(f).get("lines", [])

    top_decay = 0.0
    if meta_lines:
        decay_values = [line.get("decay_pct_per_km", 0) for line in meta_lines if isinstance(line.get("decay_pct_per_km"), (int, float))]
        if decay_values:
            top_decay = min(decay_values)

    import datetime as _dt
    updated = _dt.date.today().isoformat()

    stats = {
        "total": total,
        "pct_undervalued": pct_undervalued,
        "top_decay": top_decay,
        "updated": updated,
    }

    inject_app_shell(m, stats)

    payload_json = build_listing_payload(df)
    inject_listing_data(m, payload_json)

    transit_fgs = build_transit_layer(stations_by_line, stations)
    for fg in transit_fgs.values():
        fg.add_to(m)

    listings_fgs, color_data = build_listing_markers(df)
    for fg in listings_fgs.values():
        fg.add_to(m)

    has_ghosts = "is_ghost" in df.columns and df["is_ghost"].any()
    if has_ghosts:
        ghost_fg = build_ghost_markers(df)
        ghost_fg.add_to(m)

    if narrative_html or meta_lines:
        inject_narrative_panel(m, narrative_html, meta_lines)

    folium.LayerControl(collapsed=True).add_to(m)

    if "nearest_station_line" in df.columns:
        _raw_lines = df["nearest_station_line"].dropna().unique()
        unique_lines = sorted({s.strip() for c in _raw_lines for s in str(c).split(",") if s.strip()})
    else:
        unique_lines = []
    inject_color_toggle(m, color_data, list(unique_lines))
    inject_lang_toggle(m)
    inject_bootstrap_js(m)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(args.output))
    print(f"Saved map to {args.output}")


if __name__ == "__main__":
    main()

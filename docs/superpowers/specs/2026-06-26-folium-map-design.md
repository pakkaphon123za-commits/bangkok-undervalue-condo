# Folium Map Design Spec — `src/report.py`

Date: 2026-06-26
Status: Approved

## Goal

Build `src/report.py` (Phase 8) that generates `docs/index.html` — an interactive folium map of Bangkok condo listings with transit network overlay, deployable to GitHub Pages.

## Inputs

- `data/interim/listings_enriched.parquet` — cleaned + enriched listings with:
  - `listing_id, name, price_thb, first_price_thb, area_sqm_num, price_per_sqm, bedrooms, bathrooms, detail_url, address, latitude, longitude, thumbnail, year_built, listed_dt, updated_dt, nearest_station, nearest_station_km, nearest_station_line, is_ghost` (is_ghost added by Phase 6/7; absent until then)
- `data/processed/stations.geojson` — 201 stations with `name, name_th, ref, lines, operational, coordinates`

## Architecture

```
src/report.py
  ├── load_listings()          reads enriched parquet
  ├── load_stations()          reads stations.geojson
  ├── build_line_polylines()   11 colored polylines, stations sorted by ref
  ├── build_listing_markers()  CircleMarkers in one FeatureGroup
  ├── build_ghost_markers()    hollow red rings in separate FeatureGroup
  ├── inject_color_toggle()    <select> dropdown + JS recolor + legend swap
  └── main()                   assembles folium.Map, saves docs/index.html

docs/index.html                single self-contained HTML (folium embeds JS/CSS inline)
```

Single script, single output file. No external dependencies at serve time — GitHub Pages serves the static HTML as-is.

## Map Layout

- **Base:** OpenStreetMap tiles, zoom_start=12, center [13.7563, 100.5018]
- **Layer 1 — Transit Network (always on):**
  - 11 polylines connecting stations per line, colored by official Bangkok transit brand colors
  - Station markers: small white CircleMarkers (radius=4) with colored border matching the line
  - Station popup: name, line(s), operational/planned status, ref code
- **Layer 2 — Listings (always on):**
  - ~10k CircleMarkers (radius=5, fill_opacity=0.7)
  - Colored by active mode (see Color Scales below)
  - Popup: thumbnail image, building name, price, area, price/sqm, beds/baths, nearest station + distance, line, listed date, year built, FazWaz link
- **Layer 3 — Ghost Listings (toggle, off by default):**
  - Hollow red ring markers (fill=False, color=#c0392b, weight=2, radius=6)
  - Popup: same as listing + red "GHOST" badge + days-on-market count
  - Only shown if `is_ghost` column exists in input parquet; otherwise layer and toggle hidden
- **LayerControl (top-right):** toggles Transit Network and Ghost Listings on/off
- **Color toggle dropdown (top-right, below LayerControl):** `<select>` with JS recolor

## Color Toggle (Approach B — single layer + JS recolor)

One FeatureGroup with all listing markers. Each marker stores `data-price-bin`, `data-dist-bin`, `data-line` as HTML data attributes. A `<select>` dropdown injects JS that loops all markers and calls `setStyle({fillColor, color})` when the user changes selection.

Why not 3 FeatureGroups + LayerControl: 30k DOM elements vs 10k, and LayerControl uses checkboxes (could enable multiple modes simultaneously — nonsensical for color encoding).

### Dropdown options
1. **Price per sqm** (default)
2. **Distance to station**
3. **By transit line**

### Legend
Bottom-right, updates via JS when dropdown changes. Swaps between:
- Quantile legend (4 bins) for price/dist modes
- Categorical legend (11 lines) for line mode

## Color Scales

### Price per sqm — quantile bins, green → cyan → blue → gold
| Bin | Range | Hex | Label |
|-----|-------|-----|-------|
| 1 | < p25 | `#2ecc71` | Budget |
| 2 | p25–p50 | `#1abc9c` | Below median |
| 3 | p50–p75 | `#3498db` | Above median |
| 4 | > p75 | `#f1c40f` | Premium |

Quantiles computed from the enriched parquet at build time — adapts to market shifts between scrape runs.

### Distance to station — quantile bins, same palette (green=close, gold=far)
| Bin | Range (km) | Hex | Label |
|-----|-----------|-----|-------|
| 1 | < p25 | `#2ecc71` | Walk |
| 2 | p25–p50 | `#1abc9c` | Short ride |
| 3 | p50–p75 | `#3498db` | Transit-adjacent |
| 4 | > p75 | `#f1c40f` | Far |

### By transit line — official Bangkok transit brand colors
| Line | Hex |
|------|-----|
| BTS Sukhumvit | `#77BB44` |
| BTS Silom | `#007A33` |
| BTS Gold | `#F2A900` |
| MRT Blue | `#1B4F9C` |
| MRT Purple | `#9B26B6` |
| MRT Yellow | `#FFC20E` |
| MRT Pink | `#EC008C` |
| MRT Orange | `#FF6600` |
| Airport Rail Link | `#A0282C` |
| SRT Dark Red | `#8B1A1A` |
| SRT Light Red | `#D32F2F` |

Multi-line stations (e.g. Siam/CEN) use the first line's color. Listings near a multi-line station show the combined label in popup.

## Station Polyline Ordering

Stations sorted by ref code to produce ordered polylines:

| Line | Ref pattern | Sort logic |
|------|------------|------------|
| BTS Sukhumvit | N1-N24, E1-E23, CEN | Split at CEN into 2 branches: N (northbound) and E (eastbound) |
| BTS Silom | S1-S12, W1, CEN | Split at CEN into 2 branches: S (southbound) and W (west) |
| MRT Blue | BL01-BL38 | Single loop, sort by number |
| MRT Purple | PP01-PP15 | Sort by number |
| MRT Yellow | YL01-YL23 | Sort by number |
| MRT Pink | PK01-PK30 | Sort by number |
| Airport Rail Link | A1-A8 | Sort by number |
| BTS Gold | G1-G3 | Sort by number |
| SRT Dark/Light Red | No ref | Sort by latitude (linear north-south) |

Sukhumvit/Silom branches share the line's color but are drawn as separate polylines to avoid zigzag.

## Popup Content

### Listing popup
```
┌─────────────────────────────────┐
│ [thumbnail image 268x200]       │
│ Chewathai Residence Asoke       │
│ ─────────────────────────────── │
│ Price:        ฿5,000,000        │
│ Area:         35.6 sqm          │
│ Price/sqm:    ฿140,607          │
│ Beds/Baths:   1 / 1             │
│ Year built:   Dec 2016          │
│ Nearest:      Makkasan (0.15km) │
│ Line:         Airport Rail Link │
│ Listed:       2 months ago      │
│ [GHOST · 240 days on market]    │  ← only if is_ghost
│ [View on FazWaz →]              │  ← link to detail_url
└─────────────────────────────────┘
```

### Station popup
```
┌──────────────────────┐
│ Makkasan             │
│ Airport Rail Link    │
│ Ref: A6              │
│ Status: Operational  │
└──────────────────────┘
```

Minimal inline CSS — dark text on white, small font (12px), image full-width, link blue.

## Ghost Listing Display

- **Layer:** Separate FeatureGroup, toggle off by default
- **Marker style:** Hollow red ring (fill=False, color=#c0392b, weight=2, radius=6)
- **Detection (computed by Phase 7 `src/undervalued.py`, not report.py):**
  - `listed_dt` older than 6 months from scrape date
  - AND no price change (`first_price_thb` is NaN OR `first_price_thb == price_thb`)
- **Graceful degradation:** If `is_ghost` column missing from input parquet, ghost layer is empty and toggle hidden — map still works.

## Performance

- 10k CircleMarkers in one FeatureGroup — Leaflet handles this well
- No MarkerCluster (preserves dot-map aesthetic at all zoom levels)
- JS recolor loop: ~10k `setStyle()` calls, completes in <100ms
- Output HTML size: ~5-8MB (10k markers × ~500 bytes each + folium JS/CSS ~1MB)
- GitHub Pages serves static HTML fine at this size

## CLI

```bash
python3 src/report.py
python3 src/report.py --input data/interim/listings_enriched.parquet
python3 src/report.py --output docs/index.html
```

## Output

`docs/index.html` — single self-contained file. No external CSS/JS dependencies (folium embeds everything inline). Ready for GitHub Pages.

## TH/EN Language Toggle

Bilingual UI: English (default) and Thai. Toggle via a button in the top-right toolbar next to the color dropdown.

### What translates
- **UI labels:** dropdown options, legend titles/bin labels, popup field labels (Price, Area, Price/sqm, etc.)
- **Station names:** Thai names from OSM `name:th` tag (all 201 stations have them)
- **Line names:** Thai translations for all 11 lines (hardcoded mapping)

### What stays English
- **Listing building names:** FazWaz doesn't provide Thai names
- **Listing addresses:** FazWaz provides English-only addresses
- **FazWaz link text:** stays "View on FazWaz"

### Implementation
Same JS injection pattern as the color toggle. All translatable text stored as `<span data-en="Price" data-th="ราคา">Price</span>`. When the user clicks TH/EN, JS loops all elements with `data-en`/`data-th` and swaps `textContent`. Station popups store both names; JS shows the active one.

Thai line name mapping (hardcoded in report.py):
```python
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
```

### UI labels translation table
```python
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
```

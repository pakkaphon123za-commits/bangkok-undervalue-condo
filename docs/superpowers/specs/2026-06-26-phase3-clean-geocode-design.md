# Phase 3: Clean & Geocode — Design Spec

**Date:** 2026-06-26  
**Status:** Approved

## Purpose
Produce a clean, typed DataFrame from scraped FazWaz listings, filling in missing latitude/longitude via Nominatim geocoding so listings can be plotted on a folium map. No ghost-listing flagging yet — just typed columns ready for downstream analysis.

## Input
`data/raw/fazwaz/listings.parquet` — output from `src/scrape.py`, columns:
- `listing_id`, `name`, `price`, `first_price`, `detail_url`, `address`
- `area_sqm`, `bedrooms`, `bathrooms`, `property_type`
- `transit_stations_json` (JSON string)
- `listed_date`, `updated_date`
- `latitude`, `longitude` (may be null)

## Output
`data/interim/listings_clean.parquet` — same rows, plus:
- `price_thb` (float): parsed numeric price
- `first_price_thb` (float): parsed numeric first price
- `area_sqm_num` (float): parsed numeric area
- `price_per_sqm` (float): computed `price_thb / area_sqm_num`
- `listed_dt` (datetime): parsed listed date
- `updated_dt` (datetime): parsed updated date
- `latitude`, `longitude` filled from geocoding where previously null
- Geocode cache: `data/interim/geocode_cache.json` (address → [lat, lng])

## Processing Steps

### 1. Type Parsing
- **price / first_price:** Strip `฿` prefix, remove commas, convert to `float`. Null/empty → NaN.
- **area_sqm:** Parse numeric part (strip units like "m²", "sqm"), convert to `float`.
- **listed_date / updated_date:** Parse flexible date formats (e.g. "26 Jun 2024", "2024-06-26") via `pd.to_datetime`.
- **price_per_sqm:** Computed from parsed columns; NaN if either is NaN.

### 2. Geocoding (Nominatim)
- **Trigger:** Only for rows where `latitude` is null.
- **Query:** Use `address` field as search query.
- **Rate limit:** 1 request/second (Nominatim policy).
- **User-Agent:** `BangkokTransitPropertyAnalysis/1.0 (portfolio project)`.
- **Cache:** Before calling Nominatim, check `data/interim/geocode_cache.json`. On successful geocode, store in cache. Cache is a flat dict: `{"address string": [lat, lng]}`.
- **Retry:** If Nominatim returns no results, leave coords as NaN (don't retry same address).

### 3. Save
- Write `listings_clean.parquet` to `data/interim/`.
- Overwrite geocode cache file after each batch (resumable if interrupted).

## File
`src/clean.py` — single script, no separate module needed. Entrypoint with argparse for flags.

## Dependencies
- `httpx` (already used by scraper)
- `pandas` (already used)
- No new packages required.

## Verification
- Run on a small sample (scrape with `--max-pages 1 --max-detail 5` first to produce a test parquet).
- Spot-check: price columns are float, dates are datetime, missing coords are filled or NaN.
- Print summary: rows with/without coords before vs after, geocode hit rate.

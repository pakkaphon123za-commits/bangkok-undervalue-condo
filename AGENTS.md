# AGENTS.md — local instructions for the opencode agent (NOT committed to GitHub).

## Project: Bangkok Transit-Property Analysis

Pipeline: scrape Fazwaz condo listings → clean + geocode → enrich with nearest station → model price-decay → detect undervalued zones → LLM narratives → static folium map on GitHub Pages.

### Commands
- Smoke-test imports: `python3 -c "import src.clean; import src.enrich; import src.report; import src.model; import src.undervalued; import src.llm_narrate"`
- Tests: `python3 -m pytest tests/` — 26 tests in `tests/test_report.py`, 13 tests in `tests/test_llm_narrate.py`, all passing
- Single test: `python3 -m pytest tests/test_report.py::test_load_listings_basic -v`
- No lint/typecheck config exists. No CI exists.

### Pipeline (must run in order)
```bash
python3 src/stations.py                              # Phase 1: OSM stations → stations.geojson
python3 src/scrape.py --max-pages 5 --max-detail 10  # Phase 2: test scrape (full run: omit flags)
python3 src/clean.py                                 # Phase 3: parse types + geocode null coords
python3 src/enrich.py                                # Phase 4: nearest station per listing
python3 src/model.py                                 # Phase 5: price-decay model + residuals
python3 src/undervalued.py                           # Phase 6: detect undervalued zones
python3 src/llm_narrate.py   # Phase 7: LLM narrative
python3 src/report.py        # Phase 8: folium map → docs/index.html
```
- Each step's output feeds the next. Never run enrich before clean; never run report before enrich.
- `--input`/`--output` flags override defaults on all scripts.
- `clean.py` supports `--no-geocode` (skip Nominatim) and `--geocode-limit N` (cap rows).

### Import convention (critical)
- `src/scrape.py` imports via `from sources.fazwaz import ...` — works because Python adds the script's directory to `sys.path`. Do NOT change to relative imports.
- All other `src/*.py` modules use `from __future__ import annotations` and absolute file paths via `Path(__file__).resolve().parent.parent`.

### Environment quirks
- Python 3.14.4. **No venv in use.** pip installed with `--break-system-packages`.
- Install new packages: `python3 -m pip install --user --break-system-packages <pkg>`
- `python3` (not `python`) — the system symlink. README says `python` but that doesn't exist here.
- `numpy` is used in `enrich.py` and `report.py` but NOT listed in `requirements.txt` — comes as a pandas/geopandas transitive dep. Add to requirements.txt if it breaks.

### Scraping (Phase 2)
- **Entrypoint:** `python3 src/scrape.py` (runs from repo root)
- Flags: `--rate-limit` (default 1.0s), `--concurrency` (default 4), `--max-pages`, `--max-detail`
- Full run: 333 search pages, ~10k listings. With concurrency 4 + rate-limit 0.25, search ~5 min, detail pages much faster than serial.
- HTML cache: `data/raw/fazwaz/search_*.html` and `detail_*.html` — auto-managed, never committed.
- Output: `data/raw/fazwaz/listings.parquet`
- Cache is keyed by listing ID. Re-runs skip already-fetched pages/detail.
- Rate limiter uses `asyncio.Lock` so concurrent tasks don't race on `_last_request`; `asyncio.Semaphore` caps in-flight requests.
- Detail fetch dedup: cards deduped by `listing_id` before fetching detail (FazWaz pagination returns dups).
- Non-HTTP detail URLs (e.g. `forceLogin`) are skipped — ~288 listings have these (no coords, need geocoding).

### Geocoding (Phase 3, in clean.py)
- Uses Nominatim (OSM) — free, 1.2 req/s rate limit, `countrycodes=th` filter.
- **Coordinate priority chain:** detail page coords (building-level, ~100%) → geocode by building name (OSM, ~28% hit) → geocode by sub-district address (centroid, last resort).
- Geocode cache: `data/interim/geocode_cache.json` — stores `[lat,lng]` for hits, `None` for misses (prevents re-querying failed addresses).
- `_resolve_from_cache` checks ALL queries before returning a miss — address fallback works even when name miss is cached.
- Many Fazwaz listings share the same sub-district address → address-only geocoding gives centroid, not building. Detail page coords are the real fix.
- Known miss: "Chomphon" (Fazwaz spelling) vs "Chom Phon" (OSM spelling) — romanization mismatch.

### Data validation in clean.py (hard-earned fixes)
- **Dedup:** `drop_duplicates(subset=["listing_id"])` at load — removes ~77 dups from FazWaz pagination.
- **Coord bounds:** lat outside 13.0–14.0 or lng outside 100.0–101.0 → nulled (then geocoded). Catches FazWaz data errors (e.g. 15 Celes Asoke listings at lat=3.74, ~1000km off).
- **Price = 0:** `price_thb` nulled if parsed value is 0 (FazWaz placeholder). `price_per_sqm` becomes null too.
- **Bathrooms:** `fillna(0)` — prevents parquet crash on string/null values.
- **Date regex:** matches `year|month|week|day|hour|minute` — "Updated 21 minutes ago" now parses.

### Enrichment (Phase 4, in enrich.py)
- Filters to 189 operational stations by default (12 planned excluded). Use `--include-planned` to include all 201.
- Vectorized haversine via numpy — handles 10k listings in <1s.
- Output columns: `nearest_station`, `nearest_station_km`, `nearest_station_line`.

### Report (Phase 8, in report.py)
- Generates `docs/index.html` — self-contained folium map with transit overlay, color-toggle (price/distance/line), TH/EN language toggle, ghost listing layer.
- Output is committed to git (GitHub Pages target). Re-run after data changes.
- Tests in `tests/test_report.py` cover data loading, station sorting by ref/lat, popup HTML, transit layer, toggles, and end-to-end HTML generation.

### Data conventions
- `src/stations.py` → `data/processed/stations.geojson` (201 stations with `name` + `name_th`, committed — ODbL open data)
- `src/scrape.py` → `data/raw/fazwaz/listings.parquet` (not committed)
- `src/clean.py` → `data/interim/listings_clean.parquet` + `data/interim/geocode_cache.json`
- `src/enrich.py` → `data/interim/listings_enriched.parquet`
- `src/model.py` → `data/interim/listings_modeled.parquet` + `data/processed/decay_curves.json`
- `src/undervalued.py` → `data/interim/listings_modeled.parquet` (updated in place) + `data/processed/undervalued_summary.json`
- `src/report.py` → `docs/index.html` (committed — the website)
- `data/raw/`, `data/interim/`, `data/processed/*` are all gitignored EXCEPT `stations.geojson`, `decay_curves.json`, and `undervalued_summary.json`
- Only commit a tiny sample (~20 rows) for testing, never full scraped data.

### Architecture (phases)
- Phase 1 done: `src/stations.py` — Overpass API query, 3-tier classification, dedup, GeoJSON output
- Phase 2 done: `src/sources/fazwaz.py` + `src/scrape.py` — Fazwaz scraper (concurrent, rate-limited, cached)
- Phase 3 done: `src/clean.py` — type parsing, relative dates, coord validation, Nominatim geocoding with retry + cache
- Phase 4 done: `src/enrich.py` — nearest station matching (vectorized, operational-only by default)
- Phase 5 done: `src/model.py` — mixed-effects price-decay model + residuals
- Phase 6 done: `src/undervalued.py` — MAD-based z-score undervaluation detection
- Phase 7 not yet built: `src/llm_narrate.py`
- Phase 8 done: `src/report.py` — folium map → `docs/index.html`

### OSM station knowledge (avoid repeating past bugs)
- Yellow/Pink monorail lines use `railway=stop` not `railway=station` in OSM
- CEN/Siam special-cased to both Sukhumvit + Silom lines (not in any OSM route relation)
- Only proper line-code refs (N/E/S/W/G/BL/PP/YL/PK/etc) used for ref-based dedup; bare numbers excluded (prevents false collisions)
- Thai-only route names need Thai keyword matching (สุขุมวิท, สีลม, สีน้ำเงิน, etc.)
- BTS Gold Line: only 3 stations in OSM (G1-G3); other 4 not yet mapped — OSM data gap, not a code bug

### Fazwaz scraping specifics (avoid repeating HTML analysis)
- Search cards: `.result-search__item` (30/page), data in `onmouseenter` attribute JSON
- JSON fields: name, price, firstPrice, detailUrl, formatted_address, area, bedrooms, bathrooms, propertyType, nearPlaceGroup (transit stations)
- Dates: `.all-status__new ::text` (listed date), `.all-status__lastUpdated ::text` (update date) — stored as relative strings ("listed 2 months ago")
- Detail page lat/lng: `viewpoint=<lat>,<lng>` in Google Maps Street View URL
- Pagination: `?page=N`, last page detected from pagination bar links (currently page 333)
- Force `Accept-Language: en-US` and follow redirects (FazWaz redirects by locale)
- `bedrooms` field: "Studio" coerced to 0 (int)
- `first_price` is NaN for new listings (no price history) — this is correct, not a bug
- ~28.5% of listings have empty `listed_date` strings — FazWaz data gap, not a parse bug

### Git
- AGENTS.md is gitignored — local-only, never committed
- Never commit: `data/raw`, `data/interim`, `data/processed`, `config.yaml`, `.env`, agent files, IDE files
- No force-push, no empty commits, no `git commit --amend`
- Write concise commit messages. Inspect `git status` + `git diff` + `git log --oneline -10` before committing.

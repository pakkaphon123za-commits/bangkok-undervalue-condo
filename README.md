# Bangkok Transit-Property Analysis

Analyzing how Bangkok condo prices decay with walking distance from BTS/MRT stations, and identifying undervalued zones near new and upcoming line extensions.

## What this project does

- Scrapes condo listings from Fazwaz (with latitude/longitude) across Bangkok
- Fetches every BTS/MRT station with coordinates via OpenStreetMap (Overpass API)
- Computes walking distance from each listing to its nearest station
- Fits a **price-decay curve per transit line** (how price-per-sqm drops with distance)
- Detects **undervalued zones** — areas where prices are lower than the model predicts, especially near upcoming line extensions (Orange Line, Purple Line South)
- Publishes results as an interactive website on GitHub Pages

## Transit lines covered

Operational: BTS Sukhumvit, BTS Silom, BTS Gold, MRT Blue, MRT Purple, MRT Yellow, MRT Pink, SRT Dark Red, SRT Light Red, Airport Rail Link.

Upcoming (flagged for undervalued-zone detection): MRT Orange, MRT Purple South, SRT Dark Red south extension.

## Tech stack

- Python 3.14 — scraping (httpx + parsel), modeling (statsmodels, scikit-learn)
- Overpass API — station coordinates
- OSRM — walking-distance routing
- Plotly + Folium — interactive charts and map
- GitHub Pages — static site hosting (no server)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp config.example.yaml config.yaml   # edit if needed
cp .env.example .env                  # add OPENCODE_GO_API_KEY (optional)

# 1. Fetch stations
python src/stations.py

# 2. Scrape Fazwaz listings (test first, then remove flags for full run)
python src/scrape.py --max-pages 5 --max-detail 10

# 3. Clean + enrich + model + detect zones
python src/clean.py
python src/enrich.py
python src/model.py
python src/undervalued.py

# 4. (optional) LLM narratives
python src/llm_narrate.py

# 5. Build the website
python src/report.py
# open index.html or visit the GitHub Pages URL
```

## Data & privacy

- Scraped HTML is cached locally under `data/raw/` (gitignored, never committed) to avoid re-fetching.
- Only a tiny anonymized sample (~20 rows) is committed for testing.
- No API keys or secrets are committed.
- The website shows a snapshot baked in at build time; re-run the pipeline to refresh.

## License

MIT — see [LICENSE](LICENSE).

# Bangkok Transit-Property Analysis

[![View map](https://img.shields.io/badge/View_interactive_map-2ea44f?logo=githubpages)](https://pakkaphon123za-commits.github.io/bangkok-undervalue-condo/)

A small project that scrapes Bangkok condo listings, matches them to the nearest BTS/MRT/SRT station, and estimates which ones are underpriced relative to similar units nearby.

**Live site:** https://pakkaphon123za-commits.github.io/bangkok-undervalue-condo/

## What it does

- Scrapes condo listings from Fazwaz with coordinates, price, area, and specs.
- Pulls Bangkok transit station locations from OpenStreetMap via Overpass.
- Computes distance from each listing to its nearest station.
- Fits a price-decay model per transit line to see how price-per-sqm drops with distance.
- Flags listings that trade below the model prediction.
- Builds a self-contained interactive map and publishes it to GitHub Pages.

## Key findings

Out of 9,599 listings near transit, **6.91%** are priced below the model prediction. The BTS Sukhumvit Line has the sharpest price decay (−26.2% per km) and the largest share of undervalued units. Standout station zones include **On Nut**, **Bearing**, **Bang Na**, and **Phetkasem 48**.

Three upcoming extensions — MRT Orange, MRT Purple South, and SRT Dark Red South — are worth watching because they tend to reprice nearby submarkets once they open.

Read the full market brief in [docs/narrative.md](docs/narrative.md).

## Transit lines

Operational: BTS Sukhumvit, BTS Silom, BTS Gold, MRT Blue, MRT Purple, MRT Yellow, MRT Pink, SRT Dark Red, SRT Light Red, Airport Rail Link.

Tracked for future impact: MRT Orange, MRT Purple South, SRT Dark Red south extension.

## Stack

- Python 3.14 — `httpx` + `parsel` for scraping, `statsmodels`/`scikit-learn` for modeling.
- Overpass API — station coordinates.
- OSRM — walking-distance routing.
- Plotly + Folium — charts and map.
- GitHub Pages — static hosting.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp config.example.yaml config.yaml
cp .env.example .env          # optional: only needed for LLM narrative step

python src/stations.py
python src/scrape.py --max-pages 5 --max-detail 10
python src/clean.py
python src/enrich.py
python src/model.py
python src/undervalued.py
python src/report.py
# open docs/index.html or visit the GitHub Pages URL
```

To regenerate the LLM narrative:

```bash
python src/llm_narrate.py
```

## Data & privacy

- Scraped HTML is cached under `data/raw/` and gitignored.
- A 200-row sample is committed in `data/sample/` so the map can render without scraping.
- No API keys or secrets are committed.
- The website is a static snapshot; rerun the pipeline to update it.

## Caveats

- Distances are straight-line, not walking time.
- "Undervalued" means below the model prediction, not a guaranteed investment.
- Some lines (Pink, Dark Red, Light Red) have small sample sizes, so treat their numbers cautiously.

## License

MIT — see [LICENSE](LICENSE).

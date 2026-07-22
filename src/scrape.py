"""Scrape Fazwaz condo listings and write raw data to data/raw/fazwaz/listings.parquet.

Usage:
    python src/scrape.py                    # full run (all pages, all detail pages)
    python src/scrape.py --max-pages 5       # test: 5 search pages
    python src/scrape.py --max-detail 10    # test: only fetch 10 detail pages
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

from sources.fazwaz import FazwazClient, ListingRecord

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fazwaz"
OUTPUT_PATH = RAW_DIR / "listings.parquet"


def _coerce_bedrooms(val) -> int:
    if val is None:
        return 0
    if isinstance(val, str) and val.lower() == "studio":
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _coerce_bathrooms(val) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def listings_to_dataframe(listings: list[ListingRecord]) -> pd.DataFrame:
    rows = []
    for rec in listings:
        transit_json = json.dumps(rec.transit_stations, ensure_ascii=False) if rec.transit_stations else None
        rows.append(
            {
                "listing_id": rec.listing_id,
                "name": rec.name,
                "price": rec.price,
                "first_price": rec.first_price,
                "detail_url": rec.detail_url,
                "address": rec.address,
                "area_sqm": rec.area_sqm,
                "bedrooms": _coerce_bedrooms(rec.bedrooms),
                "bathrooms": _coerce_bathrooms(rec.bathrooms),
                "property_type": rec.property_type,
                "transit_stations_json": transit_json,
                "listed_date": rec.listed_date,
                "updated_date": rec.updated_date,
                "latitude": rec.latitude,
                "longitude": rec.longitude,
                "thumbnail": rec.thumbnail,
                "year_built": rec.year_built,
            }
        )
    return pd.DataFrame(rows)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Fazwaz condo listings")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-detail", type=int, default=None)
    parser.add_argument("--rate-limit", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    client = FazwazClient(
        rate_limit=args.rate_limit,
        cache_dir=RAW_DIR,
        timeout=args.timeout,
    )

    try:
        listings, stats = await client.scrape(
            property_type="condo-for-sale",
            region="thailand/bangkok",
            max_pages=args.max_pages,
            max_detail_pages=args.max_detail,
            max_concurrency=args.concurrency,
        )
    finally:
        await client.close()

    df = listings_to_dataframe(listings)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nWritten {len(listings)} listings to {OUTPUT_PATH}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nScrape stats:")
    print(f"  search pages:   {stats.search_pages} ({stats.search_errors} errors)")
    print(f"  detail pages:   {stats.detail_pages} ({stats.detail_errors} errors)")
    print(f"  total cards:    {stats.total_cards}")
    print(f"  elapsed:        {stats.elapsed_s:.1f}s")
    print(f"  with coords:    {df['latitude'].notna().sum()}")
    print(f"  without coords: {df['latitude'].isna().sum()}")


if __name__ == "__main__":
    asyncio.run(main())

"""Clean scraped FazWaz listings: parse types, geocode missing coords.

Usage:
    python src/clean.py                        # full clean + geocode
    python src/clean.py --input data/raw/fazwaz/listings.parquet
    python src/clean.py --geocode-limit 10     # only geocode 10 rows
    python src/clean.py --no-geocode           # skip geocoding entirely
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fazwaz"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
GEOCODE_CACHE_PATH = INTERIM_DIR / "geocode_cache.json"
OUTPUT_PATH = INTERIM_DIR / "listings_clean.parquet"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "BangkokTransitPropertyAnalysis/1.0 (portfolio project)"
NOMINATIM_MAX_RETRIES = 3

BANGKOK_LAT_RANGE = (13.0, 14.0)
BANGKOK_LNG_RANGE = (100.0, 101.0)

RELATIVE_DATE_RE = re.compile(
    r"(?:listed|updated)\s+(\d+)\s+(year|month|week|day|hour|minute)s?\s+ago",
    re.IGNORECASE,
)


def _parse_price(raw: str | None) -> float | None:
    if pd.isna(raw) or not raw:
        return None
    try:
        cleaned = raw.replace("฿", "").replace(",", "").strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def _parse_area(raw: str | None) -> float | None:
    if pd.isna(raw) or not raw:
        return None
    try:
        cleaned = re.sub(r"[^\d.]", "", str(raw))
        return float(cleaned) if cleaned else None
    except (ValueError, AttributeError):
        return None


def _parse_relative_date(raw: str | None, ref_date: datetime) -> datetime | None:
    if pd.isna(raw) or not raw:
        return None
    m = RELATIVE_DATE_RE.search(str(raw))
    if not m:
        return None
    num = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "year":
        delta = timedelta(days=num * 365)
    elif unit == "month":
        delta = timedelta(days=num * 30)
    elif unit == "week":
        delta = timedelta(weeks=num)
    elif unit == "day":
        delta = timedelta(days=num)
    elif unit == "hour":
        delta = timedelta(hours=num)
    elif unit == "minute":
        delta = timedelta(minutes=num)
    else:
        return None
    return ref_date - delta


def _parse_date(raw: str | None, ref_date: datetime) -> datetime | None:
    if pd.isna(raw) or not raw:
        return None
    relative = _parse_relative_date(raw, ref_date)
    if relative is not None:
        return relative
    try:
        return pd.to_datetime(raw)
    except Exception:
        return None


def load_and_clean(input_path: Path, ref_date: datetime | None = None) -> pd.DataFrame:
    if ref_date is None:
        ref_date = datetime.now()

    df = pd.read_parquet(input_path)

    before = len(df)
    df = df.drop_duplicates(subset=["listing_id"], keep="first")
    if len(df) < before:
        print(f"  removed {before - len(df)} duplicate listings")

    bad_coord_mask = df["latitude"].notna() & (
        (df["latitude"] < BANGKOK_LAT_RANGE[0])
        | (df["latitude"] > BANGKOK_LAT_RANGE[1])
        | (df["longitude"] < BANGKOK_LNG_RANGE[0])
        | (df["longitude"] > BANGKOK_LNG_RANGE[1])
    )
    bad_count = bad_coord_mask.sum()
    if bad_count:
        print(f"  nulling {bad_count} listings with coordinates outside Bangkok bounds")
        df.loc[bad_coord_mask, ["latitude", "longitude"]] = None

    df["price_thb"] = df["price"].apply(_parse_price).where(lambda x: x > 0)
    df["first_price_thb"] = df["first_price"].apply(_parse_price).where(lambda x: x > 0)
    df["area_sqm_num"] = df["area_sqm"].apply(_parse_area)
    df["bathrooms"] = df["bathrooms"].fillna(0)
    df["price_per_sqm"] = df.apply(
        lambda r: r["price_thb"] / r["area_sqm_num"]
        if pd.notna(r["price_thb"]) and pd.notna(r["area_sqm_num"]) and r["area_sqm_num"] != 0
        else None,
        axis=1,
    )
    df["listed_dt"] = df["listed_date"].apply(lambda x: _parse_date(x, ref_date))
    df["updated_dt"] = df["updated_date"].apply(lambda x: _parse_date(x, ref_date))

    return df


def load_cache() -> dict[str, list[float] | None]:
    if GEOCODE_CACHE_PATH.exists():
        with open(GEOCODE_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict[str, list[float] | None]) -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    with open(GEOCODE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


async def _nominatim_geocode(
    client: httpx.AsyncClient,
    query: str,
    rate_limit: float = 1.2,
) -> tuple[float | None, float | None]:
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
        "countrycodes": "th",
    }
    for attempt in range(NOMINATIM_MAX_RETRIES):
        try:
            resp = await client.get(NOMINATIM_URL, params=params)
            if resp.status_code in (429, 503):
                wait = 2 ** attempt * rate_limit
                print(f"  rate-limited ({resp.status_code}), retrying in {wait:.0f}s...")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            results = resp.json()
            if results:
                return float(results[0]["lat"]), float(results[0]["lon"])
            return None, None
        except Exception as exc:
            if attempt < NOMINATIM_MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            print(f"  geocode error [{query[:40]}]: {exc}")
            return None, None
    return None, None


def _resolve_from_cache(
    cache: dict[str, list[float] | None],
    name_query: str | None,
    addr_query: str | None,
) -> tuple[float | None, float | None, str | None]:
    """Check cache for name query first, then address query.
    Returns (lat, lng, cache_key_used).
    - If a hit is found: returns (lat, lng, hit_key)
    - If all queries are cached as misses: returns (None, None, first_miss_key) so caller can skip
    - If any query is not yet cached: returns (None, None, None) so caller proceeds to geocode"""
    queries = [q for q in (name_query, addr_query) if q]
    all_cached = True
    first_miss: str | None = None
    for query in queries:
        if query in cache:
            if cache[query] is not None:
                return cache[query][0], cache[query][1], query
            if first_miss is None:
                first_miss = query
        else:
            all_cached = False
    if all_cached and queries:
        return None, None, first_miss
    return None, None, None


async def _geocode_with_fallback(
    client: httpx.AsyncClient,
    name_query: str | None,
    addr_query: str | None,
    rate_limit: float,
) -> tuple[float | None, float | None, str | None]:
    """Try geocoding by building name first, fall back to address.
    Returns (lat, lng, query_used). query_used is None if both queries missed."""
    for query in (name_query, addr_query):
        if not query:
            continue
        lat, lng = await _nominatim_geocode(client, query, rate_limit)
        if lat is not None:
            return lat, lng, query
        await asyncio.sleep(rate_limit)
    return None, None, None


async def geocode_missing(
    df: pd.DataFrame,
    limit: int | None = None,
    rate_limit: float = 1.0,
) -> pd.DataFrame:
    cache = load_cache()
    missing_mask = df["latitude"].isna()
    missing_idxs = df.index[missing_mask]

    if limit is not None:
        missing_idxs = missing_idxs[:limit]

    print(f"Geocoding {len(missing_idxs)} listings with missing coordinates...")
    hit_count = 0
    miss_count = 0

    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(30.0),
    ) as client:
        for i, idx in enumerate(missing_idxs):
            address = df.at[idx, "address"]
            name = df.at[idx, "name"]
            if (pd.isna(address) or not address) and (pd.isna(name) or not name):
                miss_count += 1
                continue

            name_query = str(name).strip() if pd.notna(name) and name else None
            addr_query = str(address).strip() if pd.notna(address) and address else None

            lat, lng, cache_key_used = _resolve_from_cache(cache, name_query, addr_query)
            if lat is not None:
                df.at[idx, "latitude"] = lat
                df.at[idx, "longitude"] = lng
                hit_count += 1
                continue
            if cache_key_used is not None and cache_key_used in cache and cache[cache_key_used] is None:
                miss_count += 1
                continue

            await asyncio.sleep(rate_limit)

            lat, lng, query_used = await _geocode_with_fallback(
                client, name_query, addr_query, rate_limit
            )

            if lat is not None:
                df.at[idx, "latitude"] = lat
                df.at[idx, "longitude"] = lng
                cache[query_used] = [lat, lng]
                hit_count += 1
            else:
                if query_used:
                    cache[query_used] = None
                miss_count += 1

            if (i + 1) % 10 == 0:
                save_cache(cache)
                print(f"  {i + 1}/{len(missing_idxs)} | hits: {hit_count} | misses: {miss_count}")

    save_cache(cache)
    print(f"Geocode done: {hit_count} hits, {miss_count} misses")
    return df


async def main() -> None:
    parser = argparse.ArgumentParser(description="Clean scraped listings")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DIR / "listings.parquet",
        help="Path to raw listings parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path for cleaned output parquet",
    )
    parser.add_argument(
        "--no-geocode",
        action="store_true",
        help="Skip geocoding step",
    )
    parser.add_argument(
        "--geocode-limit",
        type=int,
        default=None,
        help="Max geocode attempts (default: all missing)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.2,
        help="Seconds between Nominatim requests (default: 1.2)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        print("Run the scraper first: python src/scrape.py --max-pages 1 --max-detail 5")
        return

    ref_date = datetime.now()
    print(f"Loading {args.input}...")
    print(f"Reference date: {ref_date.strftime('%Y-%m-%d')}")

    df = load_and_clean(args.input, ref_date)
    print(f"Loaded {len(df)} listings")

    before_coords = df["latitude"].notna().sum()
    print(f"Coordinates before geocode: {before_coords}/{len(df)}")

    if not args.no_geocode:
        df = await geocode_missing(
            df,
            limit=args.geocode_limit,
            rate_limit=args.rate_limit,
        )

    after_coords = df["latitude"].notna().sum()
    print(f"Coordinates after geocode:  {after_coords}/{len(df)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Written {len(df)} rows to {args.output}")
    print(f"Columns: {list(df.columns)}")

    price_parsed = df["price_thb"].notna().sum()
    area_parsed = df["area_sqm_num"].notna().sum()
    date_parsed = df["listed_dt"].notna().sum()
    print(f"Price parsed: {price_parsed}/{len(df)}")
    print(f"Area parsed:  {area_parsed}/{len(df)}")
    print(f"Dates parsed: {date_parsed}/{len(df)}")


if __name__ == "__main__":
    asyncio.run(main())

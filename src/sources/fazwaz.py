"""Fazwaz.com condo listing scraper.

Extracts search result cards from listing pages and detail-page lat/lng from
Google Street View viewpoint links.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from parsel import Selector

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_HEADERS = {
    "User-Agent": "BangkokTransitPropertyAnalysis/1.0 (portfolio project)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}

# UK → EN version of FazWaz has latitude/longitude in wire:snapshot
# Some detail pages redirect to French or other locales; force EN domain.
BASE_URL = "https://www.fazwaz.com"


@dataclass
class CardRecord:
    listing_id: str
    name: str
    price: str
    first_price: str | None
    detail_url: str
    address: str
    area_sqm: str
    bedrooms: int
    bathrooms: int
    property_type: str
    transit_stations: list[dict[str, str]]
    listed_date: str
    updated_date: str
    thumbnail: str | None = None


@dataclass
class ListingRecord:
    listing_id: str
    name: str
    price: str
    first_price: str | None
    detail_url: str
    address: str
    area_sqm: str
    bedrooms: int
    bathrooms: int
    property_type: str
    transit_stations: list[dict[str, str]]
    listed_date: str
    updated_date: str
    latitude: float | None = None
    longitude: float | None = None
    thumbnail: str | None = None
    year_built: str | None = None


@dataclass
class ScrapeStats:
    search_pages: int = 0
    search_errors: int = 0
    detail_pages: int = 0
    detail_errors: int = 0
    total_cards: int = 0
    elapsed_s: float = 0.0


def _parse_card_json(raw: str) -> dict[str, Any]:
    import html as htmlmod
    decoded = htmlmod.unescape(raw)
    decoded = decoded.replace("\\/", "/").replace('\\"', '"')
    return json.loads(decoded)


def _extract_card_onmouseenter(onmouse: str) -> dict[str, Any] | None:
    m = re.search(
        r"callFuncIfExists\('[^']*',\s*\d+,\s*'(\{.*?\})'\)", onmouse
    )
    if not m:
        return None
    return _parse_card_json(m.group(1))


_VIEWPOINT_RE = re.compile(r"viewpoint=([\d.]+),([\d.]+)")
_YEAR_BUILT_RE = re.compile(r"Construction:</span>\s*<span[^>]*>Completed\s*\(([^)]+)\)")


def _extract_coords(html: str) -> tuple[float, float] | None:
    m = _VIEWPOINT_RE.search(html)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def _extract_year_built(html: str) -> str | None:
    m = _YEAR_BUILT_RE.search(html)
    if m:
        return m.group(1).strip()
    return None


class FazwazClient:
    def __init__(
        self,
        rate_limit: float = 1.0,
        cache_dir: str | Path | None = None,
        timeout: float = 60.0,
    ):
        self._rate_limit = rate_limit
        self._timeout = timeout
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request = 0.0
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=DEFAULT_HEADERS,
                timeout=httpx.Timeout(self._timeout),
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _rate_limit_wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._rate_limit:
            await asyncio.sleep(self._rate_limit - elapsed)

    async def _get(self, url: str, cache_key: str | None = None) -> str:
        if cache_key and self._cache_dir:
            cache_path = self._cache_dir / cache_key
            if cache_path.exists():
                return cache_path.read_text(encoding="utf-8")

        await self._rate_limit_wait()
        client = await self._get_client()
        resp = await client.get(url)
        self._last_request = time.monotonic()
        resp.raise_for_status()
        text = resp.text

        if cache_key and self._cache_dir:
            (self._cache_dir / cache_key).write_text(text, encoding="utf-8")

        return text

    async def fetch_search_page(
        self,
        property_type: str,
        region: str,
        page: int,
    ) -> tuple[list[CardRecord], int]:
        url = f"{BASE_URL}/{property_type}/{region}"
        if page > 1:
            url = f"{url}?page={page}"
        cache_key = f"search_{property_type.replace('/', '_')}_{page}.html"
        html = await self._get(url, cache_key)
        sel = Selector(html)

        cards: list[CardRecord] = []
        for item in sel.css(".result-search__item"):
            listing_id = item.attrib.get("data-id", "")
            onmouse = item.attrib.get("onmouseenter", "")
            data = _extract_card_onmouseenter(onmouse)
            if not data:
                continue

            new_text = (item.css(".all-status__new ::text").get("") or "").strip()
            updated_text = (item.css(".all-status__lastUpdated ::text").get("") or "").strip()

            cards.append(
                CardRecord(
                    listing_id=listing_id,
                    name=data.get("name", ""),
                    price=data.get("price", ""),
                    first_price=data.get("firstPrice"),
                    detail_url=data.get("detailUrl", ""),
                    address=data.get("formatted_address", ""),
                    area_sqm=data.get("area", ""),
                    bedrooms=data.get("bedrooms", 0),
                    bathrooms=data.get("bathrooms", 0),
                    property_type=data.get("propertyType", ""),
                    transit_stations=data.get("nearPlaceGroup", []),
                    listed_date=new_text,
                    updated_date=updated_text,
                    thumbnail=data.get("thumbnail"),
                )
            )

        last_page = self._parse_last_page(sel, page)
        return cards, last_page

    @staticmethod
    def _parse_last_page(sel: Selector, current: int) -> int:
        last = current
        for link in sel.css("a[href]"):
            href = link.attrib.get("href", "")
            m = re.search(r"page=(\d+)", href)
            if m:
                p = int(m.group(1))
                if p > last:
                    last = p
        return last

    async def fetch_detail_page(
        self, url: str, listing_id: str
    ) -> tuple[tuple[float, float] | None, str | None]:
        cache_key = f"detail_{listing_id}.html"
        html = await self._get(url, cache_key)
        return _extract_coords(html), _extract_year_built(html)

    async def scrape(
        self,
        property_type: str = "condo-for-sale",
        region: str = "thailand/bangkok",
        max_pages: int | None = None,
        max_detail_pages: int | None = None,
    ) -> tuple[list[ListingRecord], ScrapeStats]:
        stats = ScrapeStats()
        t0 = time.monotonic()

        all_cards: list[CardRecord] = []
        page = 1
        last_page = 1

        while True:
            if max_pages is not None and page > max_pages:
                break
            if page > last_page:
                break

            try:
                cards, detected_last = await self.fetch_search_page(
                    property_type, region, page
                )
                if detected_last > last_page:
                    last_page = detected_last
                all_cards.extend(cards)
                stats.search_pages += 1
                stats.total_cards += len(cards)
                print(
                    f"  page {page}/{last_page}: {len(cards)} cards "
                    f"(total {stats.total_cards})"
                )
            except Exception as exc:
                stats.search_errors += 1
                print(f"  page {page} error: {exc}")

            if not cards and page > 1:
                break
            page += 1

        listings: list[ListingRecord] = []
        detail_limit = (
            max_detail_pages if max_detail_pages is not None else len(all_cards)
        )
        detail_count = 0

        for card in all_cards:
            if detail_count >= detail_limit:
                break
            lat, lng, year_built = None, None, None
            if card.detail_url:
                try:
                    coords, yb = await self.fetch_detail_page(
                        card.detail_url, card.listing_id
                    )
                    if coords:
                        lat, lng = coords
                        stats.detail_pages += 1
                    if yb:
                        year_built = yb
                except Exception as exc:
                    stats.detail_errors += 1
                    print(f"  detail {card.listing_id} error: {exc}")
            else:
                stats.detail_errors += 1

            detail_count += 1
            listings.append(
                ListingRecord(
                    listing_id=card.listing_id,
                    name=card.name,
                    price=card.price,
                    first_price=card.first_price,
                    detail_url=card.detail_url,
                    address=card.address,
                    area_sqm=card.area_sqm,
                    bedrooms=card.bedrooms,
                    bathrooms=card.bathrooms,
                    property_type=card.property_type,
                    transit_stations=card.transit_stations,
                    listed_date=card.listed_date,
                    updated_date=card.updated_date,
                    latitude=lat,
                    longitude=lng,
                    thumbnail=card.thumbnail,
                    year_built=year_built,
                )
            )

        stats.elapsed_s = time.monotonic() - t0
        return listings, stats

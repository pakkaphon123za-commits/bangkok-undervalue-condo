"""Tests for scrape.py — Phase 2 raw scraper type coercion."""
from __future__ import annotations

import numpy as np
import pytest

from src.scrape import (
    _coerce_bathrooms,
    _coerce_bedrooms,
    listings_to_dataframe,
)
from sources.fazwaz import ListingRecord


def test_coerce_bedrooms_studio_lowercase():
    assert _coerce_bedrooms("Studio") == 0


def test_coerce_bedrooms_studio_mixed_case():
    assert _coerce_bedrooms("STUDIO") == 0


def test_coerce_bedrooms_none():
    assert _coerce_bedrooms(None) == 0


def test_coerce_bedrooms_int():
    assert _coerce_bedrooms(2) == 2


def test_coerce_bedrooms_int_string():
    assert _coerce_bedrooms("3") == 3


def test_coerce_bedrooms_invalid():
    assert _coerce_bedrooms("two") == 0


def test_coerce_bathrooms_none():
    assert _coerce_bathrooms(None) == 0.0


def test_coerce_bathrooms_float():
    assert _coerce_bathrooms(2.5) == 2.5


def test_coerce_bathrooms_float_string():
    assert _coerce_bathrooms("2.5") == 2.5


def test_coerce_bathrooms_invalid():
    assert _coerce_bathrooms("two") == 0.0


def test_listings_to_dataframe_coerces_studio_and_null_bathrooms():
    rec = ListingRecord(
        listing_id="L1",
        name="Studio Condo",
        price="฿1,000,000",
        first_price=None,
        detail_url="https://example.com/1",
        address="Bangkok",
        area_sqm="35 sqm",
        bedrooms="Studio",
        bathrooms=None,
        property_type="Condo",
        transit_stations=[{"name": "Asoke", "distance": "500m"}],
        listed_date="listed 2 days ago",
        updated_date=None,
    )
    df = listings_to_dataframe([rec])
    assert len(df) == 1
    assert df.loc[0, "bedrooms"] == 0
    assert df.loc[0, "bathrooms"] == 0.0


def test_listings_to_dataframe_preserves_valid_values():
    rec = ListingRecord(
        listing_id="L2",
        name="Two-bed Condo",
        price="฿2,000,000",
        first_price="฿2,100,000",
        detail_url="https://example.com/2",
        address="Bangkok",
        area_sqm="55 sqm",
        bedrooms=2,
        bathrooms=2.0,
        property_type="Condo",
        transit_stations=[],
        listed_date="listed 1 week ago",
        updated_date="updated 1 day ago",
    )
    df = listings_to_dataframe([rec])
    assert df.loc[0, "bedrooms"] == 2
    assert df.loc[0, "bathrooms"] == 2.0
    assert df.loc[0, "listing_id"] == "L2"


def test_listings_to_dataframe_serializes_transit_stations():
    rec = ListingRecord(
        listing_id="L3",
        name="Transit Condo",
        price="฿3,000,000",
        first_price=None,
        detail_url="https://example.com/3",
        address="Bangkok",
        area_sqm="45 sqm",
        bedrooms=1,
        bathrooms=1.0,
        property_type="Condo",
        transit_stations=[{"name": "Asoke", "distance": "500m"}],
        listed_date=None,
        updated_date=None,
    )
    df = listings_to_dataframe([rec])
    assert "Asoke" in df.loc[0, "transit_stations_json"]

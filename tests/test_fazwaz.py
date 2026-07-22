"""Tests for sources/fazwaz.py — Phase 2 scraper internals."""
from __future__ import annotations

from sources.fazwaz import CardRecord, _dedup_cards


def _make_card(listing_id: str, name: str = "Condo") -> CardRecord:
    return CardRecord(
        listing_id=listing_id,
        name=name,
        price="฿1,000,000",
        first_price=None,
        detail_url="https://example.com/" + listing_id,
        address="Bangkok",
        area_sqm="35 sqm",
        bedrooms=1,
        bathrooms=1,
        property_type="Condo",
        transit_stations=[],
        listed_date="listed 2 days ago",
        updated_date=None,
    )


def test_dedup_cards_removes_duplicates():
    cards = [_make_card("L1"), _make_card("L1"), _make_card("L2")]
    result = _dedup_cards(cards)
    assert len(result) == 2
    assert [c.listing_id for c in result] == ["L1", "L2"]


def test_dedup_cards_keeps_unique():
    cards = [_make_card("L1"), _make_card("L2"), _make_card("L3")]
    result = _dedup_cards(cards)
    assert len(result) == 3


def test_dedup_cards_empty():
    result = _dedup_cards([])
    assert result == []


def test_dedup_cards_keeps_first_occurrence():
    cards = [
        _make_card("L1", "First"),
        _make_card("L2", "Second"),
        _make_card("L1", "Duplicate"),
    ]
    result = _dedup_cards(cards)
    assert len(result) == 2
    assert result[0].name == "First"
    assert result[1].name == "Second"

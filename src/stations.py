"""Fetch Bangkok BTS/MRT station coordinates via OpenStreetMap Overpass API.

Queries route relations for all rapid-transit lines in Greater Bangkok, extracts
member station nodes with lat/lng and line membership, classifies each line as
operational or upcoming, and writes a GeoJSON file to data/processed/stations.geojson.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

BBOX: tuple[float, float, float, float] = (13.45, 100.30, 13.95, 100.85)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "stations.geojson"
RAW_CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "overpass_response.json"

QUERY = """
[out:json][timeout:180];
(
  relation["type"="route"]["route"~"subway|light_rail|monorail|tram|train"]
    ({s},{w},{n},{e});
)->.routes;
(
  node["railway"~"station|halt|tram_stop|stop"]({s},{w},{n},{e});
  node["railway"="construction"]["name"]({s},{w},{n},{e});
  node["public_transport"="station"]({s},{w},{n},{e});
);
out body;
.routes out body;
""".format(s=BBOX[0], w=BBOX[1], n=BBOX[2], e=BBOX[3])


def classify_line(name: str, network: str, ref: str) -> tuple[str, str]:
    blob = f"{name} {network} {ref}".lower()

    if "orange" in blob or "สีส้ม" in blob:
        return "MRT Orange Line", "upcoming"
    if "purple" in blob or "สีม่วง" in blob:
        if "south" in blob or "ratburana" in blob or "rat burana" in blob:
            return "MRT Purple Line South", "upcoming"
        return "MRT Purple Line", "operational"
    if "brown" in blob or "สีน้ำตาล" in blob:
        return "MRT Brown Line", "upcoming"
    if "grey" in blob or "gray" in blob or "สีเทา" in blob:
        return "MRT Grey Line", "upcoming"
    if "silver" in blob or "สีเงิน" in blob:
        return "MRT Silver Line", "upcoming"
    if "light blue" in blob or "สีฟ้า" in blob:
        return "MRT Light Blue Line", "upcoming"

    if "sukhumvit" in blob or "สุขุมวิท" in blob:
        return "BTS Sukhumvit Line", "operational"
    if "silom" in blob or "สีลม" in blob:
        return "BTS Silom Line", "operational"
    if "gold" in blob or "สีทอง" in blob:
        return "BTS Gold Line", "operational"
    if "blue" in blob or "chaloem" in blob or "chalerm" in blob or "สีน้ำเงิน" in blob:
        return "MRT Blue Line", "operational"
    if "yellow" in blob or "nakkhara" in blob or "nakkara" in blob or "สีเหลือง" in blob:
        return "MRT Yellow Line", "operational"
    if "pink" in blob or "สีชมพู" in blob:
        return "MRT Pink Line", "operational"
    if "dark red" in blob or "สีแดงเข้ม" in blob:
        return "SRT Dark Red Line", "operational"
    if "light red" in blob or "สีแดงอ่อน" in blob:
        return "SRT Light Red Line", "operational"
    if "airport" in blob or "arl" in blob or "เชื่อมท่าอากาศยาน" in blob:
        return "Airport Rail Link", "operational"

    return name or "Unknown", "exclude"


_REF_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"^(N\d+|E\d+|CEN)$", re.I), "BTS Sukhumvit Line", "operational"),
    (re.compile(r"^(S\d+|W\d+)$", re.I), "BTS Silom Line", "operational"),
    (re.compile(r"^G\d+$", re.I), "BTS Gold Line", "operational"),
    (re.compile(r"^BL\d+", re.I), "MRT Blue Line", "operational"),
    (re.compile(r"^(PP|PR)\d+", re.I), "MRT Purple Line", "operational"),
    (re.compile(r"^YL\d+", re.I), "MRT Yellow Line", "operational"),
    (re.compile(r"^PK\d+", re.I), "MRT Pink Line", "operational"),
    (re.compile(r"^RN\d+", re.I), "SRT Dark Red Line", "operational"),
    (re.compile(r"^RW\d+", re.I), "SRT Light Red Line", "operational"),
    (re.compile(r"^A\d+$", re.I), "Airport Rail Link", "operational"),
    (re.compile(r"^OR\d+", re.I), "MRT Orange Line", "upcoming"),
]


def classify_by_ref(ref: str) -> tuple[str, str] | None:
    for pattern, line_name, status in _REF_PATTERNS:
        if pattern.match(ref.strip()):
            return line_name, status
    return None


def classify_by_name(name: str) -> tuple[str, str] | None:
    n = name.lower().strip()
    arl_only = {"rajprarop", "thanya"}
    if n in arl_only:
        return "Airport Rail Link", "operational"
    return None


def fetch_overpass(query: str) -> dict:
    headers = {
        "User-Agent": "BangkokTransitPropertyAnalysis/1.0 (portfolio project)",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }
    last_error: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        for method in ("post", "get"):
            try:
                label = f"{method.upper()} {endpoint}"
                print(f"Querying ({label}) ...")
                if method == "post":
                    resp = httpx.post(
                        endpoint,
                        data={"data": query},
                        headers=headers,
                        timeout=240,
                    )
                else:
                    resp = httpx.get(
                        endpoint,
                        params={"data": query},
                        headers=headers,
                        timeout=240,
                    )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                print(f"  failed: {exc}")
                last_error = exc
                time.sleep(5)

    raise RuntimeError(f"All Overpass endpoints failed: {last_error}")


def _set_line_status(line_status: dict[str, str], line_name: str, status: str) -> None:
    if status == "operational":
        line_status[line_name] = "operational"
    elif line_name not in line_status:
        line_status[line_name] = status


def parse_overpass(data: dict) -> tuple[list[dict], dict[str, str]]:
    relations: dict[int, dict] = {}
    nodes: dict[int, dict] = {}
    node_to_relations: dict[int, set[int]] = defaultdict(set)
    line_status: dict[str, str] = {}

    for el in data.get("elements", []):
        if el["type"] == "relation":
            rid = el["id"]
            relations[rid] = el
            for member in el.get("members", []):
                if member["type"] == "node":
                    node_to_relations[member["ref"]].add(rid)
        elif el["type"] == "node":
            nodes[el["id"]] = el

    features: list[dict] = []
    for nid, node in nodes.items():
        tags = node.get("tags", {})
        name = tags.get("name:en") or tags.get("name") or ""
        ref = tags.get("ref") or ""
        wikidata = tags.get("wikidata") or ""
        railway = tags.get("railway") or ""
        is_construction = railway == "construction"

        line_names: set[str] = set()
        statuses: set[str] = set()

        for rid in node_to_relations.get(nid, set()):
            rel = relations.get(rid)
            if not rel:
                continue
            rel_tags = rel.get("tags", {})
            rel_name = rel_tags.get("name") or rel_tags.get("name:en") or ""
            rel_network = rel_tags.get("network") or ""
            rel_ref = rel_tags.get("ref") or ""
            line_name, status = classify_line(rel_name, rel_network, rel_ref)
            if status != "exclude":
                line_names.add(line_name)
                statuses.add(status)
                _set_line_status(line_status, line_name, status)

        if not line_names and ref:
            ref_result = classify_by_ref(ref)
            if ref_result:
                line_name, status = ref_result
                if is_construction:
                    status = "upcoming"
                line_names.add(line_name)
                statuses.add(status)
                _set_line_status(line_status, line_name, status)

        if not line_names and name:
            name_result = classify_by_name(name)
            if name_result:
                line_name, status = name_result
                if is_construction:
                    status = "upcoming"
                line_names.add(line_name)
                statuses.add(status)
                _set_line_status(line_status, line_name, status)

        if not line_names:
            continue

        if is_construction:
            statuses.discard("operational")
            statuses.add("upcoming")

        operational = "operational" in statuses
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(node["lon"], 6), round(node["lat"], 6)],
            },
            "properties": {
                "name": name,
                "ref": ref,
                "wikidata": wikidata,
                "lines": sorted(line_names),
                "operational": operational,
                "osm_id": nid,
            },
        }
        features.append(feature)

    return features, line_status


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def deduplicate(features: list[dict]) -> list[dict]:
    by_ref: dict[str, int] = {}
    by_name_loc: dict[str, int] = {}
    result: list[dict] = []

    def merge_into(existing: dict, new: dict) -> None:
        existing["properties"]["lines"] = sorted(
            set(existing["properties"]["lines"]) | set(new["properties"]["lines"])
        )
        existing["properties"]["operational"] = (
            existing["properties"]["operational"] or new["properties"]["operational"]
        )
        if not existing["properties"]["ref"] and new["properties"]["ref"]:
            existing["properties"]["ref"] = new["properties"]["ref"]
        if not existing["properties"]["name"] and new["properties"]["name"]:
            existing["properties"]["name"] = new["properties"]["name"]

    for f in features:
        ref = f["properties"]["ref"]
        lon, lat = f["geometry"]["coordinates"]
        name = f["properties"]["name"]

        if ref and ref.upper() in by_ref:
            merge_into(result[by_ref[ref.upper()]], f)
            continue

        loc_key = f"{name}_{round(lat, 3)}_{round(lon, 3)}"
        if loc_key in by_name_loc:
            merge_into(result[by_name_loc[loc_key]], f)
            if ref:
                by_ref[ref.upper()] = by_name_loc[loc_key]
            continue

        if name:
            merged = False
            for idx, existing in enumerate(result):
                ex_name = existing["properties"]["name"]
                ex_lon, ex_lat = existing["geometry"]["coordinates"]
                dist = _haversine_m(lat, lon, ex_lat, ex_lon)
                if dist <= 100 and ex_name == name:
                    merge_into(existing, f)
                    if ref:
                        by_ref[ref.upper()] = idx
                    by_name_loc[loc_key] = idx
                    merged = True
                    break
                if dist <= 50 and existing["properties"]["lines"] == f["properties"]["lines"]:
                    merge_into(existing, f)
                    if ref:
                        by_ref[ref.upper()] = idx
                    by_name_loc[loc_key] = idx
                    merged = True
                    break
            if merged:
                continue

        idx = len(result)
        result.append(f)
        if ref:
            by_ref[ref.upper()] = idx
        by_name_loc[loc_key] = idx

    return result


def print_summary(features: list[dict], line_status: dict[str, str]) -> None:
    line_counts: dict[str, int] = defaultdict(int)
    for f in features:
        for line in f["properties"]["lines"]:
            line_counts[line] += 1

    op_count = sum(1 for f in features if f["properties"]["operational"])
    upcoming_count = len(features) - op_count

    print("\n" + "=" * 60)
    print("  Bangkok Rapid Transit Stations — Summary")
    print("=" * 60)
    print(f"{'Line':<32} {'Stations':>8}  {'Status':<12}")
    print("-" * 60)
    for line in sorted(line_counts):
        status = line_status.get(line, "?")
        print(f"  {line:<30} {line_counts[line]:>8}  {status:<12}")
    print("-" * 60)
    print(f"  {'TOTAL STATIONS':<30} {len(features):>8}")
    print(f"  {'  operational':<30} {op_count:>8}")
    print(f"  {'  upcoming-only':<30} {upcoming_count:>8}")
    print("=" * 60)


def main() -> None:
    data = fetch_overpass(QUERY)

    RAW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False))
    print(f"Raw Overpass response cached to {RAW_CACHE_PATH}")

    features, line_status = parse_overpass(data)
    features = deduplicate(features)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "source": "OpenStreetMap (Overpass API)",
            "bbox": list(BBOX),
            "total_stations": len(features),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(geojson, ensure_ascii=False, indent=2))
    print(f"Written {len(features)} stations to {OUTPUT_PATH}")

    print_summary(features, line_status)


if __name__ == "__main__":
    sys.exit(main())

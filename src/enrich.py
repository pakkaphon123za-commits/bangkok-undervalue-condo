"""Enrich clean listings with nearest transit station distances.

Usage:
    python src/enrich.py
    python src/enrich.py --input data/interim/listings_clean.parquet
    python src/enrich.py --stations data/processed/stations.geojson
    python src/enrich.py --include-planned   # include non-operational stations
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATIONS = PROJECT_ROOT / "data" / "processed" / "stations.geojson"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "listings_clean.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "listings_enriched.parquet"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_vectorized(
    lats: np.ndarray,
    lons: np.ndarray,
    s_lats: np.ndarray,
    s_lons: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine. lats/lons: (N,), s_lats/s_lons: (M,) -> (N, M) km."""
    R = 6371.0
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    s_lats_rad = np.radians(s_lats)
    s_lons_rad = np.radians(s_lons)

    dlat = s_lats_rad[np.newaxis, :] - lats_rad[:, np.newaxis]
    dlon = s_lons_rad[np.newaxis, :] - lons_rad[:, np.newaxis]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lats_rad[:, np.newaxis])
        * np.cos(s_lats_rad[np.newaxis, :])
        * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_stations(
    geojson_path: Path,
    include_planned: bool = False,
) -> list[dict[str, Any]]:
    with open(geojson_path, encoding="utf-8") as f:
        data = json.load(f)
    stations = []
    for feat in data["features"]:
        props = feat["properties"]
        operational = props.get("operational", True)
        if not include_planned and not operational:
            continue
        lon, lat = feat["geometry"]["coordinates"]
        stations.append(
            {
                "name": props["name"],
                "ref": props.get("ref", ""),
                "line": ", ".join(props.get("lines", [])),
                "lat": lat,
                "lon": lon,
                "operational": operational,
            }
        )
    return stations


def find_nearest(
    lat: float,
    lon: float,
    stations: list[dict[str, Any]],
) -> tuple[str, float, str]:
    best_name = ""
    best_dist = float("inf")
    best_line = ""
    for s in stations:
        dist = haversine_km(lat, lon, s["lat"], s["lon"])
        if dist < best_dist:
            best_dist = dist
            best_name = s["name"]
            best_line = s["line"]
    return best_name, best_dist, best_line


def enrich(
    df: pd.DataFrame,
    stations: list[dict[str, Any]],
) -> pd.DataFrame:
    if not stations:
        df["nearest_station"] = None
        df["nearest_station_km"] = None
        df["nearest_station_line"] = None
        return df

    has_coords = df["latitude"].notna() & df["longitude"].notna()
    coord_df = df[has_coords]

    s_lats = np.array([s["lat"] for s in stations])
    s_lons = np.array([s["lon"] for s in stations])
    s_names = [s["name"] for s in stations]
    s_lines = [s["line"] for s in stations]

    if len(coord_df) > 0:
        lats = coord_df["latitude"].to_numpy(dtype=float)
        lons = coord_df["longitude"].to_numpy(dtype=float)
        dists = haversine_vectorized(lats, lons, s_lats, s_lons)
        nearest_idx = np.argmin(dists, axis=1)
        nearest_dists = dists[np.arange(len(coord_df)), nearest_idx]

        nearest_names = [s_names[i] for i in nearest_idx]
        nearest_lines = [s_lines[i] for i in nearest_idx]
        nearest_dists_rounded = np.round(nearest_dists, 3)
    else:
        nearest_names = []
        nearest_lines = []
        nearest_dists_rounded = []

    df["nearest_station"] = None
    df["nearest_station_km"] = None
    df["nearest_station_line"] = None
    df.loc[has_coords, "nearest_station"] = nearest_names
    df.loc[has_coords, "nearest_station_km"] = nearest_dists_rounded
    df.loc[has_coords, "nearest_station_line"] = nearest_lines

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich listings with station distances")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-planned",
        action="store_true",
        help="Include non-operational (planned) stations in matching",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}")
        return

    print(f"Loading listings: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"  {len(df)} listings, {df['latitude'].notna().sum()} with coords")

    print(f"Loading stations: {args.stations}")
    stations = load_stations(args.stations, include_planned=args.include_planned)
    total_stations = len(stations)
    op_count = sum(1 for s in stations if s["operational"])
    if args.include_planned:
        print(f"  {total_stations} stations (including {total_stations - op_count} planned)")
    else:
        print(f"  {total_stations} operational stations ({op_count} operational)")

    df = enrich(df, stations)

    matched = df["nearest_station"].notna().sum()
    print(f"Matched: {matched}/{len(df)}")
    if matched > 0:
        print(f"  Min distance: {df['nearest_station_km'].min():.3f} km")
        print(f"  Max distance: {df['nearest_station_km'].max():.3f} km")
        print(f"  Avg distance: {df['nearest_station_km'].mean():.3f} km")
        print()
        print("Sample matches:")
        sample = df.dropna(subset=["nearest_station"]).head(5)
        for _, r in sample.iterrows():
            print(
                f"  {r['name'][:35]:35s} -> {r['nearest_station']:25s} "
                f"({r['nearest_station_km']:.3f} km) [{r['nearest_station_line']}]"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

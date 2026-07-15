"""Generate LLM-written analyst narrative for Bangkok transit-property analysis."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "listings_modeled.parquet"
DEFAULT_DECAY = PROJECT_ROOT / "data" / "processed" / "decay_curves.json"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "undervalued_summary.json"
DEFAULT_NARRATIVE_OUTPUT = PROJECT_ROOT / "docs" / "narrative.md"
DEFAULT_META_OUTPUT = PROJECT_ROOT / "data" / "processed" / "narrative_meta.json"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_env_key(path: Path, key: str = "OLLAMA_API_KEY") -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def compute_station_stats(df: pd.DataFrame, min_n: int = 5) -> pd.DataFrame:
    df = df.copy()
    df["undervalued_by_pct"] = df["undervalued_by_pct"].fillna(0.0)

    grouped = (
        df.groupby("nearest_station")
        .agg(
            n=("nearest_station", "size"),
            line=("nearest_station_line", "first"),
            n_undervalued=("is_undervalued", "sum"),
            median_zscore=("residual_zscore", "median"),
        )
        .reset_index()
        .rename(columns={"nearest_station": "station"})
    )
    grouped["n_undervalued"] = grouped["n_undervalued"].astype(int)

    und = (
        df[df["is_undervalued"]]
        .groupby("nearest_station")["undervalued_by_pct"]
        .median()
        .reset_index()
        .rename(columns={"nearest_station": "station", "undervalued_by_pct": "median_undervalued_by_pct"})
    )
    grouped = grouped.merge(und, on="station", how="left")
    grouped["median_undervalued_by_pct"] = grouped["median_undervalued_by_pct"].fillna(0.0).round(2)
    grouped["pct_undervalued"] = (grouped["n_undervalued"] / grouped["n"] * 100).round(2)

    grouped = grouped[grouped["n"] >= min_n].sort_values(by="n_undervalued", ascending=False).reset_index(drop=True)
    return grouped


def call_llm(
    messages: list[dict],
    base_url: str,
    model: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    timeout: float = 60.0,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    last_error: Exception | None = None
    for _ in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code >= 500:
                last_error = e
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_error = e
            continue
    raise last_error or RuntimeError("LLM request failed after retry")


def build_prompt(decay: dict, summary: dict, station_stats: pd.DataFrame) -> list[dict]:
    global_data = summary["global"]
    lines: list[dict] = []
    for line_name, line_summary in summary["lines"].items():
        decay_line = decay.get("lines", {}).get(line_name, {})
        lines.append({
            "name": line_name,
            "n": line_summary["n"],
            "n_undervalued": line_summary["n_undervalued"],
            "pct_undervalued": line_summary["pct_undervalued"],
            "decay_pct_per_km": decay_line.get("decay_pct_per_km"),
            "used_global_stats": line_summary.get("used_global_stats", False),
        })

    top_stations: list[dict] = []
    for _, row in station_stats.head(10).iterrows():
        top_stations.append({
            "name": row["station"],
            "line": row["line"],
            "n": int(row["n"]),
            "n_undervalued": int(row["n_undervalued"]),
            "pct_undervalued": float(row["pct_undervalued"]),
            "median_undervalued_by_pct": float(row["median_undervalued_by_pct"]),
        })

    system_msg = (
        "You are a real-estate analyst writing an investor brief on Bangkok condos "
        "near mass transit. Write in clear English prose, preserve Thai line and station "
        "names when the data includes them, avoid hedging filler, lead with the strongest "
        "finding, reference upcoming lines (MRT Orange, MRT Purple South, SRT Dark Red "
        "south extension) where relevant, and never fabricate numbers — only use the "
        "statistics provided."
    )

    user_payload = {
        "global": global_data,
        "lines": lines,
        "top_undervalued_stations": top_stations,
        "instructions": (
            "Produce a Markdown brief with exactly these sections: "
            "# Bangkok Condo Market Brief, then an executive summary of 2–3 paragraphs, "
            "then ## Per-line analysis with 1–2 sentences per line, "
            "then ## Undervalued station zones with commentary on the top stations, "
            "then ## Upcoming line impact, and end with a one-line methodology footnote."
        ),
    }

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


if __name__ == "__main__":
    pass

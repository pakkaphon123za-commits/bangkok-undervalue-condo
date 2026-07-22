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


def build_meta(decay: dict, summary: dict, station_stats: pd.DataFrame, model: str) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    lines: list[dict] = []
    for line_name, line_summary in summary["lines"].items():
        decay_line = decay.get("lines", {}).get(line_name, {})
        lines.append({
            "name": line_name,
            "name_th": "",
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
            "name_th": "",
            "line": row["line"],
            "n": int(row["n"]),
            "n_undervalued": int(row["n_undervalued"]),
            "pct_undervalued": float(row["pct_undervalued"]),
            "median_undervalued_by_pct": float(row["median_undervalued_by_pct"]),
        })

    return {
        "generated_at": now,
        "model": model,
        "global": summary["global"],
        "lines": lines,
        "top_stations": top_stations,
    }


def render_narrative(markdown_text: str, station_stats: pd.DataFrame) -> str:
    text = markdown_text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = ["### Top undervalued stations", ""]
    lines.append("| Station | Line | N | Undervalued | Median discount |")
    lines.append("|---|---|---:|---:|---:|")
    if len(station_stats) == 0:
        lines.append("| — | — | — | — | — |")
    else:
        for _, row in station_stats.iterrows():
            lines.append(
                f"| {row['station']} | {row['line']} | {int(row['n'])} | "
                f"{int(row['n_undervalued'])} ({float(row['pct_undervalued']):.2f}%) | "
                f"{float(row['median_undervalued_by_pct']):.1f}% |"
            )
    table = "\n".join(lines)
    return f"{text}\n\n{table}\n"


def write_outputs(narrative_md: str, meta_dict: dict, narrative_path: Path, meta_path: Path) -> None:
    narrative_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_narrative = narrative_path.with_suffix(narrative_path.suffix + ".tmp")
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")

    with open(tmp_narrative, "w", encoding="utf-8") as f:
        f.write(narrative_md if narrative_md.endswith("\n") else narrative_md + "\n")
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)

    os.replace(tmp_narrative, narrative_path)
    os.replace(tmp_meta, meta_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate LLM narrative for Bangkok transit-property analysis")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--decay", type=Path, default=DEFAULT_DECAY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--narrative-output", type=Path, default=DEFAULT_NARRATIVE_OUTPUT)
    parser.add_argument("--meta-output", type=Path, default=DEFAULT_META_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    llm_config = config.get("llm", {})
    if not llm_config.get("enabled", False):
        print("LLM disabled in config; skipping narrative generation.")
        return

    env_key_name = llm_config.get("api_key_env", "OLLAMA_API_KEY")
    api_key = load_env_key(PROJECT_ROOT / ".env", key=env_key_name)
    if not api_key:
        print(f"{env_key_name} not found in .env; aborting.")
        raise SystemExit(1)

    base_url = llm_config.get("base_url", "")
    model = llm_config.get("model", "glm-5.2")
    temperature = float(llm_config.get("temperature", 0.7))
    max_tokens = int(llm_config.get("max_tokens", 2000))
    timeout = float(llm_config.get("timeout", 60.0))

    for path, label in [(args.input, "Input"), (args.decay, "Decay curves"), (args.summary, "Summary")]:
        if not path.exists():
            print(f"{label} not found: {path}")
            raise SystemExit(1)

    df = pd.read_parquet(args.input)
    with open(args.decay, encoding="utf-8") as f:
        decay = json.load(f)
    with open(args.summary, encoding="utf-8") as f:
        summary = json.load(f)

    station_stats = compute_station_stats(df)
    messages = build_prompt(decay, summary, station_stats)
    raw_narrative = call_llm(messages, base_url, model, api_key, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    if len(raw_narrative.strip()) < 100:
        print("Warning: LLM returned a very short narrative.")
    narrative_text = render_narrative(raw_narrative, station_stats)
    meta = build_meta(decay, summary, station_stats, model)
    write_outputs(narrative_text, meta, args.narrative_output, args.meta_output)

    print(f"Narrative saved to {args.narrative_output}")
    print(f"Meta saved to {args.meta_output}")
    print(f"Lines analyzed: {len(summary['lines'])}")
    print(f"Top stations: {len(station_stats.head(10))}")


if __name__ == "__main__":
    main()

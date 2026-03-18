"""
Viral DNA analyzer.

Takes chart data and extracts:
  - Audio fingerprints of what's trending (BPM, key, energy, valence)
  - Hook timing patterns (where the drop/chorus hits)
  - Cross-cultural overlap (songs that trend everywhere = universal DNA)
  - Language distribution per region
  - Trend velocity (songs climbing fast = early signal)

This data becomes the conditioning signal for training:
  model learns to generate lyrics that match viral audio DNA.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


# ── Load chart data ───────────────────────────────────────────────────────────

def load_chart_week(week: str, charts_dir: str = "data/charts") -> list[dict]:
    path = Path(charts_dir) / f"{week}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No chart data for week {week} — run chart_scraper.py first")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_viral_scores(week: str, charts_dir: str = "data/charts") -> list[dict]:
    path = Path(charts_dir) / f"viral_scores_{week}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# ── Audio DNA extraction ──────────────────────────────────────────────────────

AUDIO_FEATURES = ["bpm", "danceability", "energy", "valence", "acousticness", "speechiness"]

def extract_audio_dna(records: list[dict], region: Optional[str] = None) -> dict:
    """
    Compute the average audio fingerprint of charting songs.
    If region is specified, filter to that region only.
    """
    filtered = records if not region else [r for r in records if r.get("region") == region]
    # Only records with audio features
    with_audio = [r for r in filtered if r.get("bpm", 0) > 0]

    if not with_audio:
        return {}

    dna = {}
    for feat in AUDIO_FEATURES:
        vals = [r[feat] for r in with_audio if feat in r and r[feat] is not None]
        if vals:
            dna[feat] = {
                "mean":   round(float(np.mean(vals)), 3),
                "std":    round(float(np.std(vals)), 3),
                "median": round(float(np.median(vals)), 3),
                "p25":    round(float(np.percentile(vals, 25)), 3),
                "p75":    round(float(np.percentile(vals, 75)), 3),
            }

    # BPM distribution buckets
    bpms = [r["bpm"] for r in with_audio if r.get("bpm", 0) > 0]
    dna["bpm_buckets"] = {
        "slow (60-90)":   sum(1 for b in bpms if 60 <= b < 90),
        "mid (90-120)":   sum(1 for b in bpms if 90 <= b < 120),
        "fast (120-150)": sum(1 for b in bpms if 120 <= b < 150),
        "hyper (150+)":   sum(1 for b in bpms if b >= 150),
    }

    # Key distribution (0=C, 1=C#, 2=D, ...)
    KEY_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    keys = [r["key"] for r in with_audio if r.get("key", -1) >= 0]
    key_dist = defaultdict(int)
    for k in keys:
        key_dist[KEY_NAMES[k]] += 1
    dna["key_distribution"] = dict(sorted(key_dist.items(), key=lambda x: -x[1]))

    # Major vs minor
    modes = [r.get("mode", -1) for r in with_audio if r.get("mode", -1) >= 0]
    dna["major_ratio"] = round(sum(1 for m in modes if m == 1) / max(len(modes), 1), 3)

    return dna


# ── Trend velocity ────────────────────────────────────────────────────────────

def compute_trend_velocity(records: list[dict]) -> list[dict]:
    """
    Songs with low weeks_on_chart but high streams = rising fast.
    These are the early signals — train on these more.
    """
    rising = []
    for r in records:
        weeks = r.get("weeks_on_chart", 1)
        streams = r.get("streams", 0)
        rank = r.get("rank", 100)

        if weeks <= 2 and rank <= 30 and streams > 0:
            velocity = streams / max(weeks, 1) / max(rank, 1)
            rising.append({**r, "velocity": round(velocity, 2)})

    rising.sort(key=lambda x: x["velocity"], reverse=True)
    return rising


# ── Cross-cultural DNA ────────────────────────────────────────────────────────

def extract_universal_dna(viral_scores: list[dict], min_countries: int = 10) -> dict:
    """
    Extract audio DNA from songs that chart in 10+ countries.
    These songs have proven cross-cultural appeal — the model should learn their patterns.
    """
    universal = [s for s in viral_scores if s.get("n_countries", 0) >= min_countries]

    if not universal:
        return {"message": f"No songs found in {min_countries}+ countries yet"}

    dna = {
        "n_universal_songs": len(universal),
        "min_countries_threshold": min_countries,
        "top_songs": [
            {
                "title":       s["title"],
                "artist":      s["artist"],
                "n_countries": s["n_countries"],
                "viral_score": s["viral_score"],
                "avg_rank":    s["avg_rank"],
            }
            for s in universal[:20]
        ],
    }

    # Audio features of universal hits
    for feat in ["bpm", "danceability", "energy", "valence", "speechiness"]:
        vals = [s[feat] for s in universal if feat in s and s[feat]]
        if vals:
            dna[f"universal_{feat}"] = {
                "mean": round(float(np.mean(vals)), 3),
                "std":  round(float(np.std(vals)), 3),
            }

    return dna


# ── Region style profiles ─────────────────────────────────────────────────────

REGIONS = ["Africa", "Middle East", "Asia", "Latin America", "Europe", "Americas", "Oceania"]

def build_region_profiles(records: list[dict]) -> dict[str, dict]:
    """Build audio DNA per region — used for style conditioning."""
    profiles = {}
    for region in REGIONS:
        dna = extract_audio_dna(records, region=region)
        if dna:
            region_records = [r for r in records if r.get("region") == region]
            profiles[region] = {
                "n_songs":    len(region_records),
                "n_countries": len(set(r["country"] for r in region_records)),
                "audio_dna":  dna,
                "top_artists": _top_artists(region_records, n=10),
            }
    return profiles


def _top_artists(records: list[dict], n: int = 10) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        if r.get("artist"):
            counts[r["artist"]] += 1
    return [a for a, _ in sorted(counts.items(), key=lambda x: -x[1])[:n]]


# ── Training conditioning vector ──────────────────────────────────────────────

def build_conditioning_vector(
    genre: str,
    region: Optional[str],
    viral_scores: list[dict],
    records: list[dict],
) -> np.ndarray:
    """
    Build a conditioning vector that encodes:
      - Target genre audio DNA (BPM, energy, valence, danceability)
      - Region style profile
      - Viral score stats
    This gets injected into the model alongside the style vector.
    Returns a 32-dim float vector.
    """
    vec = np.zeros(32, dtype=np.float32)

    # Global audio DNA (dims 0-7)
    dna = extract_audio_dna(records)
    for i, feat in enumerate(["bpm", "danceability", "energy", "valence",
                               "acousticness", "speechiness", "major_ratio"]):
        if feat == "major_ratio":
            vec[i] = dna.get("major_ratio", 0.5)
        elif feat in dna:
            vec[i] = dna[feat].get("mean", 0)
        # Normalize BPM to 0-1
    if vec[0] > 1:
        vec[0] = vec[0] / 250.0

    # Region profile (dims 8-15)
    if region:
        region_dna = extract_audio_dna(records, region=region)
        for i, feat in enumerate(["bpm", "danceability", "energy", "valence"]):
            if feat in region_dna:
                val = region_dna[feat].get("mean", 0)
                vec[8 + i] = val / 250.0 if feat == "bpm" else val

    # Viral stats (dims 16-23)
    if viral_scores:
        top_viral = viral_scores[:50]
        viral_vals = [s["viral_score"] for s in top_viral]
        vec[16] = float(np.mean(viral_vals)) / 100.0
        vec[17] = float(np.max(viral_vals)) / 100.0
        n_country_vals = [s["n_countries"] for s in top_viral]
        vec[18] = float(np.mean(n_country_vals)) / 55.0  # normalize to max countries

    # Remaining dims 24-31 reserved for future features
    return vec


# ── Full analysis report ──────────────────────────────────────────────────────

def run_analysis(week: str = None, charts_dir: str = "data/charts") -> dict:
    import datetime
    if not week:
        week = datetime.datetime.utcnow().strftime("%Y-%W")

    print(f"Analyzing charts for week {week}...")

    records      = load_chart_week(week, charts_dir)
    viral_scores = load_viral_scores(week, charts_dir)

    print(f"  {len(records)} chart entries loaded")
    print(f"  {len(viral_scores)} unique songs scored")

    # Global DNA
    global_dna = extract_audio_dna(records)
    print(f"\nGlobal Audio DNA:")
    for feat in ["bpm", "danceability", "energy", "valence", "speechiness"]:
        if feat in global_dna:
            d = global_dna[feat]
            print(f"  {feat:15s} mean={d['mean']:.3f}  std={d['std']:.3f}  median={d['median']:.3f}")

    # Universal hits
    universal = extract_universal_dna(viral_scores, min_countries=8)
    print(f"\nUniversal hits ({universal.get('n_universal_songs', 0)} songs in 8+ countries):")
    for s in universal.get("top_songs", [])[:5]:
        print(f"  [{s['n_countries']:2d} countries | rank {s['avg_rank']:5.1f}] {s['title']} — {s['artist']}")

    # Rising fast
    rising = compute_trend_velocity(records)
    print(f"\nRising fast ({len(rising)} songs):")
    for s in rising[:5]:
        print(f"  [{s['rank']:3d}] {s['title']} — {s['artist']} ({s['country_name']}, {s['weeks_on_chart']}wk)")

    # Region profiles
    region_profiles = build_region_profiles(records)
    print(f"\nRegion profiles built for: {list(region_profiles.keys())}")

    # Save full analysis
    report = {
        "week":            week,
        "total_entries":   len(records),
        "unique_songs":    len(viral_scores),
        "global_dna":      global_dna,
        "universal_dna":   universal,
        "region_profiles": region_profiles,
        "rising_fast":     rising[:20],
        "top_viral":       viral_scores[:50],
    }

    out_file = Path(charts_dir) / f"analysis_{week}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull analysis saved → {out_file}")

    return report


if __name__ == "__main__":
    import sys
    week = sys.argv[1] if len(sys.argv) > 1 else None
    run_analysis(week)

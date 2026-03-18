"""
Global chart scraper — 55 countries x Top 100 = 5,500 songs/week.

Sources:
  - Spotify Charts API (official)
  - Fallback: web scrape charts.spotify.com

Output: data/charts/YYYY-WW.jsonl  (one record per chart entry)
Each record:
{
  "rank":        1,
  "country":     "MA",
  "country_name":"Morocco",
  "region":      "Africa",
  "track_id":    "spotify:track:xxx",
  "title":       "Song Name",
  "artist":      "Artist Name",
  "streams":     1234567,
  "peak_rank":   1,
  "weeks_on_chart": 3,
  "scraped_at":  "2026-03-17T00:00:00",
  "week":        "2026-13"
}
"""

import json
import os
import time
import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

CHARTS_DIR = Path("data/charts")
TRACKS_DIR = Path("data/chart_tracks")

# ── 55 Countries ──────────────────────────────────────────────────────────────

COUNTRIES = {
    # Africa
    "NG": ("Nigeria",       "Africa"),
    "ZA": ("South Africa",  "Africa"),
    "MA": ("Morocco",       "Africa"),
    "EG": ("Egypt",         "Africa"),
    "KE": ("Kenya",         "Africa"),
    "GH": ("Ghana",         "Africa"),
    "SN": ("Senegal",       "Africa"),
    "TZ": ("Tanzania",      "Africa"),
    "ET": ("Ethiopia",      "Africa"),
    "CI": ("Ivory Coast",   "Africa"),

    # Middle East
    "SA": ("Saudi Arabia",  "Middle East"),
    "AE": ("UAE",           "Middle East"),
    "LB": ("Lebanon",       "Middle East"),
    "TR": ("Turkey",        "Middle East"),
    "IL": ("Israel",        "Middle East"),
    "PK": ("Pakistan",      "Middle East"),
    "IQ": ("Iraq",          "Middle East"),

    # Asia
    "KR": ("South Korea",   "Asia"),
    "JP": ("Japan",         "Asia"),
    "IN": ("India",         "Asia"),
    "PH": ("Philippines",   "Asia"),
    "ID": ("Indonesia",     "Asia"),
    "TH": ("Thailand",      "Asia"),
    "VN": ("Vietnam",       "Asia"),
    "MY": ("Malaysia",      "Asia"),
    "BD": ("Bangladesh",    "Asia"),
    "TW": ("Taiwan",        "Asia"),

    # Latin America
    "MX": ("Mexico",              "Latin America"),
    "BR": ("Brazil",              "Latin America"),
    "CO": ("Colombia",            "Latin America"),
    "AR": ("Argentina",           "Latin America"),
    "CL": ("Chile",               "Latin America"),
    "PE": ("Peru",                "Latin America"),
    "DO": ("Dominican Republic",  "Latin America"),
    "PR": ("Puerto Rico",         "Latin America"),
    "VE": ("Venezuela",           "Latin America"),
    "EC": ("Ecuador",             "Latin America"),

    # Europe
    "GB": ("UK",          "Europe"),
    "FR": ("France",      "Europe"),
    "DE": ("Germany",     "Europe"),
    "ES": ("Spain",       "Europe"),
    "IT": ("Italy",       "Europe"),
    "PT": ("Portugal",    "Europe"),
    "NL": ("Netherlands", "Europe"),
    "SE": ("Sweden",      "Europe"),
    "PL": ("Poland",      "Europe"),
    "RO": ("Romania",     "Europe"),
    "GR": ("Greece",      "Europe"),
    "NO": ("Norway",      "Europe"),

    # Americas + Oceania
    "US": ("USA",         "Americas"),
    "CA": ("Canada",      "Americas"),
    "AU": ("Australia",   "Oceania"),
    "NZ": ("New Zealand", "Oceania"),
    "JM": ("Jamaica",     "Americas"),
}

# ── Spotify auth ──────────────────────────────────────────────────────────────

def get_spotify_token() -> str:
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def spotify_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── Spotify Charts (via internal charts API) ──────────────────────────────────

CHARTS_URL = "https://charts-spotify-com-service.spotify.com/auth/v0/charts/regional-{country}-weekly/latest"

def fetch_chart_spotify(country_code: str, token: str) -> list[dict]:
    """Fetch top 100 for a country via Spotify's charts service."""
    url = CHARTS_URL.format(country=country_code.lower())
    try:
        r = requests.get(url, headers=spotify_headers(token), timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        entries = data.get("entries", [])
        results = []
        for entry in entries[:100]:
            track = entry.get("trackMetadata", {})
            chart_data = entry.get("chartEntryData", {})
            results.append({
                "rank":           chart_data.get("currentRank", 0),
                "peak_rank":      chart_data.get("peakRank", 0),
                "weeks_on_chart": chart_data.get("consecutiveChartEntryCount", 1),
                "streams":        chart_data.get("rankingMetric", {}).get("value", 0),
                "track_id":       track.get("trackUri", ""),
                "title":          track.get("trackName", ""),
                "artist":         ", ".join(a.get("name","") for a in track.get("artists", [])),
                "album":          track.get("albumName", ""),
                "duration_ms":    track.get("durationMs", 0),
            })
        return results
    except Exception as e:
        print(f"    Spotify charts error [{country_code}]: {e}")
        return []


# ── Fallback: scrape charts.spotify.com CSV ───────────────────────────────────

CHARTS_CSV_URL = "https://charts.spotify.com/charts/view/regional-{country}-weekly/latest"

def fetch_chart_csv_fallback(country_code: str) -> list[dict]:
    """Download the public CSV from charts.spotify.com as fallback."""
    import csv, io
    url = f"https://charts.spotify.com/charts/view/regional-{country_code.lower()}-weekly/latest"
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
            allow_redirects=True,
        )
        if "csv" not in r.headers.get("Content-Type", "") and r.status_code != 200:
            return []
        # Try to find CSV download link or parse directly
        lines = r.text.splitlines()
        reader = csv.DictReader(lines)
        results = []
        for i, row in enumerate(reader):
            if i >= 100:
                break
            results.append({
                "rank":           int(row.get("rank", i + 1)),
                "peak_rank":      int(row.get("peak_rank", i + 1)),
                "weeks_on_chart": int(row.get("weeks_on_chart", 1)),
                "streams":        int(row.get("streams", 0)) if row.get("streams", "").isdigit() else 0,
                "track_id":       row.get("uri", ""),
                "title":          row.get("track_name", ""),
                "artist":         row.get("artist_names", ""),
                "album":          row.get("source", ""),
                "duration_ms":    0,
            })
        return results
    except Exception as e:
        print(f"    CSV fallback error [{country_code}]: {e}")
        return []


# ── Track audio features ──────────────────────────────────────────────────────

def get_audio_features(track_ids: list[str], token: str) -> dict[str, dict]:
    """Fetch BPM, key, danceability, energy, valence for up to 100 tracks."""
    clean_ids = [tid.replace("spotify:track:", "") for tid in track_ids if tid]
    if not clean_ids:
        return {}
    features = {}
    batch_size = 50
    for i in range(0, len(clean_ids), batch_size):
        batch = clean_ids[i:i+batch_size]
        try:
            r = requests.get(
                "https://api.spotify.com/v1/audio-features",
                headers=spotify_headers(token),
                params={"ids": ",".join(batch)},
                timeout=10,
            )
            if r.status_code == 200:
                for feat in r.json().get("audio_features", []):
                    if feat:
                        features[feat["id"]] = {
                            "bpm":           feat.get("tempo", 0),
                            "key":           feat.get("key", -1),
                            "mode":          feat.get("mode", 0),        # 1=major, 0=minor
                            "danceability":  feat.get("danceability", 0),
                            "energy":        feat.get("energy", 0),
                            "valence":       feat.get("valence", 0),     # 0=sad, 1=happy
                            "acousticness":  feat.get("acousticness", 0),
                            "speechiness":   feat.get("speechiness", 0), # high = rap
                            "loudness":      feat.get("loudness", 0),
                            "duration_ms":   feat.get("duration_ms", 0),
                        }
            time.sleep(0.1)
        except Exception as e:
            print(f"    Audio features error: {e}")
    return features


# ── Main scrape loop ──────────────────────────────────────────────────────────

def scrape_all_charts(out_dir: str = "data/charts") -> list[dict]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    week = datetime.datetime.utcnow().strftime("%Y-%W")
    out_file = out_path / f"{week}.jsonl"
    now = datetime.datetime.utcnow().isoformat()

    # Load token
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        print("WARNING: SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET not set")
        print("Add them to your .env — get free credentials at developer.spotify.com")
        token = ""
    else:
        token = get_spotify_token()
        print(f"Spotify token: OK")

    all_records = []
    total_countries = len(COUNTRIES)

    for i, (code, (name, region)) in enumerate(COUNTRIES.items(), 1):
        print(f"[{i:2d}/{total_countries}] {name:20s} ({code}) ...", end=" ", flush=True)

        # Try Spotify charts API first, then CSV fallback
        entries = []
        if token:
            entries = fetch_chart_spotify(code, token)
        if not entries:
            entries = fetch_chart_csv_fallback(code)

        if not entries:
            print("no data")
            continue

        # Enrich with audio features
        track_ids = [e["track_id"] for e in entries]
        audio_features = {}
        if token and track_ids:
            audio_features = get_audio_features(track_ids, token)

        # Build records
        for entry in entries:
            tid = entry["track_id"].replace("spotify:track:", "")
            feat = audio_features.get(tid, {})
            record = {
                **entry,
                "country":      code,
                "country_name": name,
                "region":       region,
                "week":         week,
                "scraped_at":   now,
                **feat,
            }
            all_records.append(record)

        print(f"{len(entries)} songs")
        time.sleep(0.5)

    # Save
    with open(out_file, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_records)} chart entries → {out_file}")
    return all_records


# ── Viral score computation ───────────────────────────────────────────────────

def compute_viral_scores(records: list[dict]) -> list[dict]:
    """
    Score each song by how globally viral it is.
    Songs appearing in more countries with higher rank = higher viral score.
    """
    from collections import defaultdict

    song_stats: dict[str, dict] = defaultdict(lambda: {
        "countries": [],
        "ranks": [],
        "streams": 0,
        "title": "",
        "artist": "",
        "audio": {},
    })

    for r in records:
        key = r["title"].lower().strip() + "::" + r["artist"].lower().strip()
        song_stats[key]["countries"].append(r["country"])
        song_stats[key]["ranks"].append(r["rank"])
        song_stats[key]["streams"] += r.get("streams", 0)
        song_stats[key]["title"]  = r["title"]
        song_stats[key]["artist"] = r["artist"]
        for feat in ["bpm","danceability","energy","valence","speechiness"]:
            if feat in r:
                song_stats[key]["audio"][feat] = r[feat]

    scored = []
    for key, stats in song_stats.items():
        n_countries  = len(set(stats["countries"]))
        avg_rank     = sum(stats["ranks"]) / len(stats["ranks"])
        rank_score   = max(0, (101 - avg_rank) / 100)   # 1.0 = rank 1
        global_score = round(n_countries * rank_score, 3)

        scored.append({
            "key":         key,
            "title":       stats["title"],
            "artist":      stats["artist"],
            "n_countries": n_countries,
            "avg_rank":    round(avg_rank, 1),
            "total_streams": stats["streams"],
            "viral_score": global_score,
            "countries":   sorted(set(stats["countries"])),
            **stats["audio"],
        })

    scored.sort(key=lambda x: x["viral_score"], reverse=True)

    # Save viral scores
    viral_file = Path("data/charts") / f"viral_scores_{datetime.datetime.utcnow().strftime('%Y-%W')}.jsonl"
    with open(viral_file, "w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Viral scores saved → {viral_file}")
    print(f"\nTop 10 globally viral songs this week:")
    for s in scored[:10]:
        print(f"  [{s['viral_score']:5.1f}] {s['title']} — {s['artist']} ({s['n_countries']} countries, avg rank {s['avg_rank']})")

    return scored


# ── Weekly scheduler ─────────────────────────────────────────────────────────

def run_weekly():
    """Run every week automatically — saves to data/charts/YYYY-WW.jsonl"""
    import signal

    print("=" * 60)
    print(f"  Global Chart Scraper — {len(COUNTRIES)} countries x Top 100")
    print("=" * 60)

    running = True
    def _stop(sig, frame):
        nonlocal running
        print("\nStopping after this cycle...")
        running = False
    signal.signal(signal.SIGINT, _stop)

    while running:
        week = datetime.datetime.utcnow().strftime("%Y-%W")
        week_file = Path("data/charts") / f"{week}.jsonl"

        if week_file.exists():
            print(f"Week {week} already scraped — waiting for next week...")
        else:
            print(f"\nScraping week {week}...")
            records = scrape_all_charts()
            if records:
                compute_viral_scores(records)

        if not running:
            break

        # Sleep until next Monday
        now = datetime.datetime.utcnow()
        days_until_monday = (7 - now.weekday()) % 7 or 7
        next_run = now + datetime.timedelta(days=days_until_monday)
        next_run = next_run.replace(hour=6, minute=0, second=0)
        wait_sec = (next_run - now).total_seconds()
        print(f"\nNext scrape: {next_run.strftime('%Y-%m-%d %H:%M')} UTC ({wait_sec/3600:.1f}h)")
        time.sleep(wait_sec)


if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        records = scrape_all_charts()
        compute_viral_scores(records)
    else:
        run_weekly()

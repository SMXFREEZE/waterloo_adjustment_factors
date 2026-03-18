"""
Global chart scraper — 55 countries x Top 100 = 5,500 songs/week.

NO API KEY NEEDED — uses free public sources:
  1. Billboard charts    (no key needed — hot-100, global 200, + genre charts)
  2. Deezer API          (free, no auth — dominates Africa/Morocco/France/LatAm)
  3. charts.spotify.com  (public CSV download, no login)
  4. Last.fm API         (free, no premium — get key at last.fm/api)
  5. iTunes RSS          (100% free, no key needed)

Output: data/charts/YYYY-WW.jsonl
"""

import csv
import io
import json
import os
import time
import datetime
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY", "")   # free at last.fm/api
CHARTS_DIR     = Path("data/charts")

# ── 55 Countries ──────────────────────────────────────────────────────────────

COUNTRIES = {
    # Africa
    "NG": ("Nigeria",            "Africa"),
    "ZA": ("South Africa",       "Africa"),
    "MA": ("Morocco",            "Africa"),
    "EG": ("Egypt",              "Africa"),
    "KE": ("Kenya",              "Africa"),
    "GH": ("Ghana",              "Africa"),
    "SN": ("Senegal",            "Africa"),
    "TZ": ("Tanzania",           "Africa"),
    "ET": ("Ethiopia",           "Africa"),
    "CI": ("Ivory Coast",        "Africa"),
    # Middle East
    "SA": ("Saudi Arabia",       "Middle East"),
    "AE": ("UAE",                "Middle East"),
    "LB": ("Lebanon",            "Middle East"),
    "TR": ("Turkey",             "Middle East"),
    "IL": ("Israel",             "Middle East"),
    "PK": ("Pakistan",           "Middle East"),
    "IQ": ("Iraq",               "Middle East"),
    # Asia
    "KR": ("South Korea",        "Asia"),
    "JP": ("Japan",              "Asia"),
    "IN": ("India",              "Asia"),
    "PH": ("Philippines",        "Asia"),
    "ID": ("Indonesia",          "Asia"),
    "TH": ("Thailand",           "Asia"),
    "VN": ("Vietnam",            "Asia"),
    "MY": ("Malaysia",           "Asia"),
    "BD": ("Bangladesh",         "Asia"),
    "TW": ("Taiwan",             "Asia"),
    # Latin America
    "MX": ("Mexico",             "Latin America"),
    "BR": ("Brazil",             "Latin America"),
    "CO": ("Colombia",           "Latin America"),
    "AR": ("Argentina",          "Latin America"),
    "CL": ("Chile",              "Latin America"),
    "PE": ("Peru",               "Latin America"),
    "DO": ("Dominican Republic", "Latin America"),
    "PR": ("Puerto Rico",        "Latin America"),
    "VE": ("Venezuela",          "Latin America"),
    "EC": ("Ecuador",            "Latin America"),
    # Europe
    "GB": ("UK",                 "Europe"),
    "FR": ("France",             "Europe"),
    "DE": ("Germany",            "Europe"),
    "ES": ("Spain",              "Europe"),
    "IT": ("Italy",              "Europe"),
    "PT": ("Portugal",           "Europe"),
    "NL": ("Netherlands",        "Europe"),
    "SE": ("Sweden",             "Europe"),
    "PL": ("Poland",             "Europe"),
    "RO": ("Romania",            "Europe"),
    "GR": ("Greece",             "Europe"),
    "NO": ("Norway",             "Europe"),
    # Americas + Oceania
    "US": ("USA",                "Americas"),
    "CA": ("Canada",             "Americas"),
    "AU": ("Australia",          "Oceania"),
    "NZ": ("New Zealand",        "Oceania"),
    "JM": ("Jamaica",            "Americas"),
}

# Countries supported by Spotify Charts public CSV
SPOTIFY_CHART_COUNTRIES = {
    "US","GB","CA","AU","DE","FR","ES","IT","PT","NL","SE","NO","PL","AR",
    "MX","BR","CO","CL","PE","DO","EC","TR","ZA","NG","KE","GH","EG","MA",
    "SA","AE","IN","ID","PH","MY","TH","VN","KR","JP","TW","IL","NZ","JM",
}

# Countries with iTunes charts
ITUNES_COUNTRY_MAP = {
    "US":"us","GB":"gb","CA":"ca","AU":"au","DE":"de","FR":"fr","ES":"es",
    "IT":"it","JP":"jp","KR":"kr","BR":"br","MX":"mx","IN":"in","TR":"tr",
    "SA":"sa","AE":"ae","MA":"ma","NG":"ng","ZA":"za","EG":"eg","ID":"id",
    "PH":"ph","TH":"th","VN":"vn","MY":"my","TW":"tw","AR":"ar","CO":"co",
    "CL":"cl","NO":"no","SE":"se","NL":"nl","PL":"pl","RO":"ro","GR":"gr",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ── Source 1: Billboard (no API key needed) ───────────────────────────────────

# Billboard charts we pull — covers every major genre globally
BILLBOARD_CHARTS = [
    ("hot-100",                   "US",     "Americas", "pop/hip_hop/rnb"),
    ("billboard-global-200",      "GLOBAL", "Global",   "global"),
    ("billboard-global-excl-us",  "GLOBAL", "Global",   "global_excl_us"),
    ("rap-song",                  "US",     "Americas", "hip_hop/rap"),
    ("hot-rnb-hip-hop-songs",     "US",     "Americas", "rnb/hip_hop"),
    ("hot-country-songs",         "US",     "Americas", "country"),
    ("hot-rock-songs",            "US",     "Americas", "rock"),
    ("hot-latin-songs",           "LATIN",  "Latin America", "latin"),
    ("tropical-songs",            "LATIN",  "Latin America", "tropical"),
    ("reggaeton-songs",           "LATIN",  "Latin America", "reggaeton"),
    ("k-pop",                     "KR",     "Asia",     "kpop"),
    ("afrobeats",                 "AFRICA", "Africa",   "afrobeats"),
    ("dance-electronic-songs",    "US",     "Global",   "edm"),
    ("pop-songs",                 "US",     "Americas", "pop"),
    ("adult-contemporary",        "US",     "Americas", "pop"),
    ("uk-songs",                  "GB",     "Europe",   "uk"),
    ("france-songs",              "FR",     "Europe",   "french"),
    ("germany-songs",             "DE",     "Europe",   "german"),
    ("italy-songs",               "IT",     "Europe",   "italian"),
    ("spain-songs",               "ES",     "Europe",   "spanish"),
    ("brazil-songs",              "BR",     "Latin America", "brazilian"),
    ("mexico-songs",              "MX",     "Latin America", "mexican"),
    ("colombia-songs",            "CO",     "Latin America", "colombian"),
    ("argentina-songs",           "AR",     "Latin America", "argentinian"),
    ("japan-songs",               "JP",     "Asia",     "jpop"),
    ("philippines-songs",         "PH",     "Asia",     "opm"),
    ("india-songs",               "IN",     "Asia",     "indian"),
]

def fetch_billboard_all() -> list[dict]:
    """
    Fetch all Billboard charts — no API key, no login.
    Returns combined list of records tagged by country/region/genre.
    """
    try:
        import billboard
    except ImportError:
        print("  billboard.py not installed — run: pip install billboard.py")
        return []

    now = datetime.datetime.utcnow().isoformat()
    week = datetime.datetime.utcnow().strftime("%Y-%W")
    all_records = []

    for chart_name, country, region, genre_tag in BILLBOARD_CHARTS:
        try:
            chart = billboard.ChartData(chart_name)
            country_name = COUNTRIES.get(country, (country, region))[0] if country in COUNTRIES else country
            for entry in chart[:100]:
                all_records.append({
                    "rank":           entry.rank,
                    "title":          entry.title,
                    "artist":         entry.artist,
                    "track_id":       f"billboard:{chart_name}:{entry.rank}",
                    "streams":        0,
                    "peak_rank":      entry.peakPos or entry.rank,
                    "weeks_on_chart": entry.weeks or 1,
                    "country":        country,
                    "country_name":   country_name,
                    "region":         region,
                    "genre_tag":      genre_tag,
                    "chart_name":     chart_name,
                    "week":           week,
                    "scraped_at":     now,
                    "source":         "billboard",
                })
            print(f"  Billboard [{chart_name}]: {len(chart[:100])} songs")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Billboard [{chart_name}] error: {e}")

    return all_records


# ── Source 2: Spotify Charts public CSV (no API key) ─────────────────────────

def fetch_spotify_charts_csv(country_code: str) -> list[dict]:
    """
    Download the public weekly chart CSV from charts.spotify.com.
    No login or API key required.
    """
    url = f"https://charts.spotify.com/charts/view/regional-{country_code.lower()}-weekly/latest"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        # The page returns CSV directly for some regions
        text = r.text
        if "rank" not in text.lower():
            return []
        lines = text.splitlines()
        # Find the header line
        header_idx = next((i for i, l in enumerate(lines) if "rank" in l.lower()), None)
        if header_idx is None:
            return []
        reader = csv.DictReader(lines[header_idx:])
        results = []
        for i, row in enumerate(reader):
            if i >= 100:
                break
            try:
                results.append({
                    "rank":           int(row.get("rank", i + 1)),
                    "title":          row.get("track_name", row.get("Track Name", "")),
                    "artist":         row.get("artist_names", row.get("Artist", "")),
                    "track_id":       row.get("uri", row.get("URL", "")),
                    "streams":        int(row.get("streams", 0)) if str(row.get("streams","")).isdigit() else 0,
                    "peak_rank":      int(row.get("peak_rank", i + 1)),
                    "weeks_on_chart": int(row.get("weeks_on_chart", 1)),
                    "source":         "spotify_charts",
                })
            except Exception:
                continue
        return results
    except Exception as e:
        return []


# ── Source 2: Last.fm API (free, no premium) ──────────────────────────────────

LASTFM_COUNTRY_MAP = {
    "US": "United States", "GB": "United Kingdom", "CA": "Canada",
    "AU": "Australia",     "DE": "Germany",        "FR": "France",
    "ES": "Spain",         "IT": "Italy",          "JP": "Japan",
    "BR": "Brazil",        "MX": "Mexico",         "TR": "Turkey",
    "PL": "Poland",        "NL": "Netherlands",    "SE": "Sweden",
    "NO": "Norway",        "AR": "Argentina",      "CO": "Colombia",
    "IN": "India",         "RO": "Romania",        "GR": "Greece",
}

def fetch_lastfm_charts(country_code: str) -> list[dict]:
    """Fetch top tracks by country from Last.fm (free API, no premium)."""
    if not LASTFM_API_KEY:
        return []
    country_name = LASTFM_COUNTRY_MAP.get(country_code)
    if not country_name:
        return []
    try:
        r = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method":  "geo.gettoptracks",
                "country": country_name,
                "api_key": LASTFM_API_KEY,
                "format":  "json",
                "limit":   100,
            },
            headers=HEADERS,
            timeout=10,
        )
        if r.status_code != 200:
            return []
        tracks = r.json().get("tracks", {}).get("track", [])
        results = []
        for i, t in enumerate(tracks[:100]):
            results.append({
                "rank":           i + 1,
                "title":          t.get("name", ""),
                "artist":         t.get("artist", {}).get("name", ""),
                "track_id":       t.get("url", ""),
                "streams":        int(t.get("listeners", 0)),
                "peak_rank":      i + 1,
                "weeks_on_chart": 1,
                "source":         "lastfm",
            })
        return results
    except Exception:
        return []


# ── Source 3: iTunes RSS (100% free, no key needed) ──────────────────────────

def fetch_itunes_charts(country_code: str) -> list[dict]:
    """Fetch top songs from iTunes RSS feed — completely free, no API key."""
    itunes_code = ITUNES_COUNTRY_MAP.get(country_code)
    if not itunes_code:
        return []
    url = f"https://itunes.apple.com/{itunes_code}/rss/topsongs/limit=100/json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        feed = r.json().get("feed", {}).get("entry", [])
        results = []
        for i, entry in enumerate(feed[:100]):
            title  = entry.get("im:name", {}).get("label", "")
            artist = entry.get("im:artist", {}).get("label", "")
            results.append({
                "rank":           i + 1,
                "title":          title,
                "artist":         artist,
                "track_id":       entry.get("id", {}).get("label", ""),
                "streams":        0,
                "peak_rank":      i + 1,
                "weeks_on_chart": 1,
                "source":         "itunes",
            })
        return results
    except Exception:
        return []


# ── Source 4: Deezer API (free, no auth, strong in Africa/France/LatAm) ──────

# Deezer country IDs — strongest where Spotify is weak (Morocco, Africa, France)
DEEZER_COUNTRY_IDS = {
    "FR": 23,   "MA": 216,  "DZ": 6,    "TN": 216,  "SN": 216,
    "CI": 216,  "CM": 216,  "ML": 216,  "BF": 216,
    "BR": 18,   "MX": 152,  "CO": 49,   "AR": 11,   "CL": 46,
    "PE": 174,  "VE": 236,  "EC": 63,   "DO": 61,
    "BE": 16,   "CH": 226,  "DE": 74,   "ES": 67,   "IT": 107,
    "PT": 178,  "NL": 159,  "PL": 178,  "RO": 185,
    "NG": 166,  "ZA": 205,  "KE": 119,  "GH": 81,   "EG": 64,
    "TR": 229,  "SA": 194,  "AE": 231,  "LB": 130,
    "PH": 175,  "ID": 100,  "MY": 145,  "TH": 222,  "VN": 238,
    "US": 232,  "GB": 78,   "CA": 37,   "AU": 13,
}

def fetch_deezer_charts(country_code: str) -> list[dict]:
    """
    Fetch top tracks from Deezer API — completely free, no API key, no auth.
    Especially good for Morocco, francophone Africa, France, LatAm.
    API: https://api.deezer.com/chart/{country_id}/tracks
    """
    country_id = DEEZER_COUNTRY_IDS.get(country_code)
    if not country_id:
        return []
    try:
        r = requests.get(
            f"https://api.deezer.com/chart/{country_id}/tracks",
            params={"limit": 100},
            headers=HEADERS,
            timeout=10,
        )
        if r.status_code != 200:
            return []
        tracks = r.json().get("data", [])
        results = []
        for i, t in enumerate(tracks[:100]):
            results.append({
                "rank":           i + 1,
                "title":          t.get("title", ""),
                "artist":         t.get("artist", {}).get("name", ""),
                "track_id":       f"deezer:{t.get('id','')}",
                "streams":        t.get("rank", 0),
                "peak_rank":      i + 1,
                "weeks_on_chart": 1,
                "duration_ms":    t.get("duration", 0) * 1000,
                "source":         "deezer",
            })
        return results
    except Exception:
        return []


# ── Source 5: Last.fm global top tracks (fallback for any country) ────────────

def fetch_lastfm_global() -> list[dict]:
    """Fetch global top 100 from Last.fm — works without country."""
    if not LASTFM_API_KEY:
        return []
    try:
        r = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method":  "chart.gettoptracks",
                "api_key": LASTFM_API_KEY,
                "format":  "json",
                "limit":   100,
            },
            headers=HEADERS,
            timeout=10,
        )
        tracks = r.json().get("tracks", {}).get("track", [])
        return [
            {
                "rank":    i + 1,
                "title":   t.get("name", ""),
                "artist":  t.get("artist", {}).get("name", ""),
                "track_id": t.get("url", ""),
                "streams": int(t.get("playcount", 0)),
                "peak_rank": i + 1,
                "weeks_on_chart": 1,
                "source": "lastfm_global",
            }
            for i, t in enumerate(tracks[:100])
        ]
    except Exception:
        return []


# ── Smart fetch: tries all sources ────────────────────────────────────────────

def fetch_chart(country_code: str) -> list[dict]:
    """
    Try sources in order: Deezer → Spotify CSV → Last.fm → iTunes → skip
    Returns the first source that gives data.
    Deezer first for Africa/Morocco/France/LatAm where it dominates.
    """
    # 1. Deezer (free, no auth — best for niche markets)
    entries = fetch_deezer_charts(country_code)
    if entries:
        return entries
    time.sleep(0.3)

    # 2. Spotify Charts CSV
    if country_code in SPOTIFY_CHART_COUNTRIES:
        entries = fetch_spotify_charts_csv(country_code)
        if entries:
            return entries
        time.sleep(0.3)

    # 3. Last.fm by country
    entries = fetch_lastfm_charts(country_code)
    if entries:
        return entries

    # 4. iTunes RSS
    entries = fetch_itunes_charts(country_code)
    if entries:
        return entries

    return []


# ── Main scrape ────────────────────────────────────────────────────────────────

def scrape_all_charts(out_dir: str = "data/charts") -> list[dict]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    week    = datetime.datetime.utcnow().strftime("%Y-%W")
    now     = datetime.datetime.utcnow().isoformat()
    out_file = out_path / f"{week}.jsonl"

    if not LASTFM_API_KEY:
        print("TIP: Set LASTFM_API_KEY in .env for better coverage (free at last.fm/api)")

    all_records = []

    # ── Billboard first (covers 25+ genre charts globally) ────────────────────
    print("\n=== Billboard Charts ===")
    billboard_records = fetch_billboard_all()
    all_records.extend(billboard_records)
    print(f"Billboard total: {len(billboard_records)} entries across {len(BILLBOARD_CHARTS)} charts")

    # ── Per-country scrape ─────────────────────────────────────────────────────
    print("\n=== Country Charts (Spotify CSV + Last.fm + iTunes) ===")
    total = len(COUNTRIES)

    for i, (code, (name, region)) in enumerate(COUNTRIES.items(), 1):
        print(f"[{i:2d}/{total}] {name:22s} ({code}) ...", end=" ", flush=True)
        entries = fetch_chart(code)

        if not entries:
            print("no data")
            continue

        for entry in entries:
            all_records.append({
                **entry,
                "country":      code,
                "country_name": name,
                "region":       region,
                "week":         week,
                "scraped_at":   now,
            })

        sources = set(e.get("source","?") for e in entries)
        print(f"{len(entries)} songs ({', '.join(sources)})")
        time.sleep(0.5)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_records)} chart entries → {out_file}")
    return all_records


# ── Viral scores ───────────────────────────────────────────────────────────────

def compute_viral_scores(records: list[dict]) -> list[dict]:
    # key → {countries: list, ranks: list, streams: int, title: str, artist: str}
    song_stats: dict[str, dict[str, list | int | str]] = {}

    for r in records:
        key = r["title"].lower().strip() + "::" + r["artist"].lower().strip()
        if key not in song_stats:
            song_stats[key] = {
                "countries": [],
                "ranks":     [],
                "streams":   0,
                "title":     r["title"],
                "artist":    r["artist"],
            }
        countries = song_stats[key]["countries"]
        ranks     = song_stats[key]["ranks"]
        if isinstance(countries, list):
            countries.append(r["country"])
        if isinstance(ranks, list):
            ranks.append(r["rank"])
        streams = song_stats[key]["streams"]
        song_stats[key]["streams"] = (streams if isinstance(streams, int) else 0) + r.get("streams", 0)
        song_stats[key]["title"]   = r["title"]
        song_stats[key]["artist"]  = r["artist"]

    scored = []
    for key, stats in song_stats.items():
        countries_list = stats["countries"] if isinstance(stats["countries"], list) else []
        ranks_list     = stats["ranks"]     if isinstance(stats["ranks"],     list) else []
        streams_val    = stats["streams"]   if isinstance(stats["streams"],   int)  else 0
        n_countries = len(set(countries_list))
        avg_rank    = sum(ranks_list) / max(len(ranks_list), 1)
        rank_score  = max(0.0, (101 - avg_rank) / 100)
        viral_score = round(n_countries * rank_score, 3)
        scored.append({
            "key":           key,
            "title":         stats["title"],
            "artist":        stats["artist"],
            "n_countries":   n_countries,
            "avg_rank":      round(avg_rank, 1),
            "total_streams": streams_val,
            "viral_score":   viral_score,
            "countries":     sorted(set(countries_list)),
        })

    scored.sort(key=lambda x: x["viral_score"], reverse=True)

    viral_file = Path("data/charts") / f"viral_scores_{datetime.datetime.utcnow().strftime('%Y-%W')}.jsonl"
    viral_file.parent.mkdir(parents=True, exist_ok=True)
    with open(viral_file, "w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nTop 10 viral songs this week:")
    for s in scored[:10]:
        print(f"  [{s['viral_score']:5.1f}] {s['title']} — {s['artist']} ({s['n_countries']} countries)")

    return scored


# ── Weekly scheduler ──────────────────────────────────────────────────────────

def run_weekly():
    import signal
    print(f"Global Chart Scraper — {len(COUNTRIES)} countries x Top 100")
    print("Sources: Billboard + Spotify Charts CSV + Last.fm + iTunes (all free)")
    print("=" * 60)

    running = True
    def _stop(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _stop)

    while running:
        week      = datetime.datetime.utcnow().strftime("%Y-%W")
        week_file = Path("data/charts") / f"{week}.jsonl"

        if not week_file.exists():
            print(f"\nScraping week {week}...")
            records = scrape_all_charts()
            if records:
                compute_viral_scores(records)

        if not running:
            break

        now = datetime.datetime.utcnow()
        days_until_monday = (7 - now.weekday()) % 7 or 7
        next_run  = (now + datetime.timedelta(days=days_until_monday)).replace(hour=6, minute=0, second=0)
        wait_sec  = (next_run - now).total_seconds()
        print(f"\nNext scrape: {next_run.strftime('%Y-%m-%d %H:%M')} UTC ({wait_sec/3600:.1f}h away)")
        time.sleep(wait_sec)


if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        records = scrape_all_charts()
        compute_viral_scores(records)
    else:
        run_weekly()

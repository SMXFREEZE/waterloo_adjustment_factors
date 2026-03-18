"""
Infinite Genius crawler + HuggingFace dataset pusher.

Flow:
  seed artists → scrape songs → annotate → save locally → push to HF
  → discover similar artists from Genius → add to queue → repeat forever

Usage:
  python scripts/infinite_crawl.py

Env vars required:
  GENIUS_TOKEN          — from genius.com/api-clients
  HUGGING_FACE_HUB_TOKEN — from huggingface.co/settings/tokens
  HF_DATASET_REPO       — e.g. "yourname/god-tier-lyrics" (created automatically)
"""

import json
import os
import sys
import time
import random
import re
import signal
from collections import deque
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────

GENIUS_TOKEN      = os.getenv("GENIUS_TOKEN", "")
HF_TOKEN          = os.getenv("HUGGING_FACE_HUB_TOKEN", "")
HF_DATASET_REPO   = os.getenv("HF_DATASET_REPO", "")

DATA_DIR          = Path("data/crawl")
SEEN_FILE         = DATA_DIR / "seen_artists.txt"
SEEN_SONGS_FILE   = DATA_DIR / "seen_songs.txt"
RAW_JSONL         = DATA_DIR / "songs.jsonl"
ANNOTATED_JSONL   = DATA_DIR / "annotated.jsonl"

PUSH_EVERY_N      = int(os.getenv("PUSH_EVERY_N", "200"))   # push to HF every N songs
MAX_SONGS_ARTIST  = int(os.getenv("MAX_SONGS_ARTIST", "50"))
SLEEP_BETWEEN     = float(os.getenv("SLEEP_BETWEEN", "1.2")) # seconds between requests
ANNOTATE          = os.getenv("ANNOTATE", "1") == "1"        # set to 0 to skip annotation

console = Console()

# ── Genre seed artists ─────────────────────────────────────────────────────────

GENRE_SEEDS: dict[str, list[str]] = {
    "trap":    ["Future", "Young Thug", "Gunna", "Lil Baby", "21 Savage", "Roddy Ricch", "Offset", "Quavo", "Takeoff", "Playboi Carti"],
    "rnb":     ["Frank Ocean", "SZA", "H.E.R.", "Bryson Tiller", "Summer Walker", "The Weeknd", "Jhene Aiko", "Daniel Caesar", "Brent Faiyaz"],
    "indie":   ["Phoebe Bridgers", "boygenius", "Sufjan Stevens", "Bon Iver", "Mitski", "Clairo", "Soccer Mommy", "Hand Habits"],
    "pop":     ["Taylor Swift", "Olivia Rodrigo", "Dua Lipa", "Ariana Grande", "Harry Styles", "Sabrina Carpenter", "Charli XCX"],
    "drill":   ["Pop Smoke", "Fivio Foreign", "Lil Durk", "King Von", "Central Cee", "Digga D", "Headie One", "Dave"],
    "alt_emo": ["Paramore", "My Chemical Romance", "Lana Del Rey", "Billie Eilish", "Halsey", "Gracie Abrams", "Conan Gray"],
    "hip_hop": ["Kendrick Lamar", "J. Cole", "Drake", "Jay-Z", "Nas", "Lupe Fiasco", "Common", "Mos Def", "Black Thought"],
    "country": ["Morgan Wallen", "Luke Combs", "Zach Bryan", "Kacey Musgraves", "Tyler Childers", "Chris Stapleton"],
    "rock":    ["Arctic Monkeys", "The Strokes", "Tame Impala", "Radiohead", "Foo Fighters", "Interpol", "The National"],
    "latin":   ["Bad Bunny", "J Balvin", "Ozuna", "Karol G", "Peso Pluma", "Rauw Alejandro", "Myke Towers"],
}

# ── Persistence helpers ────────────────────────────────────────────────────────

def load_set(path: Path) -> set[str]:
    if path.exists():
        return set(path.read_text(encoding="utf-8").splitlines())
    return set()

def append_line(path: Path, value: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(value + "\n")

# ── Genius API helpers ─────────────────────────────────────────────────────────

GENIUS_HEADERS = {"Authorization": f"Bearer {GENIUS_TOKEN}"}
GENIUS_BASE    = "https://api.genius.com"

def genius_get(endpoint: str, params: dict = {}) -> dict | None:
    try:
        r = requests.get(
            f"{GENIUS_BASE}{endpoint}",
            headers=GENIUS_HEADERS,
            params=params,
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("response", {})
        console.print(f"[yellow]Genius {r.status_code}: {endpoint}[/yellow]")
    except Exception as e:
        console.print(f"[red]Request error: {e}[/red]")
    return None

def search_artist_id(name: str) -> int | None:
    data = genius_get("/search", {"q": name})
    if not data:
        return None
    for hit in data.get("hits", []):
        artist = hit.get("result", {}).get("primary_artist", {})
        if artist.get("name", "").lower() == name.lower():
            return artist["id"]
    # Fallback: first hit's artist
    hits = data.get("hits", [])
    if hits:
        return hits[0]["result"]["primary_artist"]["id"]
    return None

def get_artist_songs(artist_id: int, max_songs: int = 50) -> list[dict]:
    songs = []
    page = 1
    while len(songs) < max_songs:
        data = genius_get(f"/artists/{artist_id}/songs", {
            "per_page": min(50, max_songs - len(songs)),
            "page": page,
            "sort": "popularity",
        })
        if not data:
            break
        batch = data.get("songs", [])
        if not batch:
            break
        songs.extend(batch)
        if not data.get("next_page"):
            break
        page += 1
        time.sleep(SLEEP_BETWEEN)
    return songs[:max_songs]

def get_song_lyrics(song_id: int) -> str | None:
    """Fetch raw lyrics via lyricsgenius (handles scraping the HTML)."""
    try:
        import lyricsgenius
        genius = lyricsgenius.Genius(GENIUS_TOKEN, verbose=False, remove_section_headers=True, skip_non_songs=True)
        song = genius.song(song_id)
        if song and song.lyrics:
            return clean_lyrics(song.lyrics)
    except Exception as e:
        pass
    return None

def get_similar_artist_names(artist_id: int) -> list[str]:
    """Discover related artists to expand the queue."""
    data = genius_get(f"/artists/{artist_id}/songs", {"per_page": 20, "sort": "popularity"})
    if not data:
        return []
    names = set()
    for song in data.get("songs", []):
        for feat in song.get("featured_artists", []):
            names.add(feat.get("name", ""))
        producer = song.get("producer_artists", [])
        for p in producer:
            names.add(p.get("name", ""))
    return [n for n in names if n]

def clean_lyrics(raw: str) -> str:
    text = re.sub(r"\[.*?\]", "", raw)
    text = re.sub(r"\d+ Contributors?.*?Lyrics", "", text, flags=re.DOTALL)
    text = re.sub(r"\d*Embed$", "", text.strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ── Annotation ─────────────────────────────────────────────────────────────────

def annotate_song(record: dict) -> dict:
    """Add phoneme/rhyme/valence annotations to a song record."""
    try:
        from src.data.phoneme_annotator import annotate_line, annotation_to_dict
        from src.data.rhyme_labeler import detect_scheme
        from src.data.valence_scorer import score_lyrics

        lines = [l.strip() for l in record["lyrics"].splitlines() if l.strip()]
        if not lines:
            return record

        # Sample first 32 lines for speed
        sample = lines[:32]
        scheme = detect_scheme(sample)
        emotions = score_lyrics("\n".join(sample))
        avg_val = sum(e.valence for e in emotions) / max(len(emotions), 1)
        avg_aro = sum(e.arousal for e in emotions) / max(len(emotions), 1)

        record["annotation"] = {
            "line_count":    len(lines),
            "scheme_type":   scheme["scheme_type"],
            "rhyme_density": scheme["rhyme_density"],
            "avg_valence":   round(avg_val, 4),
            "avg_arousal":   round(avg_aro, 4),
        }
    except Exception as e:
        record["annotation"] = {"error": str(e)}
    return record

# ── HuggingFace push ───────────────────────────────────────────────────────────

def push_to_hf(records: list[dict], repo_id: str):
    """Push a batch of records to a HuggingFace dataset repo."""
    if not HF_TOKEN or not repo_id:
        console.print("[yellow]Skipping HF push (no token or repo set)[/yellow]")
        return
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
        import pandas as pd

        api = HfApi(token=HF_TOKEN)
        # Ensure repo exists
        try:
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
        except Exception:
            pass

        df = pd.DataFrame(records)
        ds = Dataset.from_pandas(df)
        ds.push_to_hub(repo_id, token=HF_TOKEN, split="train")
        console.print(f"[green]Pushed {len(records)} songs → HF dataset: {repo_id}[/green]")
    except Exception as e:
        console.print(f"[red]HF push failed: {e}[/red]")

# ── Stats display ──────────────────────────────────────────────────────────────

class Stats:
    def __init__(self):
        self.total_songs   = 0
        self.total_artists = 0
        self.queue_size    = 0
        self.last_artist   = ""
        self.last_song     = ""
        self.since         = datetime.now()
        self.errors        = 0
        self.pushed_to_hf  = 0

    def render(self) -> Table:
        elapsed = (datetime.now() - self.since).seconds
        rate = self.total_songs / max(elapsed, 1) * 60
        t = Table(title="God-Tier Lyrics Crawler", show_header=False, border_style="cyan")
        t.add_column("Key",   style="bold cyan", min_width=20)
        t.add_column("Value", style="white")
        t.add_row("Songs collected",  str(self.total_songs))
        t.add_row("Artists scraped",  str(self.total_artists))
        t.add_row("Queue remaining",  str(self.queue_size))
        t.add_row("Rate",             f"{rate:.1f} songs/min")
        t.add_row("Pushed to HF",     str(self.pushed_to_hf))
        t.add_row("Errors",           str(self.errors))
        t.add_row("Last artist",      self.last_artist[:40])
        t.add_row("Last song",        self.last_song[:40])
        t.add_row("Running since",    self.since.strftime("%H:%M:%S"))
        return t

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    if not GENIUS_TOKEN:
        console.print("[red bold]GENIUS_TOKEN not set. Add it to your .env file.[/red bold]")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    seen_artists = load_set(SEEN_FILE)
    seen_songs   = load_set(SEEN_SONGS_FILE)

    # Build initial queue from seeds, shuffle so genres mix
    all_seeds = [(name, genre) for genre, names in GENRE_SEEDS.items() for name in names]
    random.shuffle(all_seeds)
    queue: deque[tuple[str, str]] = deque(
        (name, genre) for name, genre in all_seeds if name not in seen_artists
    )

    stats = Stats()
    buffer: list[dict] = []   # accumulate before HF push

    # Graceful shutdown
    running = True
    def _stop(sig, frame):
        nonlocal running
        console.print("\n[yellow]Stopping gracefully...[/yellow]")
        running = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    console.print(f"[cyan bold]Starting infinite crawl — {len(queue)} artists in initial queue[/cyan bold]")
    if HF_TOKEN and HF_DATASET_REPO:
        console.print(f"[green]Will push to HF dataset: {HF_DATASET_REPO} every {PUSH_EVERY_N} songs[/green]")
    else:
        console.print("[yellow]HF push disabled (set HF_DATASET_REPO + HUGGING_FACE_HUB_TOKEN to enable)[/yellow]")

    with Live(stats.render(), refresh_per_second=2, console=console) as live:
        while running:
            if not queue:
                # Re-seed from the beginning if queue somehow empties
                console.print("[yellow]Queue empty — re-seeding[/yellow]")
                for name, genre in all_seeds:
                    queue.append((name, genre))
                time.sleep(5)

            artist_name, genre = queue.popleft()
            stats.queue_size = len(queue)

            if artist_name in seen_artists:
                continue

            # --- Find artist ID ---
            artist_id = search_artist_id(artist_name)
            if not artist_id:
                stats.errors += 1
                live.update(stats.render())
                continue

            seen_artists.add(artist_name)
            append_line(SEEN_FILE, artist_name)
            stats.total_artists += 1
            stats.last_artist = artist_name

            # --- Get song list ---
            songs_meta = get_artist_songs(artist_id, MAX_SONGS_ARTIST)

            for song_meta in songs_meta:
                if not running:
                    break

                song_id    = song_meta.get("id")
                song_title = song_meta.get("title", "")
                song_key   = f"{artist_name}::{song_title}"

                if song_key in seen_songs:
                    continue

                stats.last_song = song_title

                # Fetch lyrics
                lyrics = get_song_lyrics(song_id)
                if not lyrics or len(lyrics.split()) < 60:
                    seen_songs.add(song_key)
                    append_line(SEEN_SONGS_FILE, song_key)
                    continue

                record = {
                    "artist":     artist_name,
                    "title":      song_title,
                    "genre":      genre,
                    "lyrics":     lyrics,
                    "genius_id":  song_id,
                    "scraped_at": datetime.utcnow().isoformat(),
                }

                # Annotate
                if ANNOTATE:
                    record = annotate_song(record)

                # Save locally
                with open(RAW_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                seen_songs.add(song_key)
                append_line(SEEN_SONGS_FILE, song_key)
                buffer.append(record)
                stats.total_songs += 1
                live.update(stats.render())

                # Push to HF every N songs
                if HF_TOKEN and HF_DATASET_REPO and len(buffer) >= PUSH_EVERY_N:
                    push_to_hf(buffer, HF_DATASET_REPO)
                    stats.pushed_to_hf += len(buffer)
                    buffer = []

                time.sleep(SLEEP_BETWEEN)

            # --- Discover similar artists to expand queue ---
            new_artists = get_similar_artist_names(artist_id)
            added = 0
            for new_name in new_artists:
                if new_name and new_name not in seen_artists:
                    queue.append((new_name, genre))
                    added += 1
            stats.queue_size = len(queue)
            live.update(stats.render())

        # Final push on exit
        if buffer and HF_TOKEN and HF_DATASET_REPO:
            push_to_hf(buffer, HF_DATASET_REPO)

    console.print(f"\n[green bold]Done. {stats.total_songs} songs collected from {stats.total_artists} artists.[/green bold]")
    console.print(f"Data saved to: {RAW_JSONL.resolve()}")


if __name__ == "__main__":
    main()

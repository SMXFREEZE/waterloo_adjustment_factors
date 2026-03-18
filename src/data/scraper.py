"""
Genius lyrics scraper.
Uses the lyricsgenius library to pull songs by genre/artist.
Output: JSONL files with {artist, title, genre, lyrics} per song.
"""

import os
import json
import time
import re
from pathlib import Path
from typing import Optional

import lyricsgenius
from tqdm import tqdm


GENIUS_TOKEN = os.getenv("GENIUS_TOKEN", "")

# Curated seed artists per genre — add more as needed
GENRE_SEEDS: dict[str, list[str]] = {
    "trap":    ["Future", "Young Thug", "Gunna", "Lil Baby", "21 Savage", "Roddy Ricch"],
    "rnb":     ["Frank Ocean", "SZA", "H.E.R.", "Bryson Tiller", "Summer Walker", "The Weeknd"],
    "indie":   ["Phoebe Bridgers", "boygenius", "Sufjan Stevens", "Bon Iver", "Mitski", "Clairo"],
    "pop":     ["Taylor Swift", "Olivia Rodrigo", "Dua Lipa", "Ariana Grande", "Harry Styles"],
    "drill":   ["Pop Smoke", "Fivio Foreign", "Lil Durk", "King Von", "Central Cee", "Digga D"],
    "alt_emo": ["Paramore", "My Chemical Romance", "Lana Del Rey", "Billie Eilish", "Halsey"],
    "hip_hop": ["Kendrick Lamar", "J. Cole", "Drake", "Jay-Z", "Nas", "Lupe Fiasco"],
    "country": ["Morgan Wallen", "Luke Combs", "Zach Bryan", "Kacey Musgraves", "Tyler Childers"],
    "rock":    ["Arctic Monkeys", "The Strokes", "Tame Impala", "Radiohead", "Foo Fighters"],
    "latin":   ["Bad Bunny", "J Balvin", "Ozuna", "Karol G", "Peso Pluma"],
}


def clean_lyrics(raw: str) -> str:
    """Strip Genius annotations, headers, ads, and empty lines."""
    # Remove [Verse], [Chorus] etc. section headers
    text = re.sub(r"\[.*?\]", "", raw)
    # Remove contributor lines
    text = re.sub(r"\d+ Contributors?.*?Lyrics", "", text, flags=re.DOTALL)
    # Remove Embed at end
    text = re.sub(r"\d*Embed$", "", text.strip())
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def scrape_artist(
    genius: lyricsgenius.Genius,
    artist_name: str,
    genre: str,
    max_songs: int = 50,
    out_dir: Path = Path("data/raw"),
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        artist = genius.search_artist(artist_name, max_songs=max_songs, sort="popularity")
        if not artist:
            return results
        for song in artist.songs:
            if not song.lyrics or len(song.lyrics.split()) < 80:
                continue
            lyrics = clean_lyrics(song.lyrics)
            if len(lyrics.split()) < 80:
                continue
            record = {
                "artist": artist_name,
                "title": song.title,
                "genre": genre,
                "lyrics": lyrics,
            }
            results.append(record)
    except Exception as e:
        print(f"Error scraping {artist_name}: {e}")
    return results


def run_scrape(
    genres: Optional[list[str]] = None,
    max_songs_per_artist: int = 50,
    out_dir: str = "data/raw",
    token: str = GENIUS_TOKEN,
):
    if not token:
        raise ValueError("Set GENIUS_TOKEN environment variable")

    genius = lyricsgenius.Genius(token, verbose=False, remove_section_headers=True, skip_non_songs=True)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_genres = genres or list(GENRE_SEEDS.keys())
    all_records: list[dict] = []

    for genre in target_genres:
        artists = GENRE_SEEDS.get(genre, [])
        print(f"\n=== Genre: {genre} ({len(artists)} artists) ===")
        genre_records = []
        for artist_name in tqdm(artists, desc=genre):
            records = scrape_artist(genius, artist_name, genre, max_songs_per_artist, out_path)
            genre_records.extend(records)
            time.sleep(1)  # rate limit

        # Write per-genre JSONL
        genre_file = out_path / f"{genre}.jsonl"
        with open(genre_file, "w", encoding="utf-8") as f:
            for r in genre_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(genre_records)} songs → {genre_file}")
        all_records.extend(genre_records)

    # Write combined
    combined_file = out_path / "all_songs.jsonl"
    with open(combined_file, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nTotal: {len(all_records)} songs → {combined_file}")
    return all_records


if __name__ == "__main__":
    import typer
    typer.run(run_scrape)

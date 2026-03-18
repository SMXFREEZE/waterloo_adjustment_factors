"""
Artist style vector extractor.
Computes a 128-dim style fingerprint per artist from their lyrics corpus.

Features captured:
  - avg / std syllables per line
  - rhyme density (end + internal)
  - unique vocab ratio (type/token)
  - avg line length (chars)
  - metaphor cluster embedding (via sentence-transformers, reduced to 32d)
  - emotional valence distribution (mean, std, skew)
  - punctuation density

All features are normalized and concatenated into a 128-dim vector.
This vector is stored per artist and injected as conditioning at inference.
"""

import json
import math
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from src.data.phoneme_annotator import annotate_line
from src.data.rhyme_labeler import detect_scheme, internal_rhyme_density
from src.data.valence_scorer import score_lyrics


_embed_model: Optional[SentenceTransformer] = None

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


@dataclass
class StyleVector:
    artist: str
    vector: np.ndarray          # shape: (128,)
    song_count: int
    feature_names: list[str]    # for interpretability


def extract_style_features(songs: list[str]) -> np.ndarray:
    """
    songs: list of raw lyrics strings for one artist.
    Returns a flat feature vector (before PCA compression).
    """
    all_lines: list[str] = []
    for song in songs:
        all_lines += [l.strip() for l in song.splitlines() if l.strip()]

    if not all_lines:
        return np.zeros(128)

    # ── Syllable features ──────────────────────────────────────────
    syllable_counts = []
    for line in all_lines[:500]:  # cap for speed
        ann = annotate_line(line)
        syllable_counts.append(ann.total_syllables)
    syl_arr = np.array(syllable_counts, dtype=float)
    feat_syllables = [
        float(np.mean(syl_arr)),
        float(np.std(syl_arr)),
        float(np.median(syl_arr)),
    ]

    # ── Line length features ──────────────────────────────────────
    line_lengths = [len(l) for l in all_lines]
    feat_length = [
        float(np.mean(line_lengths)),
        float(np.std(line_lengths)),
    ]

    # ── Vocab features ────────────────────────────────────────────
    all_words = " ".join(all_lines).lower()
    all_words = re.sub(r"[^a-z\s'']", " ", all_words).split()
    vocab_size = len(set(all_words))
    token_count = max(len(all_words), 1)
    feat_vocab = [
        vocab_size / token_count,               # type/token ratio
        math.log1p(vocab_size),                 # log vocab richness
    ]

    # ── Rhyme features ────────────────────────────────────────────
    rhyme_densities = []
    internal_densities = []
    for i in range(0, min(len(all_lines), 200), 4):
        chunk = all_lines[i:i+4]
        if len(chunk) == 4:
            result = detect_scheme(chunk)
            rhyme_densities.append(result["rhyme_density"])
        internal_densities.append(internal_rhyme_density(all_lines[i]))
    feat_rhyme = [
        float(np.mean(rhyme_densities)) if rhyme_densities else 0.0,
        float(np.mean(internal_densities)) if internal_densities else 0.0,
    ]

    # ── Valence/arousal features ──────────────────────────────────
    sampled = all_lines[:200]
    emotions = score_lyrics("\n".join(sampled))
    valences = [e.valence for e in emotions]
    arousals = [e.arousal for e in emotions]
    feat_emotion = [
        float(np.mean(valences)),
        float(np.std(valences)),
        float(np.mean(arousals)),
        float(np.std(arousals)),
    ]

    # ── Punctuation features ──────────────────────────────────────
    all_text = " ".join(all_lines)
    n_chars = max(len(all_text), 1)
    feat_punct = [
        all_text.count("!") / n_chars * 100,
        all_text.count("?") / n_chars * 100,
        all_text.count(",") / n_chars * 100,
        all_text.count("'") / n_chars * 100,
    ]

    # ── Metaphor embedding (semantic cluster centroid) ─────────────
    # Sample lines, embed them, average → 384-dim → later reduced to 32d via PCA
    sample = all_lines[:100]
    model = _get_embed_model()
    embeddings = model.encode(sample, show_progress_bar=False, batch_size=32)  # (N, 384)
    metaphor_centroid = embeddings.mean(axis=0)  # (384,)

    # Concatenate scalar features: 3+2+2+2+4+4 = 17 scalar dims
    scalar_feats = np.array(
        feat_syllables + feat_length + feat_vocab + feat_rhyme + feat_emotion + feat_punct,
        dtype=float,
    )
    # Pad/trim scalar to 96 dims (repeat to fill)
    scalar_96 = np.resize(scalar_feats, 96)

    # Reduce metaphor centroid to 32 dims via random projection (no PCA needed at extract time)
    rng = np.random.default_rng(seed=42)
    proj = rng.standard_normal((384, 32)) / math.sqrt(384)
    metaphor_32 = metaphor_centroid @ proj  # (32,)

    vector_128 = np.concatenate([scalar_96, metaphor_32])
    return vector_128.astype(np.float32)


def build_style_vectors(
    jsonl_path: str,
    out_dir: str = "data/style_vectors",
    min_songs: int = 1,
) -> dict[str, StyleVector]:
    """
    Read artist songs from a JSONL file, compute style vectors,
    save as .npy files.
    """
    from collections import defaultdict
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    artist_songs: dict[str, list[str]] = defaultdict(list)
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            artist_songs[record["artist"]].append(record["lyrics"])

    style_vectors: dict[str, StyleVector] = {}
    for artist, songs in artist_songs.items():
        if len(songs) < min_songs:
            continue
        print(f"  Computing style vector for {artist} ({len(songs)} songs)...")
        vec = extract_style_features(songs)
        sv = StyleVector(
            artist=artist,
            vector=vec,
            song_count=len(songs),
            feature_names=[f"f{i}" for i in range(128)],
        )
        style_vectors[artist] = sv
        np.save(out_path / f"{artist.replace(' ', '_')}.npy", vec)

    print(f"\nSaved {len(style_vectors)} style vectors → {out_path}")
    return style_vectors


def load_style_vector(artist: str, vec_dir: str = "data/style_vectors") -> Optional[np.ndarray]:
    path = Path(vec_dir) / f"{artist.replace(' ', '_')}.npy"
    if path.exists():
        return np.load(str(path))
    return None


if __name__ == "__main__":
    # Quick test on fake data
    sample_songs = [
        "I been moving in silence\nThey can't feel my weight\nEvery step I take\nI'm moving with fate",
        "Diamonds on my wrist\nWhile I dance to the beat\nGame is cold but I\nTurn up all the heat",
    ]
    vec = extract_style_features(sample_songs)
    print(f"Style vector shape: {vec.shape}")
    print(f"First 10 dims: {vec[:10]}")

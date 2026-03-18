"""
Style DNA — maps every music style/genre to its audio + lyrical fingerprint.

Every style has:
  - Audio DNA (BPM range, energy, valence, danceability, speechiness)
  - Lyrical DNA (syllable density, rhyme scheme, vocabulary, line length)
  - Flow pattern (triplet, straight, syncopated, melodic, etc.)
  - Language weight (which languages dominate this style)
  - Trend score (updated weekly from chart data)
  - Blend compatibility (which styles it mixes well with)

This powers:
  1. Style conditioning at inference
  2. Cross-style blending (e.g. 60% Afrobeats + 40% Trap)
  3. Trend-aware generation (model knows what's hot right now)
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
from pathlib import Path


@dataclass
class StyleDNA:
    name: str
    region: str

    # Audio fingerprint
    bpm_range: tuple[int, int]         # (min, max)
    energy: float                       # 0-1
    valence: float                      # 0-1  (sad → happy)
    danceability: float                 # 0-1
    speechiness: float                  # 0=sung, 1=rap/spoken
    acousticness: float                 # 0-1

    # Lyrical fingerprint
    avg_syllables_per_line: float
    rhyme_schemes: list[str]            # e.g. ["AABB", "ABAB"]
    flow_patterns: list[str]            # e.g. ["triplet", "straight"]
    vocab_complexity: float             # 0-1 (simple → complex)
    repetition_ratio: float             # chorus repeat weight

    # Language
    primary_languages: list[str]
    uses_slang: bool
    uses_code_switching: bool           # mix of languages in one song

    # Blend compatibility
    blends_well_with: list[str]

    # Structural
    typical_bpm: int
    hook_position_sec: int              # when the hook usually drops
    song_structure: list[str]           # e.g. ["VERSE","CHORUS","VERSE","CHORUS","BRIDGE","CHORUS"]

    # Trend
    trend_score: float = 0.0           # updated from chart data
    chart_countries: list[str] = field(default_factory=list)


# ── Master style library — 40+ styles ────────────────────────────────────────

STYLES: dict[str, StyleDNA] = {

    # ── HIP-HOP / RAP ──────────────────────────────────────────────────────────

    "trap": StyleDNA(
        name="Trap", region="Americas",
        bpm_range=(130, 170), typical_bpm=140,
        energy=0.75, valence=0.40, danceability=0.82, speechiness=0.85, acousticness=0.05,
        avg_syllables_per_line=9, rhyme_schemes=["AABB","AAAA","ABAB"], flow_patterns=["triplet","double-time","straight"],
        vocab_complexity=0.45, repetition_ratio=0.40,
        primary_languages=["en"], uses_slang=True, uses_code_switching=False,
        blends_well_with=["drill","hip_hop","rnb","latin_trap"],
        hook_position_sec=45, song_structure=["INTRO","VERSE","CHORUS","VERSE","CHORUS","OUTRO"],
    ),
    "drill": StyleDNA(
        name="Drill", region="Americas/Europe",
        bpm_range=(140, 150), typical_bpm=145,
        energy=0.80, valence=0.25, danceability=0.75, speechiness=0.90, acousticness=0.02,
        avg_syllables_per_line=10, rhyme_schemes=["AABB","ABAB"], flow_patterns=["straight","sliding"],
        vocab_complexity=0.50, repetition_ratio=0.35,
        primary_languages=["en"], uses_slang=True, uses_code_switching=False,
        blends_well_with=["trap","uk_rap","grime"],
        hook_position_sec=40, song_structure=["VERSE","CHORUS","VERSE","CHORUS","OUTRO"],
    ),
    "hip_hop": StyleDNA(
        name="Hip-Hop", region="Americas",
        bpm_range=(85, 100), typical_bpm=93,
        energy=0.65, valence=0.55, danceability=0.78, speechiness=0.88, acousticness=0.10,
        avg_syllables_per_line=12, rhyme_schemes=["AABB","ABAB","ABCB"], flow_patterns=["straight","syncopated","multisyl"],
        vocab_complexity=0.80, repetition_ratio=0.30,
        primary_languages=["en"], uses_slang=True, uses_code_switching=False,
        blends_well_with=["rnb","jazz_rap","boom_bap","trap"],
        hook_position_sec=50, song_structure=["INTRO","VERSE","HOOK","VERSE","HOOK","BRIDGE","HOOK","OUTRO"],
    ),
    "boom_bap": StyleDNA(
        name="Boom Bap", region="Americas",
        bpm_range=(85, 100), typical_bpm=90,
        energy=0.60, valence=0.50, danceability=0.72, speechiness=0.88, acousticness=0.15,
        avg_syllables_per_line=13, rhyme_schemes=["AABB","ABAB"], flow_patterns=["straight","multisyl"],
        vocab_complexity=0.85, repetition_ratio=0.25,
        primary_languages=["en"], uses_slang=True, uses_code_switching=False,
        blends_well_with=["hip_hop","jazz_rap"],
        hook_position_sec=55, song_structure=["VERSE","HOOK","VERSE","HOOK","VERSE"],
    ),
    "uk_rap": StyleDNA(
        name="UK Rap", region="Europe",
        bpm_range=(130, 145), typical_bpm=138,
        energy=0.72, valence=0.35, danceability=0.78, speechiness=0.88, acousticness=0.03,
        avg_syllables_per_line=11, rhyme_schemes=["AABB","ABAB"], flow_patterns=["straight","fast"],
        vocab_complexity=0.60, repetition_ratio=0.35,
        primary_languages=["en"], uses_slang=True, uses_code_switching=False,
        blends_well_with=["drill","grime","afroswing"],
        hook_position_sec=40, song_structure=["VERSE","CHORUS","VERSE","CHORUS","OUTRO"],
    ),
    "grime": StyleDNA(
        name="Grime", region="Europe",
        bpm_range=(140, 140), typical_bpm=140,
        energy=0.85, valence=0.30, danceability=0.70, speechiness=0.92, acousticness=0.02,
        avg_syllables_per_line=14, rhyme_schemes=["AABB","AAAA"], flow_patterns=["rapid","staccato"],
        vocab_complexity=0.65, repetition_ratio=0.28,
        primary_languages=["en"], uses_slang=True, uses_code_switching=False,
        blends_well_with=["uk_rap","drill"],
        hook_position_sec=35, song_structure=["VERSE","HOOK","VERSE","HOOK"],
    ),
    "latin_trap": StyleDNA(
        name="Latin Trap", region="Latin America",
        bpm_range=(130, 160), typical_bpm=145,
        energy=0.78, valence=0.50, danceability=0.88, speechiness=0.82, acousticness=0.04,
        avg_syllables_per_line=10, rhyme_schemes=["AABB","ABAB"], flow_patterns=["triplet","reggaeton"],
        vocab_complexity=0.45, repetition_ratio=0.45,
        primary_languages=["es"], uses_slang=True, uses_code_switching=True,
        blends_well_with=["trap","reggaeton","rnb"],
        hook_position_sec=40, song_structure=["INTRO","VERSE","CORO","VERSE","CORO","PUENTE","CORO"],
    ),

    # ── R&B / SOUL ──────────────────────────────────────────────────────────────

    "rnb": StyleDNA(
        name="R&B", region="Americas",
        bpm_range=(60, 100), typical_bpm=82,
        energy=0.55, valence=0.60, danceability=0.75, speechiness=0.20, acousticness=0.20,
        avg_syllables_per_line=9, rhyme_schemes=["ABAB","ABCB","AABB"], flow_patterns=["melodic","sustained"],
        vocab_complexity=0.60, repetition_ratio=0.50,
        primary_languages=["en"], uses_slang=True, uses_code_switching=False,
        blends_well_with=["soul","trap","pop","neo_soul"],
        hook_position_sec=55, song_structure=["VERSE","PRECHORUS","CHORUS","VERSE","PRECHORUS","CHORUS","BRIDGE","CHORUS"],
    ),
    "neo_soul": StyleDNA(
        name="Neo Soul", region="Americas",
        bpm_range=(70, 100), typical_bpm=85,
        energy=0.48, valence=0.58, danceability=0.70, speechiness=0.15, acousticness=0.45,
        avg_syllables_per_line=10, rhyme_schemes=["ABAB","FREE"], flow_patterns=["melodic","free"],
        vocab_complexity=0.75, repetition_ratio=0.40,
        primary_languages=["en"], uses_slang=False, uses_code_switching=False,
        blends_well_with=["rnb","soul","jazz_rap"],
        hook_position_sec=60, song_structure=["VERSE","CHORUS","VERSE","CHORUS","BRIDGE","CHORUS"],
    ),

    # ── POP ─────────────────────────────────────────────────────────────────────

    "pop": StyleDNA(
        name="Pop", region="Global",
        bpm_range=(100, 130), typical_bpm=118,
        energy=0.68, valence=0.72, danceability=0.80, speechiness=0.10, acousticness=0.12,
        avg_syllables_per_line=8, rhyme_schemes=["AABB","ABAB"], flow_patterns=["melodic","straight"],
        vocab_complexity=0.35, repetition_ratio=0.60,
        primary_languages=["en","es","ko","fr"], uses_slang=False, uses_code_switching=False,
        blends_well_with=["rnb","edm","indie_pop","kpop"],
        hook_position_sec=40, song_structure=["VERSE","PRECHORUS","CHORUS","VERSE","PRECHORUS","CHORUS","BRIDGE","CHORUS"],
    ),
    "indie_pop": StyleDNA(
        name="Indie Pop", region="Europe/Americas",
        bpm_range=(100, 130), typical_bpm=112,
        energy=0.55, valence=0.62, danceability=0.65, speechiness=0.08, acousticness=0.30,
        avg_syllables_per_line=9, rhyme_schemes=["ABAB","FREE","ABCB"], flow_patterns=["melodic","conversational"],
        vocab_complexity=0.70, repetition_ratio=0.45,
        primary_languages=["en"], uses_slang=False, uses_code_switching=False,
        blends_well_with=["indie","pop","alternative"],
        hook_position_sec=50, song_structure=["VERSE","CHORUS","VERSE","CHORUS","BRIDGE","CHORUS"],
    ),

    # ── AFROBEATS / AFROPOP ─────────────────────────────────────────────────────

    "afrobeats": StyleDNA(
        name="Afrobeats", region="Africa",
        bpm_range=(95, 110), typical_bpm=103,
        energy=0.78, valence=0.82, danceability=0.92, speechiness=0.25, acousticness=0.08,
        avg_syllables_per_line=8, rhyme_schemes=["AABB","FREE"], flow_patterns=["syncopated","call-response"],
        vocab_complexity=0.40, repetition_ratio=0.55,
        primary_languages=["en","yo","ig","pcm"], uses_slang=True, uses_code_switching=True,
        blends_well_with=["afroswing","dancehall","rnb","pop"],
        hook_position_sec=35, song_structure=["INTRO","VERSE","CHORUS","VERSE","CHORUS","BRIDGE","CHORUS"],
    ),
    "afroswing": StyleDNA(
        name="Afroswing", region="Africa/Europe",
        bpm_range=(95, 115), typical_bpm=105,
        energy=0.72, valence=0.78, danceability=0.88, speechiness=0.40, acousticness=0.10,
        avg_syllables_per_line=9, rhyme_schemes=["AABB","ABAB"], flow_patterns=["syncopated","melodic"],
        vocab_complexity=0.45, repetition_ratio=0.50,
        primary_languages=["en","yo","tw"], uses_slang=True, uses_code_switching=True,
        blends_well_with=["afrobeats","uk_rap","rnb"],
        hook_position_sec=40, song_structure=["VERSE","HOOK","VERSE","HOOK","OUTRO"],
    ),
    "amapiano": StyleDNA(
        name="Amapiano", region="Africa",
        bpm_range=(100, 115), typical_bpm=110,
        energy=0.75, valence=0.80, danceability=0.95, speechiness=0.30, acousticness=0.05,
        avg_syllables_per_line=7, rhyme_schemes=["FREE","AABB"], flow_patterns=["chanted","call-response"],
        vocab_complexity=0.30, repetition_ratio=0.65,
        primary_languages=["zu","st","en"], uses_slang=True, uses_code_switching=True,
        blends_well_with=["afrobeats","house","dancehall"],
        hook_position_sec=30, song_structure=["INTRO","LOG DRUM BREAK","VERSE","CHORUS","VERSE","CHORUS"],
    ),

    # ── LATIN ──────────────────────────────────────────────────────────────────

    "reggaeton": StyleDNA(
        name="Reggaeton", region="Latin America",
        bpm_range=(90, 100), typical_bpm=95,
        energy=0.82, valence=0.75, danceability=0.93, speechiness=0.45, acousticness=0.04,
        avg_syllables_per_line=9, rhyme_schemes=["AABB","ABAB"], flow_patterns=["dembow","triplet"],
        vocab_complexity=0.40, repetition_ratio=0.55,
        primary_languages=["es"], uses_slang=True, uses_code_switching=True,
        blends_well_with=["latin_trap","pop","dancehall"],
        hook_position_sec=38, song_structure=["INTRO","VERSO","CORO","VERSO","CORO","PUENTE","CORO"],
    ),
    "bachata": StyleDNA(
        name="Bachata", region="Latin America",
        bpm_range=(100, 140), typical_bpm=120,
        energy=0.60, valence=0.55, danceability=0.82, speechiness=0.10, acousticness=0.35,
        avg_syllables_per_line=8, rhyme_schemes=["ABAB","AABB"], flow_patterns=["melodic","romantic"],
        vocab_complexity=0.35, repetition_ratio=0.55,
        primary_languages=["es"], uses_slang=False, uses_code_switching=False,
        blends_well_with=["reggaeton","pop"],
        hook_position_sec=45, song_structure=["VERSO","CORO","VERSO","CORO","PUENTE","CORO"],
    ),
    "corrido": StyleDNA(
        name="Corridos Tumbados", region="Latin America",
        bpm_range=(130, 160), typical_bpm=148,
        energy=0.80, valence=0.45, danceability=0.75, speechiness=0.70, acousticness=0.15,
        avg_syllables_per_line=10, rhyme_schemes=["AABB","ABAB"], flow_patterns=["narrative","triplet"],
        vocab_complexity=0.55, repetition_ratio=0.38,
        primary_languages=["es"], uses_slang=True, uses_code_switching=False,
        blends_well_with=["latin_trap","banda","norteno"],
        hook_position_sec=42, song_structure=["VERSO","CORO","VERSO","CORO","PUENTE","CORO"],
    ),

    # ── K-POP / ASIAN ──────────────────────────────────────────────────────────

    "kpop": StyleDNA(
        name="K-Pop", region="Asia",
        bpm_range=(95, 140), typical_bpm=120,
        energy=0.78, valence=0.75, danceability=0.85, speechiness=0.15, acousticness=0.08,
        avg_syllables_per_line=8, rhyme_schemes=["AABB","ABAB"], flow_patterns=["melodic","rap-break"],
        vocab_complexity=0.45, repetition_ratio=0.60,
        primary_languages=["ko","en"], uses_slang=False, uses_code_switching=True,
        blends_well_with=["pop","edm","hip_hop"],
        hook_position_sec=40, song_structure=["INTRO","VERSE","PRECHORUS","CHORUS","VERSE","PRECHORUS","CHORUS","BRIDGE","CHORUS","OUTRO"],
    ),
    "j_pop": StyleDNA(
        name="J-Pop", region="Asia",
        bpm_range=(120, 150), typical_bpm=135,
        energy=0.72, valence=0.70, danceability=0.80, speechiness=0.12, acousticness=0.12,
        avg_syllables_per_line=9, rhyme_schemes=["AABB","ABAB"], flow_patterns=["melodic"],
        vocab_complexity=0.45, repetition_ratio=0.55,
        primary_languages=["ja"], uses_slang=False, uses_code_switching=True,
        blends_well_with=["pop","kpop","anime_ost"],
        hook_position_sec=45, song_structure=["VERSE","CHORUS","VERSE","CHORUS","BRIDGE","CHORUS"],
    ),

    # ── ARABIC / MIDDLE EAST ───────────────────────────────────────────────────

    "arabic_pop": StyleDNA(
        name="Arabic Pop", region="Middle East",
        bpm_range=(90, 120), typical_bpm=105,
        energy=0.65, valence=0.60, danceability=0.78, speechiness=0.15, acousticness=0.20,
        avg_syllables_per_line=9, rhyme_schemes=["AABB","AAAA"], flow_patterns=["melodic","ornamental"],
        vocab_complexity=0.55, repetition_ratio=0.50,
        primary_languages=["ar"], uses_slang=False, uses_code_switching=False,
        blends_well_with=["pop","khaleeji","rnb"],
        hook_position_sec=48, song_structure=["INTRO","VERSE","CHORUS","VERSE","CHORUS","OUTRO"],
    ),
    "mahraganat": StyleDNA(
        name="Mahraganat", region="Middle East",
        bpm_range=(120, 150), typical_bpm=138,
        energy=0.88, valence=0.65, danceability=0.90, speechiness=0.75, acousticness=0.02,
        avg_syllables_per_line=10, rhyme_schemes=["AABB","AAAA"], flow_patterns=["rapid","chanted"],
        vocab_complexity=0.35, repetition_ratio=0.60,
        primary_languages=["ar"], uses_slang=True, uses_code_switching=False,
        blends_well_with=["arabic_pop","trap"],
        hook_position_sec=30, song_structure=["INTRO","VERSE","CHORUS","VERSE","CHORUS"],
    ),

    # ── MOROCCAN ───────────────────────────────────────────────────────────────

    "chaabi": StyleDNA(
        name="Chaabi", region="Africa",
        bpm_range=(100, 140), typical_bpm=118,
        energy=0.70, valence=0.70, danceability=0.85, speechiness=0.20, acousticness=0.40,
        avg_syllables_per_line=8, rhyme_schemes=["AABB","AAAA"], flow_patterns=["call-response","melodic"],
        vocab_complexity=0.40, repetition_ratio=0.60,
        primary_languages=["ar","ber","fr"], uses_slang=True, uses_code_switching=True,
        blends_well_with=["gnawa","rai","arabic_pop"],
        hook_position_sec=40, song_structure=["INTRO","VERSE","REFRAIN","VERSE","REFRAIN"],
    ),
    "rai": StyleDNA(
        name="Raï", region="Africa",
        bpm_range=(95, 130), typical_bpm=112,
        energy=0.72, valence=0.65, danceability=0.82, speechiness=0.25, acousticness=0.30,
        avg_syllables_per_line=9, rhyme_schemes=["AABB","FREE"], flow_patterns=["melodic","improvisational"],
        vocab_complexity=0.45, repetition_ratio=0.50,
        primary_languages=["ar","fr","ber"], uses_slang=True, uses_code_switching=True,
        blends_well_with=["chaabi","arabic_pop","pop"],
        hook_position_sec=45, song_structure=["INTRO","VERSE","CHORUS","VERSE","CHORUS","OUTRO"],
    ),

    # ── DANCEHALL / REGGAE ─────────────────────────────────────────────────────

    "dancehall": StyleDNA(
        name="Dancehall", region="Americas",
        bpm_range=(90, 110), typical_bpm=100,
        energy=0.80, valence=0.78, danceability=0.90, speechiness=0.60, acousticness=0.05,
        avg_syllables_per_line=9, rhyme_schemes=["AABB","ABAB"], flow_patterns=["patois","syncopated"],
        vocab_complexity=0.45, repetition_ratio=0.55,
        primary_languages=["en","patois"], uses_slang=True, uses_code_switching=True,
        blends_well_with=["reggaeton","afrobeats","pop"],
        hook_position_sec=38, song_structure=["INTRO","VERSE","CHORUS","VERSE","CHORUS","BRIDGE","CHORUS"],
    ),

    # ── INDIE / ALTERNATIVE ────────────────────────────────────────────────────

    "indie": StyleDNA(
        name="Indie", region="Europe/Americas",
        bpm_range=(90, 130), typical_bpm=108,
        energy=0.55, valence=0.52, danceability=0.60, speechiness=0.07, acousticness=0.35,
        avg_syllables_per_line=9, rhyme_schemes=["ABAB","FREE","ABCB"], flow_patterns=["conversational","melodic"],
        vocab_complexity=0.80, repetition_ratio=0.38,
        primary_languages=["en"], uses_slang=False, uses_code_switching=False,
        blends_well_with=["indie_pop","alternative","folk"],
        hook_position_sec=55, song_structure=["VERSE","CHORUS","VERSE","CHORUS","BRIDGE","CHORUS"],
    ),
    "alt_emo": StyleDNA(
        name="Alt/Emo", region="Americas/Europe",
        bpm_range=(90, 150), typical_bpm=128,
        energy=0.72, valence=0.30, danceability=0.58, speechiness=0.08, acousticness=0.20,
        avg_syllables_per_line=10, rhyme_schemes=["ABAB","AABB","FREE"], flow_patterns=["melodic","screamo"],
        vocab_complexity=0.70, repetition_ratio=0.45,
        primary_languages=["en"], uses_slang=False, uses_code_switching=False,
        blends_well_with=["indie","pop_punk","alternative"],
        hook_position_sec=48, song_structure=["VERSE","CHORUS","VERSE","CHORUS","BRIDGE","CHORUS"],
    ),

    # ── ELECTRONIC / EDM ──────────────────────────────────────────────────────

    "edm": StyleDNA(
        name="EDM", region="Global",
        bpm_range=(128, 140), typical_bpm=132,
        energy=0.92, valence=0.75, danceability=0.90, speechiness=0.05, acousticness=0.02,
        avg_syllables_per_line=6, rhyme_schemes=["AABB","FREE"], flow_patterns=["melodic","minimal"],
        vocab_complexity=0.25, repetition_ratio=0.75,
        primary_languages=["en"], uses_slang=False, uses_code_switching=False,
        blends_well_with=["pop","trap","future_bass"],
        hook_position_sec=32, song_structure=["INTRO","BUILD","DROP","BREAK","BUILD","DROP","OUTRO"],
    ),

    # ── FRENCH RAP ─────────────────────────────────────────────────────────────

    "french_rap": StyleDNA(
        name="French Rap", region="Europe",
        bpm_range=(90, 140), typical_bpm=115,
        energy=0.70, valence=0.40, danceability=0.78, speechiness=0.88, acousticness=0.05,
        avg_syllables_per_line=12, rhyme_schemes=["AABB","ABAB"], flow_patterns=["fast","multisyl"],
        vocab_complexity=0.75, repetition_ratio=0.35,
        primary_languages=["fr"], uses_slang=True, uses_code_switching=True,
        blends_well_with=["trap","pop","rnb"],
        hook_position_sec=45, song_structure=["COUPLET","REFRAIN","COUPLET","REFRAIN","PONTE","REFRAIN"],
    ),

    # ── COUNTRY ────────────────────────────────────────────────────────────────

    "country": StyleDNA(
        name="Country", region="Americas",
        bpm_range=(90, 130), typical_bpm=108,
        energy=0.60, valence=0.65, danceability=0.65, speechiness=0.08, acousticness=0.55,
        avg_syllables_per_line=9, rhyme_schemes=["AABB","ABAB"], flow_patterns=["narrative","melodic"],
        vocab_complexity=0.50, repetition_ratio=0.50,
        primary_languages=["en"], uses_slang=True, uses_code_switching=False,
        blends_well_with=["pop","folk","rock"],
        hook_position_sec=50, song_structure=["VERSE","CHORUS","VERSE","CHORUS","BRIDGE","CHORUS"],
    ),
}


# ── Style blending ────────────────────────────────────────────────────────────

def blend_styles(blend: dict[str, float]) -> StyleDNA:
    """
    Blend multiple styles by weight.
    e.g. blend = {"afrobeats": 0.6, "trap": 0.3, "kpop": 0.1}
    Returns a new StyleDNA with interpolated features.
    """
    assert abs(sum(blend.values()) - 1.0) < 0.01, "Weights must sum to 1.0"

    # Weighted average of numeric fields
    result_bpm_min = 0.0
    result_bpm_max = 0.0
    result_typical_bpm = 0.0
    energy = danceability = valence = speechiness = acousticness = 0.0
    avg_syl = repetition = vocab = 0.0
    hook_pos = 0.0

    all_languages: set[str] = set()
    all_rhymes: list[str] = []
    all_flows: list[str] = []
    all_blends: list[str] = []
    uses_slang = False
    uses_cs = False

    for style_name, weight in blend.items():
        s = STYLES[style_name]
        result_bpm_min     += s.bpm_range[0] * weight
        result_bpm_max     += s.bpm_range[1] * weight
        result_typical_bpm += s.typical_bpm * weight
        energy             += s.energy * weight
        danceability       += s.danceability * weight
        valence            += s.valence * weight
        speechiness        += s.speechiness * weight
        acousticness       += s.acousticness * weight
        avg_syl            += s.avg_syllables_per_line * weight
        repetition         += s.repetition_ratio * weight
        vocab              += s.vocab_complexity * weight
        hook_pos           += s.hook_position_sec * weight
        all_languages.update(s.primary_languages)
        all_rhymes.extend(s.rhyme_schemes)
        all_flows.extend(s.flow_patterns)
        all_blends.extend(s.blends_well_with)
        uses_slang = uses_slang or s.uses_slang
        uses_cs    = uses_cs or s.uses_code_switching

    blend_name = " x ".join(f"{int(w*100)}% {n}" for n, w in blend.items())
    return StyleDNA(
        name=blend_name,
        region="Blended",
        bpm_range=(int(result_bpm_min), int(result_bpm_max)),
        typical_bpm=int(result_typical_bpm),
        energy=round(energy, 3),
        valence=round(valence, 3),
        danceability=round(danceability, 3),
        speechiness=round(speechiness, 3),
        acousticness=round(acousticness, 3),
        avg_syllables_per_line=round(avg_syl, 1),
        rhyme_schemes=list(dict.fromkeys(all_rhymes))[:4],
        flow_patterns=list(dict.fromkeys(all_flows))[:4],
        vocab_complexity=round(vocab, 3),
        repetition_ratio=round(repetition, 3),
        primary_languages=list(all_languages),
        uses_slang=uses_slang,
        uses_code_switching=uses_cs,
        blends_well_with=list(dict.fromkeys(all_blends))[:6],
        hook_position_sec=int(hook_pos),
        song_structure=["INTRO","VERSE","CHORUS","VERSE","CHORUS","BRIDGE","CHORUS","OUTRO"],
    )


def update_trend_scores(viral_scores: list[dict]):
    """Update STYLES trend scores from weekly chart viral data."""
    # Map style names to viral data (rough genre tag matching)
    genre_viral: dict[str, float] = {}
    for song in viral_scores:
        # Infer genre from audio features
        bpm = song.get("bpm", 0)
        speech = song.get("speechiness", 0)
        dance = song.get("danceability", 0)
        n_countries = song.get("n_countries", 1)

        if speech > 0.66:
            genre_viral["trap"] = genre_viral.get("trap", 0) + song["viral_score"]
            genre_viral["hip_hop"] = genre_viral.get("hip_hop", 0) + song["viral_score"] * 0.5
        if dance > 0.85 and bpm > 90:
            genre_viral["afrobeats"] = genre_viral.get("afrobeats", 0) + song["viral_score"]
            genre_viral["reggaeton"] = genre_viral.get("reggaeton", 0) + song["viral_score"] * 0.5
        if n_countries > 10:
            genre_viral["pop"] = genre_viral.get("pop", 0) + song["viral_score"]

    # Normalize and update
    max_score = max(genre_viral.values(), default=1)
    for genre, score in genre_viral.items():
        if genre in STYLES:
            STYLES[genre].trend_score = round(score / max_score, 3)


def get_trending_styles(top_n: int = 10) -> list[tuple[str, float]]:
    """Return top N styles sorted by current trend score."""
    scored = [(name, s.trend_score) for name, s in STYLES.items()]
    return sorted(scored, key=lambda x: -x[1])[:top_n]


def style_to_prompt_prefix(style: StyleDNA) -> str:
    """Convert a StyleDNA into a text prompt prefix for the model."""
    langs = "/".join(style.primary_languages[:3])
    return (
        f"[STYLE: {style.name}] "
        f"[BPM: {style.typical_bpm}] "
        f"[LANG: {langs}] "
        f"[FLOW: {style.flow_patterns[0] if style.flow_patterns else 'melodic'}] "
        f"[ENERGY: {'high' if style.energy > 0.7 else 'mid' if style.energy > 0.4 else 'low'}] "
        f"[VIBE: {'happy' if style.valence > 0.6 else 'dark' if style.valence < 0.4 else 'neutral'}]"
    )


if __name__ == "__main__":
    print(f"Loaded {len(STYLES)} styles\n")

    # Demo blend
    blended = blend_styles({"afrobeats": 0.5, "trap": 0.3, "kpop": 0.2})
    print(f"Blended style: {blended.name}")
    print(f"  BPM: {blended.typical_bpm}")
    print(f"  Energy: {blended.energy}")
    print(f"  Languages: {blended.primary_languages}")
    print(f"  Prompt: {style_to_prompt_prefix(blended)}")

    print(f"\nAll styles:")
    for name, style in STYLES.items():
        print(f"  {name:20s} {style.region:20s} BPM:{style.typical_bpm:3d}  speech:{style.speechiness:.2f}")

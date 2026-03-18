"""
Phonosemantic Texture Engine
============================
The secret weapon of the world's greatest lyricists.

Phonosemantics = the phenomenon where the SOUND of words carries meaning.
This is not metaphor. This is acoustic psychology.

  "Crash" sounds violent because of the hard K + harsh SH
  "Silk" sounds smooth because of the liquid L + soft K
  "Murmur" sounds quiet because of the resonant nasals
  "Glitter" sounds bright because of high vowels + bright consonants

Kendrick Lamar on "DAMN.":
  - "Humble" — the H + M + B + L = heavy, grounded, downward energy
  - "Element" — EH + L + M = forward motion, medium brightness
  - The entire album has a PHONETIC TEXTURE that feels oppressive → released

Drake on moody R&B:
  - "Marvin's Room" — M + R + V + N = warm, resonant, close
  - Every dark hook uses low back vowels (OH, OW, OO) = phonetic darkness

Bad Bunny:
  - Spanish has naturally warmer phonetic texture (more vowels, softer stops)
  - His dark tracks use AH/OH vowels + D/G stops instead of bright EE/T/K

This module:
  1. Analyzes the phonetic texture of any line (brightness, warmth, weight, sharpness)
  2. Defines ideal texture targets for each mood/section combination
  3. Scores candidate lines by phonosemantic alignment
  4. Can SUGGEST word substitutions that improve phonosemantic fit
     (e.g., "cold night" → "dark hollow night" for darker texture)

TEXTURE DIMENSIONS
------------------
  brightness  — [-1 dark, +1 bright]   (vowel height + voiceless stops)
  warmth      — [-1 cold, +1 warm]     (nasals M/N/NG + voiced resonants)
  weight      — [-1 light, +1 heavy]   (low vowels + voiced stops + /G/ /B/)
  sharpness   — [-1 smooth, +1 sharp]  (clusters + voiceless stops + /S/ /K/)
  openness    — [-1 closed, +1 open]   (low vowels AA/AO/AW + open mouth shape)
  resonance   — [-1 dry, +1 resonant]  (nasals + /L/ + /R/ + voiced fricatives)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import pronouncing
    _PRONOUNCING_OK = True
except ImportError:
    _PRONOUNCING_OK = False


# ── Phoneme → texture contribution ────────────────────────────────────────────
# Each row: (brightness, warmth, weight, sharpness, openness, resonance)

PHONEME_TEXTURE: dict[str, tuple[float, float, float, float, float, float]] = {
    # High front vowels → bright, light, closed
    "IY": ( 0.9,  0.2, -0.6,  0.0, -0.5,  0.3),
    "IH": ( 0.6,  0.2, -0.4,  0.0, -0.4,  0.2),
    "EY": ( 0.7,  0.3, -0.3,  0.0, -0.3,  0.2),
    "EH": ( 0.4,  0.2, -0.2,  0.1, -0.2,  0.1),
    # Low open vowels → dark, heavy, open
    "AA": (-0.1,  0.1,  0.3,  0.0,  0.9,  0.2),
    "AO": (-0.4,  0.1,  0.5,  0.0,  0.7,  0.3),
    "AW": (-0.5,  0.1,  0.5,  0.0,  0.6,  0.2),
    "OW": (-0.5,  0.2,  0.5,  0.0,  0.3,  0.3),
    "OY": (-0.2,  0.2,  0.3,  0.0,  0.2,  0.3),
    # Back round vowels → dark, heavy, warm
    "UW": (-0.7,  0.3,  0.7, -0.1, -0.2,  0.4),
    "UH": (-0.4,  0.2,  0.4, -0.1, -0.1,  0.2),
    # Central vowels
    "AH": (-0.2,  0.1,  0.1,  0.0,  0.3,  0.1),
    "AE": ( 0.2,  0.1, -0.1,  0.2,  0.5,  0.1),
    "ER": ( 0.0,  0.3,  0.1,  0.0,  0.0,  0.5),
    "AY": ( 0.3,  0.1,  0.0,  0.0,  0.4,  0.2),
    # Voiceless stops → sharp, light, dry
    "P":  ( 0.3,  0.0, -0.2,  0.6, -0.1, -0.2),
    "T":  ( 0.5,  0.0, -0.3,  0.8, -0.2, -0.3),
    "K":  ( 0.4, -0.1, -0.2,  0.7, -0.1, -0.2),
    # Voiced stops → grounded, heavier
    "B":  (-0.1,  0.1,  0.3,  0.3, -0.1,  0.0),
    "D":  (-0.1,  0.0,  0.2,  0.4, -0.1,  0.0),
    "G":  (-0.3,  0.0,  0.4,  0.3, -0.1,  0.0),
    # Voiceless fricatives → crisp, bright
    "F":  ( 0.2,  0.0, -0.1,  0.5,  0.0, -0.1),
    "S":  ( 0.6, -0.1, -0.2,  0.6, -0.1, -0.1),
    "SH": ( 0.4,  0.0, -0.1,  0.4,  0.0,  0.0),
    "TH": ( 0.2,  0.0, -0.1,  0.3,  0.1, -0.1),
    "HH": ( 0.3,  0.0, -0.1,  0.2,  0.2, -0.1),
    # Voiced fricatives → smooth, warm
    "V":  ( 0.0,  0.2,  0.1,  0.1,  0.0,  0.3),
    "Z":  ( 0.1,  0.1,  0.0,  0.1,  0.0,  0.3),
    "ZH": (-0.1,  0.2,  0.1,  0.0,  0.0,  0.4),
    "DH": (-0.1,  0.1,  0.1,  0.1,  0.1,  0.2),
    # Nasals → warm, resonant
    "M":  ( 0.1,  0.8,  0.3, -0.2,  0.0,  0.8),
    "N":  ( 0.0,  0.6,  0.1, -0.1,  0.0,  0.6),
    "NG": (-0.1,  0.5,  0.2, -0.1,  0.0,  0.5),
    # Liquids → smooth, resonant
    "L":  ( 0.1,  0.4,  0.0, -0.2,  0.1,  0.7),
    "R":  (-0.1,  0.3,  0.1, -0.1,  0.1,  0.6),
    # Affricates
    "CH": ( 0.3,  0.0, -0.1,  0.5,  0.0, -0.1),
    "JH": (-0.1,  0.1,  0.1,  0.3,  0.0,  0.1),
    # Glides
    "W":  (-0.2,  0.3,  0.2, -0.2,  0.1,  0.4),
    "Y":  ( 0.3,  0.2, -0.1, -0.1,  0.0,  0.3),
}

TEXTURE_DIMS = ["brightness", "warmth", "weight", "sharpness", "openness", "resonance"]


@dataclass
class TextureProfile:
    brightness: float = 0.0   # -1 dark → +1 bright
    warmth:     float = 0.0   # -1 cold → +1 warm
    weight:     float = 0.0   # -1 light → +1 heavy
    sharpness:  float = 0.0   # -1 smooth → +1 sharp
    openness:   float = 0.0   # -1 closed → +1 open
    resonance:  float = 0.0   # -1 dry → +1 resonant

    def to_array(self) -> np.ndarray:
        return np.array([
            self.brightness, self.warmth, self.weight,
            self.sharpness, self.openness, self.resonance,
        ], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "TextureProfile":
        return cls(*arr[:6].tolist())

    def distance_to(self, other: "TextureProfile") -> float:
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def describe(self) -> str:
        """Human-readable texture description."""
        parts = []
        if self.brightness > 0.4:   parts.append("bright")
        elif self.brightness < -0.4: parts.append("dark")
        if self.warmth > 0.4:       parts.append("warm")
        elif self.warmth < -0.3:    parts.append("cold")
        if self.weight > 0.4:       parts.append("heavy")
        elif self.weight < -0.4:    parts.append("light")
        if self.sharpness > 0.4:    parts.append("sharp")
        elif self.sharpness < -0.3: parts.append("smooth")
        if self.resonance > 0.4:    parts.append("resonant")
        elif self.resonance < -0.3: parts.append("dry")
        return " / ".join(parts) if parts else "neutral"


def analyze_line_texture(line: str) -> TextureProfile:
    """
    Compute the phonosemantic texture profile of a line.
    Works best with English but falls back gracefully for other languages.
    """
    words = line.lower().split()
    if not words:
        return TextureProfile()

    all_phoneme_vals: list[np.ndarray] = []

    for word in words:
        clean = word.strip(".,!?'\"")
        if not clean or not _PRONOUNCING_OK:
            continue
        try:
            import pronouncing
            phones_list = pronouncing.phones_for_word(clean)
            if not phones_list:
                continue
            phones = phones_list[0].split()
            for ph in phones:
                ph_key = ph.rstrip("012")  # strip stress markers
                if ph_key in PHONEME_TEXTURE:
                    all_phoneme_vals.append(np.array(PHONEME_TEXTURE[ph_key]))
        except Exception:
            continue

    if not all_phoneme_vals:
        # Fallback: rough approximation from character patterns
        text = line.lower()
        brightness = (text.count('e') + text.count('i') - text.count('o') - text.count('u')) / max(len(text), 1)
        warmth = (text.count('m') + text.count('n') + text.count('l')) / max(len(text), 1)
        return TextureProfile(brightness=float(brightness * 2), warmth=float(warmth * 4))

    avg = np.mean(np.stack(all_phoneme_vals), axis=0)
    return TextureProfile.from_array(np.clip(avg, -1, 1))


# ── Target texture profiles per mood × section ────────────────────────────────

MOOD_SECTION_TEXTURE: dict[str, dict[str, TextureProfile]] = {
    "dark": {
        "verse1":  TextureProfile(brightness=-0.5, warmth= 0.1, weight= 0.5, sharpness= 0.2, openness= 0.2, resonance= 0.3),
        "chorus":  TextureProfile(brightness=-0.3, warmth= 0.0, weight= 0.7, sharpness= 0.5, openness= 0.3, resonance= 0.2),
        "bridge":  TextureProfile(brightness=-0.7, warmth= 0.2, weight= 0.7, sharpness=-0.1, openness= 0.3, resonance= 0.6),
    },
    "hype": {
        "verse1":  TextureProfile(brightness= 0.3, warmth= 0.0, weight= 0.3, sharpness= 0.7, openness= 0.2, resonance=-0.1),
        "chorus":  TextureProfile(brightness= 0.7, warmth= 0.1, weight= 0.1, sharpness= 0.9, openness= 0.4, resonance=-0.2),
        "bridge":  TextureProfile(brightness= 0.2, warmth= 0.0, weight= 0.5, sharpness= 0.5, openness= 0.3, resonance= 0.0),
    },
    "romantic": {
        "verse1":  TextureProfile(brightness= 0.3, warmth= 0.7, weight=-0.2, sharpness=-0.3, openness= 0.2, resonance= 0.7),
        "chorus":  TextureProfile(brightness= 0.5, warmth= 0.8, weight=-0.1, sharpness=-0.2, openness= 0.4, resonance= 0.8),
        "bridge":  TextureProfile(brightness= 0.2, warmth= 0.9, weight= 0.0, sharpness=-0.4, openness= 0.3, resonance= 0.9),
    },
    "chill": {
        "verse1":  TextureProfile(brightness= 0.1, warmth= 0.5, weight= 0.0, sharpness=-0.4, openness= 0.3, resonance= 0.6),
        "chorus":  TextureProfile(brightness= 0.3, warmth= 0.6, weight=-0.2, sharpness=-0.3, openness= 0.4, resonance= 0.7),
        "bridge":  TextureProfile(brightness= 0.0, warmth= 0.6, weight= 0.1, sharpness=-0.5, openness= 0.3, resonance= 0.8),
    },
    "sad": {
        "verse1":  TextureProfile(brightness=-0.4, warmth= 0.3, weight= 0.3, sharpness=-0.2, openness= 0.2, resonance= 0.5),
        "chorus":  TextureProfile(brightness=-0.2, warmth= 0.4, weight= 0.2, sharpness=-0.1, openness= 0.4, resonance= 0.6),
        "bridge":  TextureProfile(brightness=-0.5, warmth= 0.5, weight= 0.4, sharpness=-0.3, openness= 0.3, resonance= 0.7),
    },
    "epic": {
        "verse1":  TextureProfile(brightness= 0.1, warmth= 0.1, weight= 0.6, sharpness= 0.3, openness= 0.5, resonance= 0.4),
        "chorus":  TextureProfile(brightness= 0.4, warmth= 0.2, weight= 0.5, sharpness= 0.6, openness= 0.7, resonance= 0.5),
        "bridge":  TextureProfile(brightness=-0.1, warmth= 0.1, weight= 0.8, sharpness= 0.2, openness= 0.5, resonance= 0.6),
    },
}


def get_target_texture(mood: str, section: str) -> TextureProfile:
    """Get the ideal phonosemantic texture for a given mood × section."""
    mood_key = mood.lower()
    section_key = section.lower().replace(" ", "_")
    mood_map = MOOD_SECTION_TEXTURE.get(mood_key, MOOD_SECTION_TEXTURE["chill"])
    # Try exact match, then fallback to verse1
    return mood_map.get(section_key) or mood_map.get("verse1") or TextureProfile()


def texture_alignment_score(line: str, mood: str, section: str) -> float:
    """
    Score [0, 1]: how well the line's phonosemantic texture
    matches the ideal texture for this mood + section.
    """
    target = get_target_texture(mood, section)
    actual = analyze_line_texture(line)
    dist   = actual.distance_to(target)
    # Max distance in 6D texture space ≈ sqrt(6 * 4) ≈ 4.9
    max_dist = (6 * 4) ** 0.5
    return float(np.clip(1.0 - dist / max_dist, 0.0, 1.0))


# ── Word substitution suggestions ────────────────────────────────────────────
# For a given target texture, suggest darker/brighter/warmer synonyms.
# Powered by a small curated brightness-ranked synonym table.

SYNONYM_BANDS: dict[str, list[list[str]]] = {
    # Groups: [darkest] → [dark] → [neutral] → [bright] → [brightest]
    "light_dark": [
        ["abyss", "void", "hollow", "shadow", "grave"],      # very dark
        ["night", "cold", "grey", "silence", "fade"],         # dark
        ["path", "road", "time", "mind", "soul"],             # neutral
        ["flame", "spark", "glow", "rise", "shine"],          # bright
        ["blaze", "fire", "light", "gold", "star"],           # very bright
    ],
    "heavy_light": [
        ["stone", "weight", "burden", "chains", "bone"],      # very heavy
        ["ground", "still", "deep", "slow", "drown"],         # heavy
        ["move", "walk", "carry", "hold", "stay"],            # neutral
        ["float", "drift", "breathe", "glide", "flow"],       # light
        ["fly", "soar", "free", "lift", "air"],               # very light
    ],
}


def suggest_texture_substitution(
    word: str,
    target_brightness: float,  # -1 to 1
    target_weight: float,       # -1 to 1
) -> Optional[str]:
    """
    Suggest a word with better phonosemantic alignment to target texture.
    Returns None if no good substitute found.
    """
    # Determine target band index (0-4)
    brightness_band = min(4, int((target_brightness + 1) / 2 * 5))
    weight_band     = min(4, int((target_weight + 1) / 2 * 5))

    # Look in light/dark band
    band_words = (
        SYNONYM_BANDS["light_dark"][brightness_band]
        + SYNONYM_BANDS["heavy_light"][weight_band]
    )
    # Return first word from band that's different from input
    for bw in band_words:
        if bw.lower() != word.lower():
            return bw
    return None

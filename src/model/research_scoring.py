"""
Research-Backed Scoring Engine
================================
Implements scoring improvements derived directly from peer-reviewed neuroscience
and computational musicology research.

SOURCES
-------
[1] "Neural Correlates of Freestyle Rap" — Liu et al., PMC3498928
    Key finding: Creative lyrics generation uses TWO DISTINCT cognitive phases:
      Phase 1 (MPFC active, DLPFC suppressed): Divergent/generative — unconstrained,
               high novelty, semantic association, emotional access via amygdala
      Phase 2 (DLPFC re-engages): Convergent/evaluative — applies phonetic, rhythmic,
               narrative constraints; right hemisphere monitors resolution

[2] arXiv:2505.00035 — 40-year computational analysis of hip-hop lyrics (1980–2020)
    Key findings:
      • Multi-syllabic rhymes: 8.3% → 27.6% of all rhymes (biggest quality predictor)
      • Internal rhyme density: 0.42 → 0.84 per line after 2000
      • Complexity-success is QUADRATIC: optimal ~65th percentile, not maximum
      • Introspective content surged 246% (now dominant theme)
      • Technical complexity correlates with critical acclaim (r=0.42)
      • Vocabulary diversity increased 23.7% over 4 decades
      • PCA: 4 dimensions explain 68.3% of lyrical variance:
          PC1 = rhyme complexity, PC2 = sentiment negativity,
          PC3 = thematic scope (personal↔societal), PC4 = vocabulary diversity

IMPLEMENTED IMPROVEMENTS
-------------------------
1. Multi-syllabic rhyme scorer (polysyllabic matching)
2. Internal rhyme density per line
3. Quadratic complexity calibrator (target 65th percentile)
4. 8-bar temporal arc weighting (bars 7-8 = resolution priority)
5. Introspection / confessional content bonus
6. Vocabulary diversity signal (type-token ratio with recency weighting)
7. Motor-rhythm stress pattern alignment (iambic preference in verse, etc.)
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

try:
    import pronouncing
    _PRONOUNCING_OK = True
except ImportError:
    _PRONOUNCING_OK = False


# ── 1. Multi-syllabic (polysyllabic) rhyme scorer ────────────────────────────
# Source: [2] — Multi-syllabic rhymes are the single biggest predictor of
# lyrical quality in hip-hop, growing from 8.3% to 27.6% in top artists.
#
# "Breaking ground" rhymes with "making sound" → polysyllabic match
# "Ground" rhymes with "sound" → monosyllabic match
#
# Score: count how many consecutive syllables (from end) match phonetically.
# polysyllabic_rhyme_score = matched_syllables / max(syllables_a, syllables_b)

def _get_word_phonemes(word: str) -> list[str]:
    """Return ARPAbet phoneme list for a word, stripped of stress markers."""
    if not _PRONOUNCING_OK:
        return []
    try:
        phones_list = pronouncing.phones_for_word(word.lower().strip(".,!?'\""))
        if phones_list:
            return [p.rstrip("012") for p in phones_list[0].split()]
    except Exception:
        pass
    return []


def _get_vowel_cluster(phonemes: list[str]) -> list[str]:
    """Extract the final vowel + following consonants (rhyme nucleus + coda)."""
    VOWELS = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
    # Find last vowel, return from there
    for i in range(len(phonemes) - 1, -1, -1):
        if phonemes[i] in VOWELS:
            return phonemes[i:]
    return phonemes[-2:] if len(phonemes) >= 2 else phonemes


def polysyllabic_rhyme_score(line_a: str, line_b: str) -> float:
    """
    Score [0, 1] measuring the DEPTH of rhyme between two lines.
    - 0.0 = no rhyme
    - 0.25 = single phoneme match (weak)
    - 0.5 = single syllable end rhyme
    - 0.75 = two-syllable match ("breaking" / "making")
    - 1.0 = three+ syllable match ("electricity" / "infidelity")

    This is the metric that separates Eminem/Kendrick from average writers.
    """
    words_a = line_a.split()
    words_b = line_b.split()
    if not words_a or not words_b:
        return 0.0

    # Check last 1, 2, 3 words for rhyme
    max_score = 0.0
    for n_words in range(1, min(4, len(words_a) + 1, len(words_b) + 1)):
        segment_a = " ".join(words_a[-n_words:])
        segment_b = " ".join(words_b[-n_words:])

        phonemes_a = []
        phonemes_b = []
        for w in segment_a.split():
            phonemes_a.extend(_get_word_phonemes(w))
        for w in segment_b.split():
            phonemes_b.extend(_get_word_phonemes(w))

        if not phonemes_a or not phonemes_b:
            continue

        # Count matching phonemes from the end
        matched = 0
        for pa, pb in zip(reversed(phonemes_a), reversed(phonemes_b)):
            if pa == pb:
                matched += 1
            else:
                break

        if matched == 0:
            continue

        # Normalize: matched / average phoneme length
        avg_len = (len(phonemes_a) + len(phonemes_b)) / 2
        raw_score = matched / avg_len

        # Bonus for matching multiple words (polysyllabic)
        if n_words >= 2:
            raw_score = min(1.0, raw_score * 1.3)
        if n_words >= 3:
            raw_score = min(1.0, raw_score * 1.2)

        max_score = max(max_score, raw_score)

    return float(np.clip(max_score, 0.0, 1.0))


# ── 2. Internal rhyme density ─────────────────────────────────────────────────
# Source: [2] — Internal rhyme density doubled from 0.42 to 0.84 per line in
# the top 200 charting hip-hop artists after 2000.
# Target: >= 0.84 internal rhymes per line for top-tier output.

def internal_rhyme_score(line: str, context_lines: list[str] | None = None) -> float:
    """
    Score [0, 1] measuring internal rhyme density within a single line.
    Checks for rhyming word pairs WITHIN the same line.

    "I'm on a roll, I'm on a wave, I'm in control, I'm in my lane"
    → "roll"/"control", "wave"/"lane" = 2 internal rhyme pairs = high score
    """
    words = [w.strip(".,!?'\"").lower() for w in line.split() if w.strip(".,!?'\"")]
    if len(words) < 4:
        return 0.0

    STOP_WORDS = {'i','the','a','an','and','or','but','in','on','at','to','for',
                  'of','with','by','my','your','we','is','was','are','be','it',
                  'that','this','they','you','he','she','im','ive','its'}
    content_words = [(i, w) for i, w in enumerate(words) if w not in STOP_WORDS]

    if len(content_words) < 3:
        return 0.0

    rhyme_pairs = 0
    checked = 0
    for i in range(len(content_words)):
        for j in range(i + 1, len(content_words)):
            idx_i, w_i = content_words[i]
            idx_j, w_j = content_words[j]
            # Don't count adjacent words (would be too easy)
            if abs(idx_i - idx_j) < 2:
                continue
            phones_i = _get_word_phonemes(w_i)
            phones_j = _get_word_phonemes(w_j)
            if not phones_i or not phones_j:
                continue
            # Match last vowel+coda
            vc_i = _get_vowel_cluster(phones_i)
            vc_j = _get_vowel_cluster(phones_j)
            if vc_i and vc_j and vc_i == vc_j and w_i != w_j:
                rhyme_pairs += 1
            checked += 1

    if checked == 0:
        return 0.0

    # Normalize: 0.84 pairs/line is "excellent" target from research
    # Map: 0 pairs = 0, 0.84+ pairs = 0.8+, 2+ pairs = 1.0
    density = rhyme_pairs / max(len(words), 1)
    return float(np.clip(density / 0.15, 0.0, 1.0))   # 0.15 density ≈ 1.0 score


# ── 3. Quadratic complexity calibrator ───────────────────────────────────────
# Source: [2] — Complexity vs commercial success is QUADRATIC.
# Optimal complexity is ~65th percentile — not maximum.
# Too simple = boring, too complex = inaccessible.
#
# Complexity = f(vocabulary_diversity, rhyme_density, syllable_density, multi_syllabic_ratio)

def complexity_score(line: str, song_lines: list[str] | None = None) -> float:
    """
    Score [0, 1] where 1.0 = optimal complexity (65th percentile target).
    Penalizes both extremes: too simple OR too dense.
    """
    words = line.lower().split()
    if not words:
        return 0.0

    # Vocabulary diversity (type-token ratio)
    unique_ratio = len(set(words)) / len(words)

    # Syllable density (syllables per word)
    total_syllables = 0
    try:
        import pronouncing
        for w in words[:10]:  # sample for speed
            phones = pronouncing.phones_for_word(w.strip(".,!?'\""))
            if phones:
                total_syllables += pronouncing.syllable_count(phones[0])
            else:
                total_syllables += max(1, len(w) // 3)
    except Exception:
        total_syllables = len(words)  # fallback: 1 syl per word

    avg_syllables_per_word = total_syllables / max(len(words), 1)

    # Multi-syllabic word ratio (words with 3+ syllables)
    long_words = sum(1 for w in words if len(w) >= 8)
    multi_ratio = long_words / max(len(words), 1)

    # Raw complexity (0-1 scale)
    raw = (unique_ratio * 0.4 + avg_syllables_per_word / 4 * 0.4 + multi_ratio * 0.2)
    raw = float(np.clip(raw, 0.0, 1.0))

    # Apply QUADRATIC calibration: peak at 0.65, penalize above and below
    # Gaussian centered at 0.65 with σ=0.20
    optimal = 0.65
    sigma = 0.20
    calibrated = float(np.exp(-((raw - optimal) ** 2) / (2 * sigma ** 2)))
    return float(np.clip(calibrated, 0.0, 1.0))


# ── 4. 8-bar temporal arc weighting ──────────────────────────────────────────
# Source: [1] — Neural activity shifts from left-hemisphere generative
# (bars 1-6) to right-hemisphere monitoring/resolution (bars 7-8).
# The final 2 lines of every 8-line section have SPECIAL COGNITIVE WEIGHT.
# Line 7 should maximize tension; line 8 should provide resolution.

def bar_position_weights(line_idx: int, total_lines: int = 8) -> dict[str, float]:
    """
    Returns target weights for scoring at this bar position.
    Line indices are 0-based within an 8-bar section.
    """
    position = line_idx % 8  # position within 8-bar cycle

    if position < 6:
        # Bars 1-6: generative phase — favor creativity, novelty, hook DNA
        return {
            "novelty":      1.2,
            "hook_dna":     1.1,
            "tension":      1.0,   # tension should be building
            "resolution":   0.5,   # don't resolve yet
            "phonetic":     1.0,
        }
    elif position == 6:
        # Bar 7: peak tension — maximize tension buildup
        return {
            "novelty":      0.8,
            "hook_dna":     1.3,   # this is where the hook DNA must fire
            "tension":      1.5,   # maximum tension before resolution
            "resolution":   0.3,
            "phonetic":     1.2,
        }
    else:
        # Bar 8 (final line): resolution phase — must resolve rhyme + emotion
        return {
            "novelty":      0.7,
            "hook_dna":     1.0,
            "tension":      0.3,   # tension should resolve
            "resolution":   1.8,   # this is the payoff line
            "phonetic":     1.5,   # rhyme MUST land on the last line
        }


def temporal_arc_score(
    line: str,
    line_idx: int,
    rhymes_with_target: bool,
    tension_state: float,
) -> float:
    """
    Score [0, 1] based on how well the line serves its temporal position.
    """
    weights = bar_position_weights(line_idx)
    position = line_idx % 8

    # Final line: MUST rhyme and MUST reduce tension
    if position == 7:
        rhyme_ok = 1.0 if rhymes_with_target else 0.0
        tension_ok = 1.0 - tension_state   # want low tension at resolution
        return float(0.6 * rhyme_ok + 0.4 * tension_ok)

    # Penultimate line: tension should be high
    if position == 6:
        return float(np.clip(tension_state * 1.2, 0.0, 1.0))

    # Middle bars: just make sure tension is building
    expected_tension = position / 6 * 0.7  # linear ramp from 0 to 0.7
    tension_alignment = 1.0 - abs(tension_state - expected_tension)
    return float(np.clip(tension_alignment, 0.0, 1.0))


# ── 5. Introspection / confessional bonus ────────────────────────────────────
# Source: [2] — Introspective content surged 246% in hip-hop (7.6% → 26.3%).
# First-person confessional writing is now the dominant mode in commercially
# successful music. Songs that make people feel "this is exactly how I feel"
# are more likely to go viral.

INTROSPECTION_MARKERS = {
    # First-person singular vulnerability
    "i feel", "i felt", "i been", "i know", "i never", "i always", "i still",
    "i can't", "i don't", "i won't", "i couldn't", "i wouldn't", "i used to",
    "i remember", "i forgot", "i need", "i miss", "i lost", "i found",
    # Confessional openings
    "they don't know", "nobody knows", "truth is", "honestly", "real talk",
    "i admit", "i confess", "i'll be honest", "if i'm honest",
    # Universal truth + personal
    "we all", "everyone", "all of us", "nobody told me",
    # Vulnerability signals
    "alone", "scared", "afraid", "broken", "numb", "lost", "empty", "hollow",
}

def introspection_score(line: str) -> float:
    """
    Score [0, 1] measuring confessional/introspective content.
    Higher = more likely to make listeners say "this is exactly me."
    """
    lower = line.lower()
    hits = sum(1 for marker in INTROSPECTION_MARKERS if marker in lower)
    # Also check for first-person singular at start of line
    if lower.strip().startswith(("i ", "i'm ", "i've ", "i'll ", "i'd ")):
        hits += 1
    return float(np.clip(hits / 3.0, 0.0, 1.0))


# ── 6. Vocabulary diversity (recency-weighted) ───────────────────────────────
# Source: [2] — Vocabulary diversity increased 23.7% in top hip-hop over 40 years.
# Simple TTR (type-token ratio) penalizes longer texts. We use recency-weighted TTR.

def vocabulary_novelty_score(line: str, recent_lines: list[str]) -> float:
    """
    Score [0, 1] measuring how many NEW content words this line introduces
    vs what's been used in the last 8 lines (recency window).

    Prevents: repetitive vocabulary that makes songs feel cheap.
    """
    STOP = {'i','the','a','an','and','or','but','in','on','at','to','for','of',
            'with','by','my','your','we','is','was','are','be','it','that',
            'this','they','you','he','she','im','ive','its','me','us','them'}

    current_words = {w.lower().strip(".,!?'\"") for w in line.split()
                     if w.lower().strip(".,!?'\"") not in STOP}

    if not current_words:
        return 0.5

    # Recency-weighted recent vocabulary (more recent = higher penalty for repeat)
    recent_vocab: dict[str, float] = {}
    for age, prev_line in enumerate(reversed(recent_lines[-8:])):
        weight = 1.0 - age * 0.1  # decay over 8 lines
        for w in prev_line.lower().split():
            clean = w.strip(".,!?'\"")
            if clean not in STOP and clean:
                recent_vocab[clean] = max(recent_vocab.get(clean, 0), weight)

    if not recent_vocab:
        return 1.0

    # Penalize overlap with recent vocab
    overlap_weight = sum(recent_vocab.get(w, 0) for w in current_words)
    max_possible = len(current_words)
    overlap_ratio = overlap_weight / max_possible
    return float(np.clip(1.0 - overlap_ratio * 0.8, 0.0, 1.0))


# ── 7. Rhythmic stress alignment ─────────────────────────────────────────────
# Source: [1] — Motor areas activate simultaneously with language areas during
# rap improvisation. Rhythm is not just timing — it's embodied motor pattern.
# The brain thinks of lyrics as MOVEMENT patterns.
#
# Iambic: weak-STRONG (most natural in English speech)
# Trochaic: STRONG-weak (forceful, commanding — common in hooks)
# Spondaic: STRONG-STRONG (intense, emphasis — one per line maximum)
# Anapestic: weak-weak-STRONG (flowing, building — common in verses)

def _get_stress_pattern(word: str) -> list[int]:
    """Returns stress pattern for a word: 1=stressed, 0=unstressed."""
    if not _PRONOUNCING_OK:
        return [1]
    try:
        import pronouncing
        phones_list = pronouncing.phones_for_word(word.lower().strip(".,!?'\""))
        if phones_list:
            stresses = [int(p[-1]) for p in phones_list[0].split() if p[-1].isdigit()]
            return stresses if stresses else [1]
    except Exception:
        pass
    return [1]


SECTION_STRESS_TARGETS: dict[str, str] = {
    "verse1":  "iambic",     # natural speech flow
    "verse2":  "iambic",
    "chorus":  "trochaic",   # forceful, commanding, memorable
    "hook":    "trochaic",
    "bridge":  "anapestic",  # flowing, building toward release
    "intro":   "iambic",
    "outro":   "iambic",
    "pre_chorus": "anapestic",
}

def stress_alignment_score(line: str, section: str = "verse1") -> float:
    """
    Score [0, 1] measuring how well the line's stress pattern fits the section.
    Iambic verse: "i BEEN through FIRE, i WALKED through FLAME"
    Trochaic chorus: "TURN IT UP, FEEL the HEAT, BURNING BRIGHT"
    """
    words = [w.strip(".,!?'\"") for w in line.split() if w.strip(".,!?'\"")]
    if not words:
        return 0.5

    target = SECTION_STRESS_TARGETS.get(section.lower(), "iambic")

    # Build full stress sequence
    full_pattern = []
    for w in words[:8]:  # first 8 words enough
        full_pattern.extend(_get_stress_pattern(w))

    if len(full_pattern) < 4:
        return 0.5

    # Count alternations
    alternations = sum(
        1 for i in range(1, len(full_pattern))
        if full_pattern[i] != full_pattern[i-1]
    )
    alternation_ratio = alternations / max(len(full_pattern) - 1, 1)

    if target == "iambic":
        # Iambic: want alternating, starting on unstressed
        starts_unstressed = 1.0 if full_pattern[0] == 0 else 0.3
        return float(np.clip(alternation_ratio * 0.7 + starts_unstressed * 0.3, 0.0, 1.0))

    elif target == "trochaic":
        # Trochaic: alternating, starting on STRESSED
        starts_stressed = 1.0 if full_pattern[0] == 1 else 0.3
        return float(np.clip(alternation_ratio * 0.7 + starts_stressed * 0.3, 0.0, 1.0))

    elif target == "anapestic":
        # Anapestic: prefer groups of 2 unstressed + 1 stressed
        # Look for 0,0,1 patterns
        anapest_count = sum(
            1 for i in range(len(full_pattern) - 2)
            if full_pattern[i:i+3] == [0, 0, 1]
        )
        return float(np.clip(anapest_count / max(len(full_pattern) // 3, 1), 0.0, 1.0))

    return 0.5


# ── Combined research score ───────────────────────────────────────────────────

def research_score(
    line: str,
    line_idx: int,
    section: str,
    recent_lines: list[str],
    rhymes_with_target: bool,
    tension_state: float,
    previous_line: Optional[str] = None,
) -> dict[str, float]:
    """
    Compute all 7 research-backed scores for a candidate line.
    Returns a dict of individual scores + weighted total.

    Weights based on research findings:
      - multi_syllabic_rhyme: 0.20  ← biggest predictor in [2]
      - internal_rhyme:       0.15  ← doubled in top artists
      - complexity:           0.15  ← quadratic optimum
      - temporal_arc:         0.15  ← 8-bar cognitive structure
      - introspection:        0.12  ← surged 246%
      - vocab_novelty:        0.13  ← diversity signal
      - stress_alignment:     0.10  ← motor-rhythm coupling
    """
    # Polysyllabic rhyme with previous line (or empty score if no previous)
    poly_score = polysyllabic_rhyme_score(line, previous_line) if previous_line else 0.5

    internal = internal_rhyme_score(line)
    complex_  = complexity_score(line, recent_lines)
    temporal  = temporal_arc_score(line, line_idx, rhymes_with_target, tension_state)
    introspect = introspection_score(line)
    vocab     = vocabulary_novelty_score(line, recent_lines)
    stress    = stress_alignment_score(line, section)

    total = (
        0.20 * poly_score
        + 0.15 * internal
        + 0.15 * complex_
        + 0.15 * temporal
        + 0.12 * introspect
        + 0.13 * vocab
        + 0.10 * stress
    )

    return {
        "polysyllabic_rhyme": round(poly_score, 3),
        "internal_rhyme":     round(internal, 3),
        "complexity":         round(complex_, 3),
        "temporal_arc":       round(temporal, 3),
        "introspection":      round(introspect, 3),
        "vocab_novelty":      round(vocab, 3),
        "stress_alignment":   round(stress, 3),
        "total":              round(total, 3),
    }

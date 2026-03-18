"""
Dopamine Arc Engine
===================
Models the neurological tension-release cycle that makes songs physically
addictive and causes goosebumps (piloerection / frisson).

THE NEUROSCIENCE
----------------
When we listen to music, the brain's reward system (nucleus accumbens) releases
dopamine in anticipation of a musical resolution — not at the resolution itself,
but BEFORE it. This is why a great hook feels inevitable in retrospect:
the brain predicted it and got flooded with dopamine.

The "goosebump moment" (frisson) occurs when:
  1. Tension has been building (arousal high, resolution low)
  2. The pattern is about to resolve in an expected way
  3. The resolution arrives — but slightly UNEXPECTED in a satisfying way
     (an unexpected rhyme, a metaphor that reframes everything,
      a sudden drop in energy, an emotional truth that hits hard)

WHAT THIS ENGINE DOES
---------------------
1. Models the tension curve across a song as a float time-series
2. Identifies the optimal placement of "release" moments (chorus hits,
   bridge breakdowns, the final line of a verse)
3. Scores candidate lines by their DOPAMINE POTENTIAL:
   - High potential: resolves accumulated tension, unexpected but satisfying
   - Low potential: continues without resolution, or resolves too early
4. Detects UNIVERSAL HOOK DNA: structural patterns found in viral songs globally
   - The Confession+Brag ("I had nothing / now I got everything")
   - The Reframe ("This ain't heartbreak / this is freedom")
   - The Universal Truth+Twist ("Love fades / but the scars stay beautiful")
   - The Call+Echo (repeat phrase, new meaning)
   - The Elevation ("From the ground / now we touching the sky")

TENSION MODEL
-------------
Tension at each line = f(
  unresolved_rhyme_debt,   ← rhyme scheme set up but not resolved yet
  arousal_level,           ← energy of recent lines
  emotional_distance,      ← how far we are from emotional target
  syllable_density,        ← compressed = higher tension
  negative_valence,        ← dark content = tension
  repetition_buildup,      ← anaphora creates tension that demands release
)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.data.phoneme_annotator import annotate_line
from src.data.rhyme_labeler import rhymes
from src.data.valence_scorer import score_line


# ── Hook DNA patterns ─────────────────────────────────────────────────────────

@dataclass
class HookPattern:
    name: str
    description: str
    example: str
    weight: float   # how strongly this pattern predicts virality (0-1)


HOOK_PATTERNS: list[HookPattern] = [
    HookPattern(
        name="confession_brag",
        description="Vulnerability followed by power assertion",
        example="I had nothing left to lose / now I got the world at my feet",
        weight=0.9,
    ),
    HookPattern(
        name="reframe",
        description="Reinterprets a negative as a positive or vice versa",
        example="This ain't heartbreak / this is freedom starting now",
        weight=0.85,
    ),
    HookPattern(
        name="elevation",
        description="Movement from low state to high state",
        example="From the bottom of the dark / now we're touching the light",
        weight=0.8,
    ),
    HookPattern(
        name="universal_truth_twist",
        description="States a universal truth then makes it personal",
        example="Everyone falls apart / but only some of us rebuild",
        weight=0.75,
    ),
    HookPattern(
        name="call_echo",
        description="Repeats a phrase with shifted meaning on second occurrence",
        example="They said I wouldn't make it / I said I wouldn't make it — now look",
        weight=0.8,
    ),
    HookPattern(
        name="paradox",
        description="Two contradictory truths held simultaneously",
        example="I'm the most alone when I'm surrounded by everyone",
        weight=0.7,
    ),
    HookPattern(
        name="sensory_anchor",
        description="Abstract emotion grounded in a specific physical sensation",
        example="I can still smell the rain the night you left",
        weight=0.75,
    ),
    HookPattern(
        name="time_compression",
        description="Collapses or expands time for emotional effect",
        example="In a second it was over / in a lifetime I'll remember",
        weight=0.7,
    ),
    HookPattern(
        name="societal_mirror",
        description="Speaks to a shared cultural truth about a generation/place",
        example="They built a city and forgot to put a home in it",
        weight=0.8,
    ),
    HookPattern(
        name="defiance",
        description="Direct address asserting power against an opposing force",
        example="You said I'd never / watch me",
        weight=0.85,
    ),
]

# ── Pattern detectors ─────────────────────────────────────────────────────────

def _has_contrast_markers(line: str) -> bool:
    """Checks for contrast words that signal reframe/paradox patterns."""
    contrast = {
        "but", "yet", "still", "though", "although", "while", "however",
        "despite", "even", "never", "always", "only", "just", "except",
        "now", "anymore", "suddenly", "instead",
    }
    words = set(line.lower().split())
    return bool(words & contrast)


def _has_elevation_markers(line: str) -> bool:
    """Checks for rise/fall vocabulary signaling elevation pattern."""
    up   = {"rise", "fly", "climb", "up", "above", "sky", "top", "high", "crown", "shine", "above", "heaven", "soar"}
    down = {"bottom", "ground", "floor", "fall", "low", "down", "beneath", "below", "nothing"}
    words = set(line.lower().split())
    return bool(words & up) and bool(words & down)


def _has_defiance_markers(line: str) -> bool:
    """Checks for patterns like 'they said X / now look' or 'watch me'."""
    defiance_phrases = [
        r"\bthey said\b", r"\bwatch me\b", r"\bnow look\b", r"\btold me\b",
        r"\bnever thought\b", r"\bprove\b", r"\bdoubt\b", r"\bnevermind\b",
    ]
    lower = line.lower()
    return any(re.search(p, lower) for p in defiance_phrases)


def _has_sensory_anchor(line: str) -> bool:
    """Checks for specific sensory words that ground abstract emotion."""
    sensory = {
        "smell", "touch", "feel", "taste", "hear", "see", "saw", "felt",
        "sound", "voice", "rain", "hands", "eyes", "mouth", "skin",
        "cold", "warm", "hot", "light", "dark",
    }
    words = set(line.lower().split())
    return bool(words & sensory)


def _has_universal_truth(line: str) -> bool:
    """Checks for phrases starting with universal statements."""
    universal_starters = [
        r"\beveryone\b", r"\bwe all\b", r"\bno one\b", r"\bthe world\b",
        r"\blife is\b", r"\btime\b", r"\bwe were\b", r"\bnobody\b",
        r"\bsome\w* say\b",
    ]
    lower = line.lower()
    return any(re.search(p, lower) for p in universal_starters)


def _has_time_markers(line: str) -> bool:
    """Checks for time compression/expansion markers."""
    time_words = {
        "second", "moment", "forever", "lifetime", "memory", "remember",
        "still", "always", "never", "yesterday", "tomorrow", "again",
        "once", "years", "nights", "days",
    }
    words = set(line.lower().split())
    return bool(words & time_words)


def detect_hook_patterns(line: str, previous_line: Optional[str] = None) -> list[str]:
    """
    Detect which hook DNA patterns are present in this line.
    Returns list of pattern names.
    """
    detected = []

    if _has_contrast_markers(line):
        detected.append("reframe")

    if _has_elevation_markers(line):
        detected.append("elevation")

    if _has_defiance_markers(line):
        detected.append("defiance")

    if _has_sensory_anchor(line):
        detected.append("sensory_anchor")

    if _has_universal_truth(line):
        detected.append("universal_truth_twist")

    if _has_time_markers(line):
        detected.append("time_compression")

    # Call+echo: current line repeats key phrase from previous line with modification
    if previous_line:
        prev_words = set(previous_line.lower().split())
        curr_words = set(line.lower().split())
        overlap_ratio = len(prev_words & curr_words) / max(len(prev_words), 1)
        if 0.3 < overlap_ratio < 0.7:  # partial repeat = call+echo
            detected.append("call_echo")

    return list(set(detected))


def hook_dna_score(line: str, previous_line: Optional[str] = None) -> float:
    """
    Score [0, 1] representing the 'hook DNA' strength of this line.
    Higher = more likely to be a memorable, viral moment.
    """
    patterns = detect_hook_patterns(line, previous_line)
    if not patterns:
        return 0.0
    # Weight by pattern virality weights
    pattern_weights = {p.name: p.weight for p in HOOK_PATTERNS}
    total_weight = sum(pattern_weights.get(p, 0.5) for p in patterns)
    # Normalize: max realistic is ~3 patterns
    return float(np.clip(total_weight / 2.5, 0.0, 1.0))


# ── Tension curve model ───────────────────────────────────────────────────────

@dataclass
class TensionState:
    """Current tension level and its components."""
    total: float             = 0.0   # [0, 1]
    rhyme_debt: float        = 0.0   # unresolved rhyme scheme
    arousal: float           = 0.0   # recent line energy
    emotional_distance: float = 0.0  # distance from emotional target
    syllable_density: float  = 0.0   # compressed lines = tension
    anaphora_buildup: float  = 0.0   # repeated structures demand release


class TensionCurve:
    """
    Tracks tension across a song's lines.
    Tension rises with arousal/negativity/unresolved rhymes,
    falls at resolution moments (rhyme completion, emotional peak).
    """

    def __init__(self, initial: float = 0.2):
        self.current = initial
        self.history: list[float] = [initial]
        self.last_lines: list[str] = []
        self.resolved_rhymes: list[str] = []
        self.pending_rhyme: Optional[str] = None

    def update(self, line: str, rhyme_target: Optional[str] = None) -> float:
        """
        Update tension after accepting a line.
        Returns the new tension level.
        """
        ann     = annotate_line(line)
        emotion = score_line(line)

        # Component 1: rhyme debt
        if rhyme_target and ann.end_phoneme:
            if rhymes(ann.end_phoneme, rhyme_target):
                # Rhyme resolved → tension drop
                rhyme_delta = -0.15
            else:
                # Rhyme pending → tension builds
                rhyme_delta = +0.08
        else:
            rhyme_delta = 0.0

        # Component 2: arousal from current line
        arousal_delta = (emotion.arousal - 0.5) * 0.10

        # Component 3: negative valence = tension
        valence_delta = (-emotion.valence) * 0.06

        # Component 4: syllable density (more syllables = compressed = tense)
        density = ann.total_syllables / 14.0  # normalize by typical max
        density_delta = (density - 0.7) * 0.05

        # Component 5: anaphora (repeated opening word = buildup)
        if len(self.last_lines) >= 2:
            first_words = [l.split()[0].lower() if l.split() else "" for l in self.last_lines[-2:]]
            curr_first = line.split()[0].lower() if line.split() else ""
            if curr_first and all(w == curr_first for w in first_words):
                anaphora_delta = +0.05
            else:
                anaphora_delta = 0.0
        else:
            anaphora_delta = 0.0

        # Decay: tension slowly resolves on its own
        decay = -0.02

        delta = rhyme_delta + arousal_delta + valence_delta + density_delta + anaphora_delta + decay
        self.current = float(np.clip(self.current + delta, 0.0, 1.0))
        self.history.append(self.current)
        self.last_lines.append(line)
        return self.current

    def reset_for_section(self, section: str):
        """Adjust tension baseline for new section."""
        targets = {
            "intro": 0.2, "verse1": 0.35, "verse2": 0.45,
            "chorus": 0.1, "chorus2": 0.1, "bridge": 0.7, "outro": 0.15,
        }
        self.current = targets.get(section.lower(), 0.3)


# ── Goosebump predictor ───────────────────────────────────────────────────────

def goosebump_potential(
    line: str,
    tension_state: float,            # current accumulated tension (0-1)
    previous_line: Optional[str] = None,
    mood: str = "dark",
) -> float:
    """
    Predict the probability [0, 1] that this line causes a goosebump moment
    (musical frisson / piloerection).

    The model: goosebumps occur when
      1. High accumulated tension exists (tension_state > 0.5)
      2. The line provides unexpected-but-satisfying resolution
      3. The line uses hook DNA patterns
      4. The line creates an emotional jump (sharp valence change)
      5. The phonosemantic texture SHIFTS (e.g., dark verse → bright chorus moment)

    This is the "Chills Score" — what separates a good line from a legendary one.
    """
    emotion = score_line(line)
    hook_score = hook_dna_score(line, previous_line)

    # Factor 1: tension × resolution potential
    # High tension + line that resolves it = maximum impact
    is_resolving = emotion.valence > 0.3 and emotion.arousal > 0.5
    tension_factor = tension_state * (1.0 if is_resolving else 0.3)

    # Factor 2: emotional jump (if we have previous line data)
    if previous_line:
        prev_emotion = score_line(previous_line)
        val_jump = abs(emotion.valence - prev_emotion.valence)
        aro_jump = abs(emotion.arousal - prev_emotion.arousal)
        jump_factor = np.clip((val_jump + aro_jump) / 2.0, 0.0, 1.0)
    else:
        jump_factor = 0.0

    # Factor 3: hook DNA
    hook_factor = hook_score

    # Factor 4: phonosemantic texture shift
    try:
        from src.model.phonosemantic import analyze_line_texture
        curr_texture = analyze_line_texture(line)
        texture_intensity = abs(curr_texture.brightness) + abs(curr_texture.sharpness)
        texture_factor = float(np.clip(texture_intensity / 2.0, 0.0, 1.0))
    except Exception:
        texture_factor = 0.0

    # Weighted combination
    goosebump = (
        0.35 * tension_factor
        + 0.25 * jump_factor
        + 0.25 * hook_factor
        + 0.15 * texture_factor
    )
    return float(np.clip(goosebump, 0.0, 1.0))


# ── Full song analysis ────────────────────────────────────────────────────────

def analyze_song_dopamine(
    sections: dict[str, list[str]],  # {section_name: [lines]}
    mood: str = "dark",
) -> dict:
    """
    Analyze the complete dopamine arc of a song.
    Identifies peak goosebump moments, tension curve, and hook DNA per line.

    Returns a rich analysis dict that can be shown to artists as:
    "Here are your strongest moments" and "Here's where your song needs work"
    """
    curve = TensionCurve()
    all_results = {}
    global_best_line = {"line": "", "section": "", "score": -1.0}

    for section_name, lines in sections.items():
        curve.reset_for_section(section_name)
        section_results = []
        prev_line = None

        for line in lines:
            tension_before = curve.current
            curve.update(line)
            tension_after = curve.current

            hook_patterns = detect_hook_patterns(line, prev_line)
            gp = goosebump_potential(line, tension_before, prev_line, mood)

            entry = {
                "line": line,
                "tension_before": round(tension_before, 3),
                "tension_after":  round(tension_after, 3),
                "tension_delta":  round(tension_after - tension_before, 3),
                "goosebump_potential": round(gp, 3),
                "hook_patterns": hook_patterns,
                "is_peak": gp > 0.6,
            }
            section_results.append(entry)

            if gp > global_best_line["score"]:
                global_best_line = {"line": line, "section": section_name, "score": gp}

            prev_line = line

        all_results[section_name] = section_results

    # Section summaries
    section_summaries = {}
    for sec, results in all_results.items():
        scores = [r["goosebump_potential"] for r in results]
        section_summaries[sec] = {
            "avg_goosebump": round(float(np.mean(scores)), 3) if scores else 0.0,
            "peak_line": max(results, key=lambda x: x["goosebump_potential"])["line"] if results else "",
            "peak_score": round(max(r["goosebump_potential"] for r in results), 3) if results else 0.0,
            "tension_curve": [r["tension_after"] for r in results],
        }

    return {
        "lines": all_results,
        "section_summaries": section_summaries,
        "peak_moment": global_best_line,
        "tension_history": curve.history,
        "mood": mood,
    }

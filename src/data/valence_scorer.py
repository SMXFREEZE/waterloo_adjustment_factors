"""
Emotional valence + arousal scoring.
Uses TextBlob for a lightweight local scorer.
When a fine-tuned RoBERTa is available (path set via VALENCE_MODEL_PATH),
it will be used instead for higher accuracy.

Output per line:
  valence: float in [-1, +1]  (negative → positive)
  arousal: float in [0, 1]    (calm → intense)

Arc trajectory tokens are assigned per section based on valence trend.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

from textblob import TextBlob


VALENCE_MODEL_PATH = os.getenv("VALENCE_MODEL_PATH", "")


@dataclass
class LineEmotion:
    text: str
    valence: float   # -1 to +1
    arousal: float   # 0 to 1


def _textblob_score(text: str) -> LineEmotion:
    blob = TextBlob(text)
    valence = blob.sentiment.polarity       # -1 to +1
    subjectivity = blob.sentiment.subjectivity  # 0 to 1 — proxy for arousal
    # Arousal heuristic: high subjectivity + exclamation/caps → high arousal
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    exclaim = text.count("!") + text.count("?")
    arousal = min(1.0, subjectivity * 0.6 + caps_ratio * 0.2 + exclaim * 0.05)
    return LineEmotion(text=text, valence=round(valence, 4), arousal=round(arousal, 4))


def _roberta_score(text: str, pipe) -> LineEmotion:
    """Use fine-tuned sentiment pipeline if available."""
    result = pipe(text, truncation=True, max_length=128)[0]
    label = result["label"].upper()
    score = result["score"]
    valence = score if "POS" in label else -score
    arousal = score  # rough proxy
    return LineEmotion(text=text, valence=round(valence, 4), arousal=round(arousal, 4))


_roberta_pipe = None


def _load_roberta():
    global _roberta_pipe
    if _roberta_pipe is None and VALENCE_MODEL_PATH:
        try:
            from transformers import pipeline
            _roberta_pipe = pipeline(
                "text-classification",
                model=VALENCE_MODEL_PATH,
                device=-1,
            )
        except Exception as e:
            print(f"[valence] Could not load RoBERTa: {e}. Using TextBlob.")
    return _roberta_pipe


def score_line(text: str) -> LineEmotion:
    pipe = _load_roberta()
    if pipe:
        return _roberta_score(text, pipe)
    return _textblob_score(text)


def score_lyrics(lyrics: str) -> list[LineEmotion]:
    lines = [l.strip() for l in lyrics.splitlines() if l.strip()]
    return [score_line(l) for l in lines]


# ── Arc trajectory ─────────────────────────────────────────────────────────────

ARC_TOKENS = ["[SETUP]", "[BUILD]", "[RELEASE]", "[REFRAME]", "[PEAK]", "[OUTRO]"]


def assign_arc_token(
    emotions: list[LineEmotion],
    section_idx: int,
    total_sections: int,
) -> str:
    """Assign an arc token to a section based on its position and valence trend."""
    if not emotions:
        return "[SETUP]"

    avg_valence = sum(e.valence for e in emotions) / len(emotions)
    avg_arousal = sum(e.arousal for e in emotions) / len(emotions)
    progress = section_idx / max(total_sections - 1, 1)

    if progress < 0.15:
        return "[SETUP]"
    elif progress < 0.4 and avg_arousal < 0.5:
        return "[BUILD]"
    elif progress < 0.6 and avg_arousal >= 0.4:
        return "[RELEASE]"
    elif progress < 0.75:
        return "[REFRAME]"
    elif avg_arousal >= 0.6:
        return "[PEAK]"
    else:
        return "[OUTRO]"


def compute_song_arc(lyrics_by_section: list[list[str]]) -> list[str]:
    """
    Given lyrics split by section (list of lists of lines),
    return an arc token for each section.
    """
    scored: list[list[LineEmotion]] = [
        [score_line(l) for l in section] for section in lyrics_by_section
    ]
    return [
        assign_arc_token(emotions, i, len(scored))
        for i, emotions in enumerate(scored)
    ]


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    lines = [
        "I been movin' in silence, they can't feel my weight",
        "Every step I take, yeah I'm moving with fate",
        "They say the game is cold but I turn up the heat",
        "Diamonds on my wrist while I dance to the beat",
    ]
    print("Line emotion scores:")
    for em in score_lyrics("\n".join(lines)):
        print(f"  [{em.valence:+.2f} val, {em.arousal:.2f} aro] {em.text}")

"""
Rhyme scheme detection.
Analyses end-phoneme similarity between lines and labels the scheme:
AABB, ABAB, ABCB, AAAA, and free verse.

Also computes:
- internal rhyme density (rhymes within a line)
- multi-syllable rhyme flag
"""

from typing import Optional
import re
from src.data.phoneme_annotator import annotate_line, LineAnnotation


# ── Phoneme similarity ─────────────────────────────────────────────────────────

def phoneme_edit_distance(a: Optional[str], b: Optional[str]) -> float:
    """Normalized edit distance on phoneme strings (vowel+coda)."""
    if a is None or b is None:
        return 1.0
    toks_a = a.split()
    toks_b = b.split()
    # dp
    la, lb = len(toks_a), len(toks_b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if toks_a[i - 1].rstrip("012") == toks_b[j - 1].rstrip("012") else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[la][lb] / max(la, lb, 1)


def rhymes(ep_a: Optional[str], ep_b: Optional[str], threshold: float = 0.35) -> bool:
    return phoneme_edit_distance(ep_a, ep_b) <= threshold


# ── Scheme detection ───────────────────────────────────────────────────────────

def _assign_labels(end_phonemes: list[Optional[str]]) -> list[str]:
    """Greedily assign letter labels to lines based on rhyme similarity."""
    labels: list[str] = []
    groups: list[tuple[str, Optional[str]]] = []  # (label, representative_phoneme)
    letter_idx = 0

    for ep in end_phonemes:
        matched = None
        for label, rep_ep in groups:
            if rhymes(ep, rep_ep):
                matched = label
                break
        if matched:
            labels.append(matched)
        else:
            new_label = chr(ord("A") + letter_idx)
            letter_idx += 1
            groups.append((new_label, ep))
            labels.append(new_label)

    return labels


def detect_scheme(lines: list[str]) -> dict:
    """
    Given a list of lyric lines, return:
      - scheme_str: e.g. "AABB" or "ABAB"
      - scheme_type: AABB / ABAB / ABCB / AAAA / free
      - rhyme_density: fraction of lines that rhyme with at least one other
      - multisyl_rhyme: True if any rhyming pair shares 2+ syllable overlap
    """
    annotations = [annotate_line(l) for l in lines]
    end_phonemes = [a.end_phoneme for a in annotations]
    labels = _assign_labels(end_phonemes)
    scheme_str = "".join(labels)

    # Classify
    scheme_type = "free"
    n = len(labels)
    if n >= 4:
        chunk = labels[:4]
        if chunk[0] == chunk[1] and chunk[2] == chunk[3]:
            scheme_type = "AABB"
        elif chunk[0] == chunk[2] and chunk[1] == chunk[3]:
            scheme_type = "ABAB"
        elif chunk[1] == chunk[3] and chunk[0] != chunk[2]:
            scheme_type = "ABCB"
        elif len(set(chunk)) == 1:
            scheme_type = "AAAA"

    # Rhyme density
    label_counts: dict[str, int] = {}
    for l in labels:
        label_counts[l] = label_counts.get(l, 0) + 1
    rhyming_lines = sum(1 for l in labels if label_counts[l] > 1)
    rhyme_density = rhyming_lines / max(n, 1)

    # Multi-syllable rhyme (simple: end_phoneme has 3+ tokens)
    multisyl_rhyme = any(
        ep is not None and len(ep.split()) >= 3
        for ep in end_phonemes
    )

    return {
        "scheme_str": scheme_str,
        "scheme_type": scheme_type,
        "rhyme_density": round(rhyme_density, 3),
        "multisyl_rhyme": multisyl_rhyme,
        "line_labels": labels,
    }


def internal_rhyme_density(line: str) -> float:
    """Fraction of word pairs within a line that rhyme (end-phoneme match)."""
    words = line.split()
    if len(words) < 2:
        return 0.0
    from src.data.phoneme_annotator import get_word_phoneme, get_end_phoneme
    eps = []
    for w in words:
        wp = get_word_phoneme(w)
        eps.append(get_end_phoneme(wp.phonemes))

    pairs = [(i, j) for i in range(len(eps)) for j in range(i + 1, len(eps))]
    if not pairs:
        return 0.0
    rhyme_count = sum(1 for i, j in pairs if rhymes(eps[i], eps[j]))
    return round(rhyme_count / len(pairs), 3)


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    verses = [
        "I been movin' in silence, they can't feel my weight",
        "Every step I take, yeah I'm moving with fate",
        "They say the game is cold but I turn up the heat",
        "Diamonds on my wrist while I dance to the beat",
    ]
    result = detect_scheme(verses)
    print("Lines:")
    for line, label in zip(verses, result["line_labels"]):
        print(f"  [{label}] {line}")
    print(f"\nScheme     : {result['scheme_str']} ({result['scheme_type']})")
    print(f"Rhyme density : {result['rhyme_density']}")
    print(f"Multi-syl rhyme: {result['multisyl_rhyme']}")

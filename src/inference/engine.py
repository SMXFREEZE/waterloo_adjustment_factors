"""
Inference engine: constrained beam search for god-tier lyrics.

Per-line generation flow:
  1. Build context: [genre token + LoRA] + [style prefix] + [section/arc tokens] + [accepted lines]
  2. Generate beam_size=8 candidate lines
  3. Post-score each candidate:
     - phonetic constraint satisfaction (end-rhyme match)
     - syllable count match (hard filter)
     - novelty vs existing song lines
     - emotional valence fit to arc target
  4. Emit top-1 (auto) or top-3 (co-write mode)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.data.phoneme_annotator import annotate_line, LineAnnotation
from src.data.rhyme_labeler import rhymes
from src.data.valence_scorer import score_line, LineEmotion
from src.model.dual_tokenizer import PHONEME_TO_ID, word_to_phoneme_ids
from src.model.emotional_geometry import trajectory_fit_score, compute_song_arc
from src.model.phonosemantic import texture_alignment_score
from src.model.dopamine_arc import (
    goosebump_potential, hook_dna_score, TensionCurve, analyze_song_dopamine,
)


# ── Syllable counter (deterministic, no LLM) ─────────────────────────────────

def count_syllables(line: str) -> int:
    return annotate_line(line).total_syllables


# ── Song memory (KV context of accepted lines) ───────────────────────────────

@dataclass
class SongMemory:
    genre: str
    mood: str = "dark"                      # dark / hype / romantic / chill / sad / epic
    style_vec: Optional[np.ndarray] = None  # (128,)
    sections: list[tuple[str, str]] = field(default_factory=list)  # (arc_token, section_name)
    accepted_lines: list[str] = field(default_factory=list)
    rhyme_scheme: str = "AABB"              # AABB / ABAB / ABCB / free
    target_syllables: int = 10             # target per line
    used_end_phonemes: list[str] = field(default_factory=list)  # to avoid repetition
    tension_curve: TensionCurve = field(default_factory=TensionCurve)
    sections_lines: dict = field(default_factory=dict)  # track lines per section

    def add_line(self, line: str, section: str = "verse1"):
        self.accepted_lines.append(line)
        ann = annotate_line(line)
        if ann.end_phoneme:
            self.used_end_phonemes.append(ann.end_phoneme)
        # Track in tension curve
        target_ep = self.get_target_end_phoneme()
        self.tension_curve.update(line, target_ep)
        # Track per-section
        if section not in self.sections_lines:
            self.sections_lines[section] = []
        self.sections_lines[section].append(line)

    def get_target_end_phoneme(self) -> Optional[str]:
        """
        Based on rhyme scheme and lines generated so far,
        return the phoneme the next line should rhyme with (or None if free).
        """
        n = len(self.accepted_lines)
        if self.rhyme_scheme == "free":
            return None

        pairs = {
            "AABB": [(0, 1), (2, 3)],
            "ABAB": [(0, 2), (1, 3)],
            "ABCB": [(1, 3)],
        }
        for a_idx, b_idx in pairs.get(self.rhyme_scheme, []):
            if n % 4 == b_idx and n % 4 > 0:
                # This line should rhyme with line a_idx of this stanza
                stanza_start = (n // 4) * 4
                target_line_idx = stanza_start + a_idx
                if target_line_idx < len(self.accepted_lines):
                    ann = annotate_line(self.accepted_lines[target_line_idx])
                    return ann.end_phoneme
        return None

    def build_prompt(self) -> str:
        """Build the full generation prompt from memory."""
        parts: list[str] = []
        parts.append(f"[GENRE_START] {self.genre} [GENRE_END]")
        if self.sections:
            arc, section = self.sections[-1]
            parts.append(f"[{section}] {arc}")
        for line in self.accepted_lines[-20:]:  # last 20 lines as context
            parts.append(line)
        return "\n".join(parts) + "\n"


# ── Candidate scorer ──────────────────────────────────────────────────────────

@dataclass
class CandidateScore:
    text:               str
    phonetic_score:     float   # 0-1  rhyme match
    syllable_ok:        bool    # within ±3 syllables of target
    novelty_score:      float   # 0-1  not repetitive
    valence_fit:        float   # 0-1  emotional arc fit (simple)
    trajectory_fit:     float   # 0-1  8D emotional geometry fit    ← NEW
    texture_alignment:  float   # 0-1  phonosemantic texture match  ← NEW
    goosebump:          float   # 0-1  predicted dopamine potential  ← NEW
    hook_dna:           float   # 0-1  universal hook pattern score  ← NEW
    total_score:        float


def score_candidate(
    line: str,
    memory: SongMemory,
    target_arc_valence: float = 0.0,
    target_arc_arousal: float = 0.5,
    section: str = "verse1",
    mood: str = "dark",
    tension_state: float = 0.3,
) -> CandidateScore:
    ann = annotate_line(line)

    # 1. Phonetic (rhyme) score
    target_ep = memory.get_target_end_phoneme()
    if target_ep is None:
        phonetic_score = 1.0
    elif ann.end_phoneme and rhymes(ann.end_phoneme, target_ep):
        phonetic_score = 1.0
    else:
        phonetic_score = 0.0

    # 2. Syllable check
    syllable_ok = abs(ann.total_syllables - memory.target_syllables) <= 3

    # 3. Novelty
    def simple_overlap(a: str, b: str) -> float:
        aw = set(a.lower().split())
        bw = set(b.lower().split())
        return len(aw & bw) / len(aw | bw) if (aw and bw) else 0.0

    max_overlap = max(
        (simple_overlap(line, prev) for prev in memory.accepted_lines),
        default=0.0,
    )
    novelty_score = 1.0 - max_overlap

    # 4. Simple valence fit (legacy, low weight)
    emotion = score_line(line)
    val_diff = abs(emotion.valence - target_arc_valence)
    aro_diff = abs(emotion.arousal - target_arc_arousal)
    valence_fit = 1.0 - (val_diff + aro_diff) / 4.0

    # 5. 8D Emotional geometry trajectory fit
    try:
        traj_fit = trajectory_fit_score(line, memory.genre, section)
    except Exception:
        traj_fit = valence_fit  # graceful fallback

    # 6. Phonosemantic texture alignment
    try:
        tex_align = texture_alignment_score(line, mood, section)
    except Exception:
        tex_align = 0.5

    # 7. Dopamine / goosebump potential
    prev_line = memory.accepted_lines[-1] if memory.accepted_lines else None
    try:
        gbump = goosebump_potential(line, tension_state, prev_line, mood)
    except Exception:
        gbump = 0.0

    # 8. Hook DNA
    try:
        hook = hook_dna_score(line, prev_line)
    except Exception:
        hook = 0.0

    # Weighted total — the Cognitive Music score
    total = (
        0.22 * phonetic_score
        + 0.15 * (1.0 if syllable_ok else 0.0)
        + 0.13 * novelty_score
        + 0.18 * traj_fit       # emotional geometry
        + 0.12 * tex_align      # phonosemantic
        + 0.12 * gbump          # goosebump predictor
        + 0.08 * hook           # hook DNA
    )

    return CandidateScore(
        text=line,
        phonetic_score=phonetic_score,
        syllable_ok=syllable_ok,
        novelty_score=novelty_score,
        valence_fit=valence_fit,
        trajectory_fit=traj_fit,
        texture_alignment=tex_align,
        goosebump=gbump,
        hook_dna=hook,
        total_score=total,
    )


# ── Generation engine ─────────────────────────────────────────────────────────

class LyricsEngine:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cpu",
        beam_size: int = 8,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.beam_size = beam_size

    @torch.no_grad()
    def generate_candidates(
        self,
        prompt: str,
        max_new_tokens: int = 60,
        temperature: float = 0.85,
        top_p: float = 0.9,
    ) -> list[str]:
        """Generate beam_size candidate lines from the prompt."""
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=self.beam_size,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.encode("\n")[0] if "\n" in self.tokenizer.get_vocab() else None,
        )

        candidates = []
        prompt_len = input_ids.shape[1]
        for out in outputs:
            new_tokens = out[prompt_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            # Take only the first line
            line = text.strip().split("\n")[0].strip()
            if line:
                candidates.append(line)

        return candidates if candidates else ["[no candidate generated]"]

    def generate_line(
        self,
        memory: SongMemory,
        target_arc_valence: float = 0.0,
        target_arc_arousal: float = 0.5,
        top_n: int = 1,
        section: str = "verse1",
    ) -> list[CandidateScore]:
        """
        Generate and rank candidates using the full Cognitive Music scoring.
        top_n=1 → auto mode, top_n=3 → co-write mode.
        """
        prompt = memory.build_prompt()
        candidates = self.generate_candidates(prompt)

        scored = [
            score_candidate(
                c, memory,
                target_arc_valence, target_arc_arousal,
                section=section,
                mood=memory.mood,
                tension_state=memory.tension_curve.current,
            )
            for c in candidates
        ]

        ok = [s for s in scored if s.syllable_ok]
        ranked = sorted(ok or scored, key=lambda s: s.total_score, reverse=True)
        return ranked[:top_n]

    def generate_verse(
        self,
        memory: SongMemory,
        num_lines: int = 8,
        section: str = "VERSE",
        arc_token: str = "[SETUP]",
        auto_accept: bool = True,
    ) -> list[str]:
        """
        Generate a full verse, accepting top-1 each time and updating memory.
        """
        memory.sections.append((arc_token, section))

        # Estimate arc valence from token
        arc_valence_map = {
            "[SETUP]": (0.0, 0.3),
            "[BUILD]": (0.1, 0.6),
            "[RELEASE]": (0.5, 0.8),
            "[REFRAME]": (-0.1, 0.4),
            "[PEAK]": (0.6, 1.0),
            "[OUTRO]": (-0.2, 0.2),
        }
        target_val, target_aro = arc_valence_map.get(arc_token, (0.0, 0.5))

        memory.tension_curve.reset_for_section(section)
        memory.sections_lines[section] = []

        generated_lines = []
        for _ in range(num_lines):
            top = self.generate_line(memory, target_val, target_aro, top_n=1, section=section)
            if top:
                best = top[0]
                if auto_accept:
                    memory.add_line(best.text, section=section)
                generated_lines.append(best.text)

        return generated_lines

    def analyze_song(self, memory: SongMemory) -> dict:
        """
        Full cognitive analysis of a generated song.
        Shows artists:
          - The 8D emotional arc across all sections
          - The tension-release curve
          - Every line's goosebump potential score
          - Hook DNA patterns detected
          - The single strongest moment in the whole song
          - Phonosemantic texture per section

        This is what no other AI music tool provides.
        """
        sections = {
            sec: lines
            for sec, lines in memory.sections_lines.items()
            if lines
        }
        if not sections:
            # Fallback: build from accepted lines
            sections = {"verse1": memory.accepted_lines}

        emotional_arc  = compute_song_arc(sections, memory.genre)
        dopamine_arc   = analyze_song_dopamine(sections, memory.mood)

        # Texture summary per section
        texture_summary = {}
        for sec, lines in sections.items():
            try:
                from src.model.phonosemantic import analyze_line_texture
                profiles = [analyze_line_texture(l) for l in lines]
                avg = np.mean([p.to_array() for p in profiles], axis=0)
                from src.model.phonosemantic import TextureProfile
                avg_profile = TextureProfile.from_array(avg)
                texture_summary[sec] = avg_profile.describe()
            except Exception:
                texture_summary[sec] = "n/a"

        return {
            "emotional_arc":  emotional_arc,
            "dopamine_arc":   dopamine_arc,
            "texture_summary": texture_summary,
            "peak_moment": dopamine_arc.get("peak_moment", {}),
            "genre": memory.genre,
            "mood":  memory.mood,
        }


# ── Co-write session ──────────────────────────────────────────────────────────

class CoWriteSession:
    """
    Interactive co-write mode.
    User can accept, reject, or edit each suggested line.
    Model updates context and generates next line aware of accepted lines.
    """

    def __init__(self, engine: LyricsEngine, genre: str, rhyme_scheme: str = "AABB"):
        self.engine = engine
        self.memory = SongMemory(genre=genre, rhyme_scheme=rhyme_scheme)
        self.history: list[dict] = []  # audit trail

    def start_section(self, section: str = "VERSE", arc_token: str = "[SETUP]"):
        self.memory.sections.append((arc_token, section))
        print(f"\n── {section} {arc_token} ──")

    def suggest(self, n: int = 3) -> list[CandidateScore]:
        """Suggest n candidates for the next line."""
        return self.engine.generate_line(self.memory, top_n=n)

    def accept(self, line: str):
        """Accept a line (user-written or from suggestions)."""
        self.memory.add_line(line)
        self.history.append({"action": "accept", "line": line})
        print(f"  ✓ {line}")

    def reject(self):
        """Skip — regenerate next suggestion."""
        self.history.append({"action": "reject"})

    def get_song(self) -> str:
        return "\n".join(self.memory.accepted_lines)


if __name__ == "__main__":
    # Local smoke test with GPT-2 (no GPU needed)
    from transformers import AutoModelForCausalLM, AutoTokenizer as HFTokenizer

    print("Loading GPT-2 for local smoke test...")
    tok = HFTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained("gpt2")

    engine = LyricsEngine(mdl, tok, device="cpu", beam_size=3)
    memory = SongMemory(genre="hip_hop", rhyme_scheme="AABB", target_syllables=10)
    memory.sections.append(("[SETUP]", "VERSE"))

    print("\nGenerating candidates for first line...")
    candidates = engine.generate_line(memory, top_n=3)
    for i, c in enumerate(candidates):
        print(f"  [{i+1}] (score={c.total_score:.2f}) {c.text}")

"""
Cortical Creative Loop (CCL) Dataset

Brain-inspired training format based on predictive coding and creative cognition research.
The CCL format teaches the model to:
  1. PERCEIVE: Encode context (section, emotion, rhythm)
  2. INTENT: Form generation goals
  3. PREDICT: Generate draft output
  4. ERROR: Detect prediction errors (optional - for DPO)
  5. SELECT: Choose final output

CCL Training format:
  [INST] Write {genre} lyrics for [{section}] ({arc}): [/INST]
  [PERCEIVE] [CONTEXT] section={section} position={pos} [EMO_STATE] valence={v} arousal={a} [RHYTHM_STATE] flow={flow}
  [INTENT] {arc} [TARGET_EMO] {mood} [TARGET_RHYTHM] {flow} [TARGET_NOVELTY] varied
  [PREDICT]
  {lyrics}
  [ERROR] [NO_ERROR]
  [SELECT]
  {lyrics}
  [MEMORY] genre={genre} sections={n} emotion={mood}
"""

import json
import random
from pathlib import Path
from typing import Optional
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

from configs.genres import SECTION_TOKENS, GENRE_DESCRIPTIONS
from src.data.valence_scorer import score_lyrics, compute_song_arc, LineEmotion
from src.data.rhyme_labeler import detect_scheme
from src.model.dual_tokenizer import word_to_phoneme_ids, PHONEME_TO_ID


SECTION_HEADER_RE = __import__("re").compile(
    r"^\[(verse|chorus|prechorus|bridge|hook|outro|intro)\]",
    __import__("re").IGNORECASE
)


def split_into_sections(lyrics: str) -> list[tuple[str, list[str]]]:
    """Split raw lyrics into (section_name, lines) tuples."""
    sections: list[tuple[str, list[str]]] = []
    current_name = "VERSE"
    current_lines: list[str] = []

    for raw_line in lyrics.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = SECTION_HEADER_RE.match(line)
        if m:
            if current_lines:
                sections.append((current_name, current_lines))
            current_name = m.group(1).upper()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_name, current_lines))

    return sections if sections else [("VERSE", [l.strip() for l in lyrics.splitlines() if l.strip()])]


def _valence_label(val: float) -> str:
    """Convert numeric valence to categorical label."""
    if val < -0.3:
        return "dark"
    elif val < 0.0:
        return "melancholic"
    elif val < 0.3:
        return "neutral"
    elif val < 0.6:
        return "uplifting"
    else:
        return "euphoric"


def _arousal_label(aro: float) -> str:
    """Convert numeric arousal to categorical label."""
    if aro < 0.3:
        return "calm"
    elif aro < 0.5:
        return "moderate"
    elif aro < 0.7:
        return "energetic"
    else:
        return "intense"


def _flow_label(lines: list[str]) -> str:
    """Estimate flow density from syllable count."""
    if not lines:
        return "standard"
    try:
        from src.data.phoneme_annotator import annotate_line
        total_syls = sum(annotate_line(l).total_syllables for l in lines)
        avg_syls = total_syls / len(lines)
        if avg_syls >= 12:
            return "dense"
        elif avg_syls >= 8:
            return "standard"
        else:
            return "sparse"
    except Exception:
        return "standard"


def format_ccl_example(
    record: dict,
    include_memory: bool = True,
    include_error: bool = True,
) -> tuple[str, int]:
    """
    Format a song record into CCL (Cortical Creative Loop) training format.

    Returns
    -------
    (full_text, instruction_char_len)
    """
    genre = record.get("genre", "hip_hop")
    lyrics = record.get("lyrics", "")
    sections = split_into_sections(lyrics)

    # Compute arc tokens
    try:
        arc_tokens = compute_song_arc([lines for _, lines in sections])
    except Exception:
        arc_tokens = ["[SETUP]"] * len(sections)

    total_sections = len(sections)
    all_parts = []
    instruction_char_len = 0

    for idx, ((section_name, lines), arc) in enumerate(zip(sections, arc_tokens)):
        if not lines:
            continue

        # Score emotions
        section_text = "\n".join(lines)
        try:
            emotions = score_lyrics(section_text)
            avg_valence = sum(e.valence for e in emotions) / max(len(emotions), 1)
            avg_arousal = sum(e.arousal for e in emotions) / max(len(emotions), 1)
        except Exception:
            avg_valence = 0.0
            avg_arousal = 0.5

        valence_lbl = _valence_label(avg_valence)
        arousal_lbl = _arousal_label(avg_arousal)
        flow_lbl = _flow_label(lines)

        # Build section-level training example
        # INSTRUCTION
        section_tok = f"[{section_name}]" if f"[{section_name}]" in SECTION_TOKENS else "[VERSE]"
        instruction = f"[INST] Write {genre} lyrics for {section_tok} ({arc}): [/INST]"

        if idx == 0:
            instruction_char_len = len(instruction)

        # PERCEIVE phase
        perceive = (
            f"[PERCEIVE] [CONTEXT] section={section_name} position={idx + 1}/{total_sections} "
            f"[EMO_STATE] valence={valence_lbl} arousal={arousal_lbl} "
            f"[RHYTHM_STATE] flow={flow_lbl}"
        )

        # INTENT phase
        intent = (
            f"[INTENT] {arc} [TARGET_EMO] {valence_lbl} "
            f"[TARGET_RHYTHM] {flow_lbl} [TARGET_NOVELTY] varied"
        )

        # PREDICT phase (the actual lyrics)
        predict_block = "[PREDICT]\n" + "\n".join(lines)

        # ERROR phase (for training, we mark as NO_ERROR since these are ground truth)
        error_block = ""
        if include_error:
            error_block = "\n[ERROR] [NO_ERROR]"

        # SELECT phase (repeat lyrics - teaches model to commit)
        select_block = "\n[SELECT]\n" + "\n".join(lines)

        # Combine section
        section_parts = [instruction, perceive, intent, predict_block]
        if include_error:
            section_parts.append(error_block)
        section_parts.append(select_block)

        all_parts.append("\n".join(section_parts))

    # MEMORY consolidation (end of song)
    if include_memory and sections:
        memory = f"[MEMORY] genre={genre} sections={total_sections} emotion={valence_lbl}"
        all_parts.append(memory)

    full_text = "\n\n".join(all_parts)
    return full_text, instruction_char_len


def compute_example_complexity(record: dict) -> tuple[int, int, int]:
    """Compute complexity metrics for curriculum ordering."""
    lyrics = record.get("lyrics", "")
    sections = split_into_sections(lyrics)

    total_lines = sum(len(lines) for _, lines in sections)
    num_sections = len(sections)

    all_lines = [line for _, lines in sections for line in lines]
    avg_len = sum(len(line) for line in all_lines) // max(len(all_lines), 1)

    return (total_lines, num_sections, avg_len)


def curriculum_sort_records(
    records: list[dict],
    max_per_artist: int = 3,
    seed: int = 42,
) -> list[dict]:
    """
    Sort records for curriculum learning:
    1. Cap songs per artist to reduce overconcentration
    2. Sort by complexity (short/simple first)
    3. Shuffle within complexity bands for variety
    """
    rng = random.Random(seed)

    # Cap per artist
    artist_counts: Counter = Counter()
    filtered = []

    shuffled = list(records)
    rng.shuffle(shuffled)

    for record in shuffled:
        artist = record.get("artist", "unknown")
        if artist_counts[artist] < max_per_artist:
            filtered.append(record)
            artist_counts[artist] += 1

    # Compute complexity
    with_complexity = []
    for record in filtered:
        lines, sections, avg_len = compute_example_complexity(record)
        complexity = lines * 10 + sections * 50 + abs(avg_len - 40)
        with_complexity.append((complexity, record))

    # Sort by complexity
    with_complexity.sort(key=lambda x: x[0])

    # Shuffle within bands
    band_size = max(len(with_complexity) // 5, 1)
    result = []
    for i in range(0, len(with_complexity), band_size):
        band = [r for _, r in with_complexity[i:i + band_size]]
        rng.shuffle(band)
        result.extend(band)

    return result


class CorticalDataset(Dataset):
    """
    CCL (Cortical Creative Loop) training dataset.

    Formats songs using brain-inspired PERCEIVE -> INTENT -> PREDICT -> ERROR -> SELECT format.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_samples: Optional[int] = None,
        curriculum_order: bool = True,
        max_per_artist: int = 3,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load records
        records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                if max_samples and len(records) >= max_samples * 2:
                    break

        # Apply curriculum ordering
        if curriculum_order:
            records = curriculum_sort_records(records, max_per_artist=max_per_artist, seed=seed)

        if max_samples and len(records) > max_samples:
            records = records[:max_samples]

        self.records = records
        print(f"[ccl_dataset] Loaded {len(self.records)} songs from {jsonl_path}")
        print(f"[ccl_dataset] Format: CCL (Cortical Creative Loop), max_length={max_length}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        text, instruction_char_len = format_ccl_example(record)

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Mask padding
        labels[attention_mask == 0] = -100

        # Mask instruction prefix
        instruction_prefix = text[:instruction_char_len]
        prefix_ids = self.tokenizer(
            instruction_prefix,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        n_prefix = len(prefix_ids)
        if n_prefix < self.max_length:
            labels[:n_prefix] = -100

        # Build phoneme IDs
        words = text.split()
        ph_ids: list[int] = []
        for w in words:
            ph_ids.extend(word_to_phoneme_ids(w))
        if len(ph_ids) >= self.max_length:
            phoneme_ids = torch.tensor(ph_ids[:self.max_length], dtype=torch.long)
        else:
            pad = [PHONEME_TO_ID["<PAD_PH>"]] * (self.max_length - len(ph_ids))
            phoneme_ids = torch.tensor(ph_ids + pad, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "phoneme_ids": phoneme_ids,
        }


def create_ccl_dataloaders(
    train_jsonl: str,
    val_jsonl: Optional[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    style_vec_dir: Optional[str] = None,
    num_workers: int = 0,
    mode: str = "ccl",
    max_per_artist: int = 3,
):
    """
    Create dataloaders for CCL training.

    Parameters
    ----------
    mode : str
        'ccl' for Cortical Creative Loop format (default)
        'simple' falls back to regular dataset
    max_per_artist : int
        Cap songs per artist to reduce overconcentration
    """
    train_ds = CorticalDataset(
        train_jsonl,
        tokenizer,
        max_length=max_length,
        curriculum_order=True,
        max_per_artist=max_per_artist,
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    val_dl = None
    if val_jsonl and Path(val_jsonl).exists():
        val_ds = CorticalDataset(
            val_jsonl,
            tokenizer,
            max_length=max_length,
            curriculum_order=False,
            max_per_artist=999,
        )
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl

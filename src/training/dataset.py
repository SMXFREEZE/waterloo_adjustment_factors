"""
Training dataset: reads annotated JSONL, formats examples for the LLM.

Each training example format:
  [GENRE_START] trap [GENRE_END] [STYLE_START] <style_prefix> [STYLE_END]
  [VERSE] [SETUP]
  <lyrics line 1>
  <lyrics line 2>
  ...
  [CHORUS] [RELEASE]
  <chorus lines>
  ...

Phoneme IDs are stored as a parallel sequence for the phonetic head.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from configs.genres import SECTION_TOKENS, ARC_TOKENS, SPECIAL_TOKENS
from src.data.phoneme_annotator import annotate_lyrics, annotation_to_dict
from src.data.rhyme_labeler import detect_scheme
from src.data.valence_scorer import score_lyrics, compute_song_arc
from src.model.dual_tokenizer import word_to_phoneme_ids, PHONEME_TO_ID


SECTION_HEADER_RE = __import__("re").compile(r"^\[(verse|chorus|prechorus|bridge|hook|outro)\]", __import__("re").IGNORECASE)


def split_into_sections(lyrics: str) -> list[tuple[str, list[str]]]:
    """
    Split raw lyrics into (section_name, lines) tuples.
    If no section markers, treat as single [VERSE].
    """
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


def format_training_example(
    record: dict,
    style_vec: Optional[np.ndarray] = None,
) -> str:
    """
    Format a song record into the training text format.
    """
    genre = record.get("genre", "hip_hop")
    lyrics = record.get("lyrics", "")
    sections = split_into_sections(lyrics)

    # Compute arc tokens per section
    arc_tokens = compute_song_arc([lines for _, lines in sections])

    parts = [f"[GENRE_START] {genre} [GENRE_END]"]
    if style_vec is not None:
        # We represent style vector as a compact token — actual injection
        # happens at embedding level, not text level
        parts.append("[STYLE_START] [STYLE_END]")

    for (section_name, lines), arc in zip(sections, arc_tokens):
        section_tok = f"[{section_name}]" if f"[{section_name}]" in SECTION_TOKENS else "[VERSE]"
        parts.append(f"{section_tok} {arc}")
        parts.extend(lines)

    return "\n".join(parts)


class LyricsDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        style_vec_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.style_vec_dir = Path(style_vec_dir) if style_vec_dir else None

        self.records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))
                if max_samples and len(self.records) >= max_samples:
                    break

        print(f"[dataset] Loaded {len(self.records)} songs from {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def _load_style_vec(self, artist: str) -> Optional[np.ndarray]:
        if self.style_vec_dir is None:
            return None
        path = self.style_vec_dir / f"{artist.replace(' ', '_')}.npy"
        if path.exists():
            return np.load(str(path)).astype(np.float32)
        return None

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        style_vec = self._load_style_vec(record.get("artist", ""))

        text = format_training_example(record, style_vec)
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
        labels[attention_mask == 0] = -100

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

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "phoneme_ids": phoneme_ids,
        }

        if style_vec is not None:
            item["style_vec"] = torch.tensor(style_vec, dtype=torch.float32)

        return item


def create_dataloaders(
    train_jsonl: str,
    val_jsonl: Optional[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 1024,
    style_vec_dir: Optional[str] = None,
    num_workers: int = 0,
):
    from torch.utils.data import DataLoader

    train_ds = LyricsDataset(train_jsonl, tokenizer, max_length, style_vec_dir)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dl = None
    if val_jsonl and Path(val_jsonl).exists():
        val_ds = LyricsDataset(val_jsonl, tokenizer, max_length, style_vec_dir)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl

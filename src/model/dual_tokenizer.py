"""
Dual tokenizer: standard BPE (from HuggingFace) + phoneme tokens.

The dual tokenizer returns two parallel token sequences:
  - semantic_ids: standard subword token IDs for the LLM
  - phoneme_ids:  integer IDs representing phoneme sequences, aligned per word

These are merged as concatenated embeddings at the model input layer.
"""

import re
from dataclasses import dataclass
from typing import Optional

import pronouncing
from transformers import AutoTokenizer, PreTrainedTokenizer

from configs.genres import SPECIAL_TOKENS


# ── Phoneme vocabulary ────────────────────────────────────────────────────────

# ARPAbet phoneme set (CMU dict uses this)
ARPABET = [
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY",
    "JH", "K", "L", "M", "N", "NG",
    "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW",
    "V", "W", "Y", "Z", "ZH",
]

# Stress suffixes 0/1/2 → separate tokens
PHONEME_VOCAB = ["<PAD_PH>", "<UNK_PH>"] + [
    f"{p}{s}" for p in ARPABET for s in ["", "0", "1", "2"]
]
PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEME_VOCAB)}
ID_TO_PHONEME = {i: p for p, i in PHONEME_TO_ID.items()}

PHONEME_VOCAB_SIZE = len(PHONEME_VOCAB)  # ~162


def word_to_phoneme_ids(word: str) -> list[int]:
    """Convert a word to a list of phoneme IDs via CMU dict."""
    clean = re.sub(r"[^a-zA-Z'']", "", word).lower()
    phones_list = pronouncing.phones_for_word(clean)
    if not phones_list:
        return [PHONEME_TO_ID["<UNK_PH>"]]
    phones = phones_list[0].split()
    return [PHONEME_TO_ID.get(p, PHONEME_TO_ID["<UNK_PH>"]) for p in phones]


# ── Dual tokenizer ────────────────────────────────────────────────────────────

@dataclass
class DualEncoding:
    semantic_ids: list[int]         # BPE token IDs
    phoneme_ids: list[int]          # phoneme IDs, padded to same length as semantic_ids
    attention_mask: list[int]
    text: str


class DualTokenizer:
    """
    Wraps a HuggingFace tokenizer and adds a parallel phoneme token stream.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B"):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        self._add_special_tokens()

    def _add_special_tokens(self):
        new_tokens = [t for t in SPECIAL_TOKENS if t not in self.tokenizer.get_vocab()]
        if new_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode(self, text: str, max_length: int = 512) -> DualEncoding:
        enc = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        semantic_ids: list[int] = enc["input_ids"]
        attention_mask: list[int] = enc["attention_mask"]

        # Build phoneme stream aligned to semantic tokens
        phoneme_ids = self._build_phoneme_stream(text, len(semantic_ids))

        return DualEncoding(
            semantic_ids=semantic_ids,
            phoneme_ids=phoneme_ids,
            attention_mask=attention_mask,
            text=text,
        )

    def _build_phoneme_stream(self, text: str, target_len: int) -> list[int]:
        """
        Produce a phoneme ID sequence of length target_len.
        Words are expanded to their phoneme sequences, then resampled
        to match target_len via repeat/truncate.
        """
        words = text.split()
        all_phoneme_ids: list[int] = []
        for word in words:
            all_phoneme_ids.extend(word_to_phoneme_ids(word))

        if not all_phoneme_ids:
            return [PHONEME_TO_ID["<PAD_PH>"]] * target_len

        # Resample to target_len
        if len(all_phoneme_ids) >= target_len:
            return all_phoneme_ids[:target_len]
        else:
            # Tile and trim
            import math
            reps = math.ceil(target_len / len(all_phoneme_ids))
            tiled = (all_phoneme_ids * reps)[:target_len]
            return tiled

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def save(self, path: str):
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str) -> "DualTokenizer":
        obj = cls.__new__(cls)
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        return obj


# ── Offline fallback tokenizer (no model download needed) ────────────────────

class OfflineDualTokenizer:
    """
    Lightweight tokenizer for local dev/testing without downloading Llama.
    Uses GPT-2 tokenizer as a proxy.
    """

    def __init__(self):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        new_tokens = [t for t in SPECIAL_TOKENS if t not in self.tokenizer.get_vocab()]
        if new_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode(self, text: str, max_length: int = 512) -> DualEncoding:
        enc = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        semantic_ids: list[int] = enc["input_ids"]
        attention_mask: list[int] = enc["attention_mask"]
        words = text.split()
        phoneme_ids: list[int] = []
        for w in words:
            phoneme_ids.extend(word_to_phoneme_ids(w))
        if len(phoneme_ids) >= len(semantic_ids):
            phoneme_ids = phoneme_ids[:len(semantic_ids)]
        else:
            phoneme_ids += [PHONEME_TO_ID["<PAD_PH>"]] * (len(semantic_ids) - len(phoneme_ids))
        return DualEncoding(
            semantic_ids=semantic_ids,
            phoneme_ids=phoneme_ids,
            attention_mask=attention_mask,
            text=text,
        )

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


if __name__ == "__main__":
    tok = OfflineDualTokenizer()
    enc = tok.encode("I been movin' in silence, they can't feel my weight")
    print(f"Semantic IDs  (first 10): {enc.semantic_ids[:10]}")
    print(f"Phoneme IDs   (first 10): {enc.phoneme_ids[:10]}")
    print(f"Attention mask(first 10): {enc.attention_mask[:10]}")
    print(f"Vocab size: {tok.vocab_size}")

"""
Vocal / topline generator using Bark (open source by Suno).

Bark can generate:
  - Singing with melody (topline)
  - Spoken word / rap delivery
  - Humming / vocal chops
  - Multi-language vocals

Voice presets (built-in to Bark):
  v2/en_speaker_0 ... v2/en_speaker_9   — English voices
  v2/fr_speaker_0 ... v2/fr_speaker_9   — French
  v2/es_speaker_0 ... v2/es_speaker_9   — Spanish
  v2/de_speaker_0                        — German
  v2/ar_speaker_0                        — Arabic
  v2/zh_speaker_0 ... v2/zh_speaker_8   — Chinese
  v2/ja_speaker_0 ... v2/ja_speaker_7   — Japanese
  v2/ko_speaker_0 ... v2/ko_speaker_8   — Korean
  v2/pt_speaker_0 ... v2/pt_speaker_8   — Portuguese
  v2/hi_speaker_0 ... v2/hi_speaker_6   — Hindi
  + many more

Singing notation in Bark:
  ♪ lyrics here ♪   — triggers melodic/singing output
  [laughs]           — laughter
  [clears throat]    — vocal effects
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Optional

import torch

# Language → recommended voice preset
LANG_VOICE_MAP: dict[str, list[str]] = {
    "en":  [f"v2/en_speaker_{i}" for i in range(10)],
    "fr":  [f"v2/fr_speaker_{i}" for i in range(10)],
    "es":  [f"v2/es_speaker_{i}" for i in range(10)],
    "de":  [f"v2/de_speaker_{i}" for i in range(1)],
    "ar":  [f"v2/ar_speaker_{i}" for i in range(1)],
    "zh":  [f"v2/zh_speaker_{i}" for i in range(9)],
    "ja":  [f"v2/ja_speaker_{i}" for i in range(8)],
    "ko":  [f"v2/ko_speaker_{i}" for i in range(9)],
    "pt":  [f"v2/pt_speaker_{i}" for i in range(9)],
    "hi":  [f"v2/hi_speaker_{i}" for i in range(7)],
    "tr":  [f"v2/tr_speaker_{i}" for i in range(4)],
    "pl":  [f"v2/pl_speaker_{i}" for i in range(5)],
    "it":  [f"v2/it_speaker_{i}" for i in range(6)],
    "ru":  [f"v2/ru_speaker_{i}" for i in range(6)],
}

BARK_SAMPLE_RATE = 24000  # Bark outputs at 24kHz


def wrap_singing(text: str) -> str:
    """Wrap lyrics with music notes to trigger Bark's singing mode."""
    # Clean text
    text = text.strip()
    # Add ♪ markers around lines to signal singing
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return " ♪ " + " ♪ ".join(lines) + " ♪"


def wrap_rap(text: str) -> str:
    """Format for rap delivery — spoken rhythmically."""
    text = text.strip()
    return text  # Bark handles rap naturally without special tokens


class VocalGenerator:
    def __init__(
        self,
        device: str = "auto",
        use_small_model: bool = True,   # True = faster, less VRAM
    ):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_small = use_small_model
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            import bark
            if self.use_small:
                os.environ["SUNO_USE_SMALL_MODELS"] = "True"
            if self.device == "cuda":
                os.environ["SUNO_OFFLOAD_CPU"] = "False"
            else:
                os.environ["SUNO_OFFLOAD_CPU"] = "True"
            from bark import preload_models
            print(f"Loading Bark models (use_small={self.use_small})...")
            preload_models()
            self._loaded = True
            print("Bark loaded.")
        except ImportError:
            raise ImportError("bark not installed. Run: pip install suno-bark")

    def generate_singing(
        self,
        lyrics: str,
        language: str = "en",
        voice_idx: int = 0,
        singing: bool = True,
    ) -> np.ndarray:
        """
        Generate singing/vocal audio from lyrics text.
        Returns numpy array at 24kHz.
        """
        self._load()
        from bark import generate_audio, SAMPLE_RATE

        # Pick voice preset
        voices = LANG_VOICE_MAP.get(language, LANG_VOICE_MAP["en"])
        voice = voices[voice_idx % len(voices)]

        # Wrap text
        text = wrap_singing(lyrics) if singing else lyrics

        print(f"Generating vocals [{language}, {voice}]: {text[:60]}...")
        audio = generate_audio(text, history_prompt=voice)
        return audio

    def generate_verse(
        self,
        lines: list[str],
        language: str = "en",
        voice_idx: int = 0,
        singing: bool = True,
        max_lines_per_chunk: int = 4,
    ) -> np.ndarray:
        """
        Generate vocals for a full verse, chunked by lines.
        Bark works best on short chunks — we generate per 4 lines then concatenate.
        """
        self._load()
        chunks = []
        for i in range(0, len(lines), max_lines_per_chunk):
            chunk_lines = lines[i:i + max_lines_per_chunk]
            chunk_text = "\n".join(chunk_lines)
            audio_chunk = self.generate_singing(
                chunk_text,
                language=language,
                voice_idx=voice_idx,
                singing=singing,
            )
            chunks.append(audio_chunk)
            print(f"  Chunk {i//max_lines_per_chunk + 1}/{math.ceil(len(lines)/max_lines_per_chunk)} done")

        if not chunks:
            return np.array([])
        return np.concatenate(chunks)

    def generate_full_song_vocals(
        self,
        song_sections: dict[str, list[str]],  # {"verse1": [...], "chorus": [...], ...}
        language: str = "en",
        voice_idx: int = 0,
        singing: bool = True,
    ) -> dict[str, np.ndarray]:
        """Generate vocals for each section of a song separately."""
        result = {}
        for section_name, lines in song_sections.items():
            print(f"\nGenerating vocals for {section_name}...")
            audio = self.generate_verse(lines, language, voice_idx, singing)
            result[section_name] = audio
        return result

    @staticmethod
    def save(audio: np.ndarray, path: str, sample_rate: int = BARK_SAMPLE_RATE):
        import soundfile as sf
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio, sample_rate)
        print(f"Vocals saved: {path}")


# Patch missing math import
import math


if __name__ == "__main__":
    gen = VocalGenerator(use_small_model=True)
    lines = [
        "I been moving in silence, they can't feel my weight",
        "Every step I take yeah I'm moving with fate",
        "They say the game is cold but I turn up the heat",
        "Diamonds on my wrist while I dance to the beat",
    ]
    audio = gen.generate_verse(lines, language="en", voice_idx=0, singing=True)
    gen.save(audio, "output/test_vocals.wav")
    print(f"Vocal audio: {len(audio)/BARK_SAMPLE_RATE:.1f}s")

"""
Instrumental generator using Meta's MusicGen.

Generates beats/instrumentals conditioned on:
  - Genre / style (trap, afrobeats, kpop, etc.)
  - BPM
  - Mood/energy/valence
  - Duration

Models (auto-downloaded from HuggingFace):
  - facebook/musicgen-small   (300M — fast, ~1.5GB, good for dev)
  - facebook/musicgen-medium  (1.5B — balanced, ~3GB)
  - facebook/musicgen-large   (3.3B — best quality, ~6GB, needs A10G)
  - facebook/musicgen-melody  (1.5B — can condition on a reference melody)

Usage:
  gen = InstrumentalGenerator(model_size="medium")
  audio = gen.generate(genre="trap", bpm=140, duration=30)
  gen.save(audio, "output/beat.wav")
"""

import os
import math
import numpy as np
from pathlib import Path
from typing import Optional

import torch

# MusicGen prompt templates per style
STYLE_PROMPTS: dict[str, str] = {
    "trap":       "dark trap beat, heavy 808 bass, hi-hats, snare rolls, {bpm} BPM, atmospheric, professional",
    "drill":      "UK drill beat, sliding 808, dark minor key, percussion, {bpm} BPM, cold atmosphere",
    "hip_hop":    "boom bap hip hop instrumental, punchy drums, jazz samples, vinyl crackle, {bpm} BPM",
    "rnb":        "smooth R&B instrumental, warm chords, soft drums, atmospheric pads, {bpm} BPM, soulful",
    "pop":        "catchy pop instrumental, bright synths, four-on-the-floor drums, {bpm} BPM, radio-ready",
    "afrobeats":  "afrobeats instrumental, talking drum, guitar, percussion, {bpm} BPM, joyful, danceable",
    "amapiano":   "amapiano log drum, piano riffs, soft percussion, {bpm} BPM, South African, groovy",
    "reggaeton":  "reggaeton dembow beat, bass, percussion, {bpm} BPM, Latin, energetic",
    "kpop":       "K-pop instrumental, bright synths, tight drums, melodic hook, {bpm} BPM, polished",
    "latin_trap": "Latin trap beat, 808 bass, reggaeton elements, {bpm} BPM, Spanish vibes",
    "corrido":    "corrido tumbado beat, bass guitar, norteño elements, {bpm} BPM, Mexican",
    "dancehall":  "dancehall riddim, digital drums, bass, {bpm} BPM, Jamaican, energetic",
    "edm":        "EDM drop, synth leads, punchy kick, {bpm} BPM, euphoric, festival-ready",
    "indie":      "indie rock instrumental, guitar, drums, bass, {bpm} BPM, lo-fi, emotional",
    "alt_emo":    "alternative emo instrumental, distorted guitar, emotional drums, {bpm} BPM, intense",
    "french_rap": "French rap instrumental, dark trap, sub bass, {bpm} BPM, Parisian, moody",
    "arabic_pop": "Arabic pop instrumental, oud, qanun, modern beats, {bpm} BPM, melodic",
    "chaabi":     "Moroccan chaabi instrumental, guembri, percussion, {bpm} BPM, North African",
    "afroswing":  "Afroswing beat, guitar, afrobeats drums, {bpm} BPM, London, smooth",
    "country":    "country instrumental, acoustic guitar, fiddle, drums, {bpm} BPM, Nashville",
    "j_pop":      "J-pop instrumental, bright synths, melodic, tight drums, {bpm} BPM, Japanese",
    "mahraganat": "Mahraganat Egyptian street music, electronic, distorted, {bpm} BPM, Cairo",
    "neo_soul":   "neo soul instrumental, keys, warm bass, live drums, {bpm} BPM, soulful, organic",
    "boom_bap":   "boom bap instrumental, sampled breaks, punchy snare, {bpm} BPM, classic hip hop",
}

MOOD_MODIFIERS = {
    "dark":    "dark, minor key, aggressive, moody",
    "happy":   "uplifting, major key, bright, joyful",
    "sad":     "melancholic, minor key, slow, emotional",
    "hype":    "energetic, loud, distorted, aggressive, hyped",
    "chill":   "relaxed, mellow, smooth, laid back",
    "romantic":"romantic, warm, soft, intimate",
    "epic":    "cinematic, epic, powerful, orchestral elements",
}


class InstrumentalGenerator:
    def __init__(
        self,
        model_size: str = "small",   # small / medium / large / melody
        device: str = "auto",
    ):
        self.model_size = model_size
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            from audiocraft.models import MusicGen
            model_id = f"facebook/musicgen-{self.model_size}"
            print(f"Loading MusicGen {self.model_size} ({model_id})...")
            self.model = MusicGen.get_pretrained(model_id)
            self.model.set_generation_params(use_sampling=True, top_k=250)
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to(self.device)
            self._loaded = True
            print(f"MusicGen loaded on {self.device}")
        except ImportError:
            raise ImportError("audiocraft not installed. Run: pip install audiocraft")

    def build_prompt(
        self,
        genre: str,
        bpm: int = 120,
        mood: str = "hype",
        extra: str = "",
    ) -> str:
        template = STYLE_PROMPTS.get(genre, STYLE_PROMPTS["hip_hop"])
        prompt = template.format(bpm=bpm)
        mood_str = MOOD_MODIFIERS.get(mood, "")
        if mood_str:
            prompt += f", {mood_str}"
        if extra:
            prompt += f", {extra}"
        return prompt

    def generate(
        self,
        genre: str,
        bpm: int = 120,
        mood: str = "hype",
        duration: int = 30,        # seconds
        extra_prompt: str = "",
        reference_audio: Optional[np.ndarray] = None,  # for musicgen-melody
        reference_sr: int = 32000,
    ) -> np.ndarray:
        """
        Generate instrumental audio.
        Returns numpy array (float32, shape: [samples]) at 32kHz.
        """
        self._load()
        prompt = self.build_prompt(genre, bpm, mood, extra_prompt)
        print(f"Generating {duration}s instrumental: {prompt[:80]}...")

        self.model.set_generation_params(duration=duration)

        if self.model_size == "melody" and reference_audio is not None:
            import torchaudio
            ref_tensor = torch.from_numpy(reference_audio).unsqueeze(0).unsqueeze(0)
            wav = self.model.generate_with_chroma(
                descriptions=[prompt],
                melody_wavs=ref_tensor,
                melody_sample_rate=reference_sr,
                progress=True,
            )
        else:
            wav = self.model.generate(descriptions=[prompt], progress=True)

        # wav shape: (batch, channels, samples) → return mono numpy
        audio = wav[0, 0].cpu().numpy()
        return audio

    def generate_sections(
        self,
        genre: str,
        bpm: int,
        sections: list[tuple[str, int]],  # [(section_name, duration_sec), ...]
        mood: str = "hype",
    ) -> dict[str, np.ndarray]:
        """
        Generate separate audio for each section (verse, chorus, bridge).
        e.g. sections = [("verse", 20), ("chorus", 15), ("bridge", 10)]
        """
        results = {}
        for section_name, duration in sections:
            # Add section-specific modifier to prompt
            section_modifier = {
                "verse":   "verse section, less intense",
                "chorus":  "chorus drop, most energetic part, full arrangement",
                "bridge":  "bridge section, breakdown, stripped back",
                "hook":    "hook, catchy, memorable",
                "intro":   "intro, building up",
                "outro":   "outro, fading out",
            }.get(section_name.lower(), "")
            audio = self.generate(
                genre=genre,
                bpm=bpm,
                mood=mood,
                duration=duration,
                extra_prompt=section_modifier,
            )
            results[section_name] = audio
            print(f"  Generated {section_name}: {duration}s")
        return results

    @staticmethod
    def save(audio: np.ndarray, path: str, sample_rate: int = 32000):
        """Save audio array to WAV file."""
        import soundfile as sf
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio, sample_rate)
        size_mb = Path(path).stat().st_size / 1e6
        print(f"Saved: {path} ({size_mb:.1f} MB)")

    @staticmethod
    def to_mp3(wav_path: str, mp3_path: str, bitrate: str = "192k"):
        """Convert WAV to MP3 using pydub."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(wav_path)
            audio.export(mp3_path, format="mp3", bitrate=bitrate)
            print(f"MP3 saved: {mp3_path}")
        except Exception as e:
            print(f"MP3 conversion failed: {e} — keeping WAV")


if __name__ == "__main__":
    gen = InstrumentalGenerator(model_size="small")
    audio = gen.generate(genre="trap", bpm=140, mood="dark", duration=15)
    gen.save(audio, "output/test_trap.wav")
    print(f"Audio shape: {audio.shape}, duration: {len(audio)/32000:.1f}s")

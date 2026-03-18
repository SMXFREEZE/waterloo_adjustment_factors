"""
Song Assembler — combines instrumental + vocals into a full song.

Pipeline:
  1. Generate lyrics (LyricsEngine)
  2. Generate instrumental per section (InstrumentalGenerator)
  3. Generate vocals per section (VocalGenerator)
  4. Mix: instrumental (full volume) + vocals (on top)
  5. Arrange sections in order: intro → verse → chorus → verse → chorus → bridge → outro
  6. Export as MP3/WAV

Also handles:
  - BPM alignment (stretch/compress to match target BPM)
  - Volume normalization
  - Fade in/out between sections
  - Vocal + instrumental level balancing
"""

import math
import os
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import soundfile as sf


INSTRUMENTAL_SR = 32000   # MusicGen sample rate
VOCAL_SR        = 24000   # Bark sample rate
OUTPUT_SR       = 44100   # Final output sample rate


@dataclass
class SongSpec:
    """Full specification for a song to generate."""
    title:        str
    genre:        str
    bpm:          int
    mood:         str                    # dark / happy / hype / chill / romantic
    language:     str                    # en / fr / es / ar / etc.
    voice_idx:    int = 0
    duration_sec: int = 180              # target total duration (~3 min)
    sections: list[tuple[str, int]] = field(default_factory=lambda: [
        ("intro",   10),
        ("verse1",  20),
        ("chorus",  15),
        ("verse2",  20),
        ("chorus2", 15),
        ("bridge",  10),
        ("outro",   10),
    ])
    lyrics: dict[str, list[str]] = field(default_factory=dict)   # section → lines
    output_dir: str = "output/songs"


@dataclass
class GeneratedSong:
    spec:         SongSpec
    instrumental: dict[str, np.ndarray]   # section → audio at 32kHz
    vocals:       dict[str, np.ndarray]   # section → audio at 24kHz
    mixed:        Optional[np.ndarray] = None  # final mix at OUTPUT_SR
    output_path:  Optional[str] = None


# ── Audio utilities ────────────────────────────────────────────────────────────

def resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Simple linear resampling."""
    if from_sr == to_sr:
        return audio
    try:
        import scipy.signal
        n_samples = int(len(audio) * to_sr / from_sr)
        return scipy.signal.resample(audio, n_samples).astype(np.float32)
    except ImportError:
        # Fallback: numpy interp
        old_indices = np.arange(len(audio))
        new_len = int(len(audio) * to_sr / from_sr)
        new_indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(new_indices, old_indices, audio).astype(np.float32)


def normalize(audio: np.ndarray, target_db: float = -14.0) -> np.ndarray:
    """Normalize audio to target dBFS."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    target_rms = 10 ** (target_db / 20)
    return (audio * (target_rms / rms)).clip(-1.0, 1.0).astype(np.float32)


def fade(audio: np.ndarray, fade_in_ms: int = 50, fade_out_ms: int = 100, sr: int = OUTPUT_SR) -> np.ndarray:
    """Apply fade in and fade out."""
    audio = audio.copy()
    fi = int(fade_in_ms * sr / 1000)
    fo = int(fade_out_ms * sr / 1000)
    if fi > 0 and fi < len(audio):
        audio[:fi] *= np.linspace(0, 1, fi)
    if fo > 0 and fo < len(audio):
        audio[-fo:] *= np.linspace(1, 0, fo)
    return audio


def mix(
    instrumental: np.ndarray,
    vocals: np.ndarray,
    instrumental_vol: float = 0.65,   # instrumental slightly lower to let vocals through
    vocal_vol: float = 0.90,
) -> np.ndarray:
    """
    Mix instrumental and vocals together.
    Pads/trims shorter track to match the longer one.
    """
    max_len = max(len(instrumental), len(vocals))
    inst = np.pad(instrumental, (0, max_len - len(instrumental)))
    vox  = np.pad(vocals,       (0, max_len - len(vocals)))
    mixed = inst * instrumental_vol + vox * vocal_vol
    return mixed.clip(-1.0, 1.0).astype(np.float32)


def pad_to_duration(audio: np.ndarray, target_sec: int, sr: int) -> np.ndarray:
    """Loop or trim audio to exact target duration."""
    target_samples = target_sec * sr
    if len(audio) >= target_samples:
        return audio[:target_samples]
    # Loop
    reps = math.ceil(target_samples / len(audio))
    return np.tile(audio, reps)[:target_samples]


# ── Song Assembler ─────────────────────────────────────────────────────────────

class SongAssembler:
    def __init__(
        self,
        musicgen_size: str = "small",   # small / medium / large
        bark_small: bool = True,
        device: str = "auto",
    ):
        self.musicgen_size = musicgen_size
        self.bark_small    = bark_small
        self.device        = device
        self._inst_gen     = None
        self._vocal_gen    = None

    @property
    def inst_gen(self):
        if self._inst_gen is None:
            from src.audio.instrumental_generator import InstrumentalGenerator
            self._inst_gen = InstrumentalGenerator(self.musicgen_size, self.device)
        return self._inst_gen

    @property
    def vocal_gen(self):
        if self._vocal_gen is None:
            from src.audio.vocal_generator import VocalGenerator
            self._vocal_gen = VocalGenerator(self.device, self.bark_small)
        return self._vocal_gen

    def generate_song(
        self,
        spec: SongSpec,
        skip_vocals: bool = False,
        skip_instrumental: bool = False,
    ) -> GeneratedSong:
        """
        Full song generation pipeline.
        1. Generate instrumental per section
        2. Generate vocals per section (if lyrics provided)
        3. Mix and arrange
        4. Save to disk
        """
        out_dir = Path(spec.output_dir) / spec.title.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Generating song: {spec.title}")
        print(f"Genre: {spec.genre} | BPM: {spec.bpm} | Mood: {spec.mood} | Lang: {spec.language}")
        print(f"Sections: {[s[0] for s in spec.sections]}")
        print(f"{'='*60}")

        # ── Step 1: Instrumentals ─────────────────────────────────────────────
        instrumentals: dict[str, np.ndarray] = {}
        if not skip_instrumental:
            print("\n[1/3] Generating instrumentals...")
            for section_name, duration in spec.sections:
                section_modifier = {
                    "intro":  "intro, building up, no vocals",
                    "verse1": "verse, mid energy",
                    "verse2": "verse, mid energy, slight variation",
                    "chorus": "chorus, most energetic, full arrangement",
                    "chorus2":"chorus, most energetic",
                    "bridge": "bridge, breakdown, stripped back",
                    "outro":  "outro, fading out",
                }.get(section_name, "")
                audio = self.inst_gen.generate(
                    genre=spec.genre,
                    bpm=spec.bpm,
                    mood=spec.mood,
                    duration=duration,
                    extra_prompt=section_modifier,
                )
                # Resample to output SR
                audio = resample(audio, INSTRUMENTAL_SR, OUTPUT_SR)
                audio = pad_to_duration(audio, duration, OUTPUT_SR)
                audio = normalize(audio, target_db=-18.0)
                instrumentals[section_name] = audio
                # Save section
                sf.write(str(out_dir / f"inst_{section_name}.wav"), audio, OUTPUT_SR)
                print(f"  ✓ {section_name}: {len(audio)/OUTPUT_SR:.1f}s")

        # ── Step 2: Vocals ────────────────────────────────────────────────────
        vocals: dict[str, np.ndarray] = {}
        if not skip_vocals and spec.lyrics:
            print("\n[2/3] Generating vocals...")
            singing = spec.genre not in ["hip_hop", "trap", "drill", "boom_bap", "uk_rap", "grime", "french_rap"]
            for section_name, lines in spec.lyrics.items():
                if not lines:
                    continue
                audio = self.vocal_gen.generate_verse(
                    lines,
                    language=spec.language,
                    voice_idx=spec.voice_idx,
                    singing=singing,
                )
                # Resample to output SR
                audio = resample(audio, VOCAL_SR, OUTPUT_SR)
                audio = normalize(audio, target_db=-14.0)
                vocals[section_name] = audio
                sf.write(str(out_dir / f"vocals_{section_name}.wav"), audio, OUTPUT_SR)
                print(f"  ✓ {section_name} vocals: {len(audio)/OUTPUT_SR:.1f}s")

        # ── Step 3: Mix + arrange ─────────────────────────────────────────────
        print("\n[3/3] Mixing and arranging...")
        full_sections = []

        for section_name, duration in spec.sections:
            inst  = instrumentals.get(section_name, np.zeros(duration * OUTPUT_SR, dtype=np.float32))
            vocal = vocals.get(section_name, np.zeros(0, dtype=np.float32))

            if len(vocal) > 0:
                section_audio = mix(inst, vocal)
            else:
                section_audio = inst

            section_audio = fade(section_audio, fade_in_ms=30, fade_out_ms=80, sr=OUTPUT_SR)
            full_sections.append(section_audio)
            print(f"  ✓ Mixed {section_name}: {len(section_audio)/OUTPUT_SR:.1f}s")

        full_song = np.concatenate(full_sections)
        full_song = normalize(full_song, target_db=-12.0)

        # Save final WAV
        wav_path = str(out_dir / f"{spec.title.replace(' ', '_')}.wav")
        sf.write(wav_path, full_song, OUTPUT_SR)
        print(f"\nFull song WAV: {wav_path} ({len(full_song)/OUTPUT_SR:.1f}s)")

        # Convert to MP3
        mp3_path = wav_path.replace(".wav", ".mp3")
        try:
            from pydub import AudioSegment
            AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3", bitrate="192k")
            print(f"Full song MP3: {mp3_path}")
            output_path = mp3_path
        except Exception:
            output_path = wav_path

        # Save metadata
        meta = {
            "title":    spec.title,
            "genre":    spec.genre,
            "bpm":      spec.bpm,
            "mood":     spec.mood,
            "language": spec.language,
            "duration": round(len(full_song) / OUTPUT_SR, 1),
            "sections": [s[0] for s in spec.sections],
            "output":   output_path,
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\nDone! Song saved to: {out_dir}")
        return GeneratedSong(
            spec=spec,
            instrumental=instrumentals,
            vocals=vocals,
            mixed=full_song,
            output_path=output_path,
        )


# ── Quick generation function ──────────────────────────────────────────────────

def generate_song(
    title: str,
    genre: str,
    bpm: int,
    mood: str,
    language: str,
    lyrics: dict[str, list[str]],
    duration: int = 180,
    musicgen_size: str = "small",
    output_dir: str = "output/songs",
) -> GeneratedSong:
    """One-shot song generation."""
    spec = SongSpec(
        title=title,
        genre=genre,
        bpm=bpm,
        mood=mood,
        language=language,
        lyrics=lyrics,
        duration_sec=duration,
        output_dir=output_dir,
    )
    assembler = SongAssembler(musicgen_size=musicgen_size)
    return assembler.generate_song(spec)


if __name__ == "__main__":
    # Quick test — instrumental only (no vocals needed)
    spec = SongSpec(
        title="Test Trap Song",
        genre="trap",
        bpm=140,
        mood="dark",
        language="en",
        lyrics={
            "verse1": [
                "I been moving in silence, they can't feel my weight",
                "Every step I take yeah I'm moving with fate",
            ],
            "chorus": [
                "Turn up the heat, diamonds on my wrist",
                "Everything I touch turns to gold I can't miss",
            ],
        },
        sections=[
            ("intro",   8),
            ("verse1",  16),
            ("chorus",  12),
            ("verse2",  16),
            ("chorus2", 12),
            ("outro",   8),
        ],
        output_dir="output/songs",
    )

    assembler = SongAssembler(musicgen_size="small", bark_small=True)
    song = assembler.generate_song(spec, skip_vocals=False)
    print(f"\nGenerated: {song.output_path}")

"""
FastAPI server for the lyrics LLM.

Endpoints:
  POST /generate        — auto-generate a full song section
  POST /cowrite/suggest — suggest N lines in co-write mode
  POST /cowrite/accept  — accept a line, update session context
  GET  /health          — health check
  GET  /genres          — list available genres

Session state is held in memory (per-request for stateless, or via session_id for co-write).
For production: swap session dict for Redis.
"""

import os
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from configs.genres import GENRES, GENRE_DESCRIPTIONS


# ── Lazy model load (loads once on first request) ─────────────────────────────

_engine = None
_sessions: dict[str, object] = {}   # session_id → CoWriteSession


def get_engine():
    global _engine
    if _engine is None:
        model_path = os.getenv("LYRICS_MODEL_PATH", "gpt2")
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        print(f"[server] Loading model: {model_path} on {device}")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=__import__("torch").bfloat16 if device == "cuda" else __import__("torch").float32,
        )
        from src.inference.engine import LyricsEngine
        _engine = LyricsEngine(mdl, tok, device=device, beam_size=int(os.getenv("BEAM_SIZE", "8")))
        print("[server] Model loaded.")
    return _engine


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="God-Tier Lyrics LLM",
    description="Phonetic-aware, genre-native lyrics generation",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    genre: str = Field("hip_hop", description="Genre name")
    section: str = Field("VERSE", description="Song section: VERSE, CHORUS, BRIDGE, etc.")
    arc_token: str = Field("[SETUP]", description="Emotional arc token")
    num_lines: int = Field(8, ge=1, le=32)
    rhyme_scheme: str = Field("AABB", description="AABB, ABAB, ABCB, or free")
    target_syllables: int = Field(10, ge=4, le=20)
    artist_style: Optional[str] = Field(None, description="Artist name for style conditioning")
    seed_lines: list[str] = Field(default_factory=list, description="Fixed anchor lines")

class GenerateResponse(BaseModel):
    lines: list[str]
    phoneme_annotations: list[dict]
    rhyme_scheme_detected: str

class CoWriteStartRequest(BaseModel):
    genre: str = "hip_hop"
    rhyme_scheme: str = "AABB"
    target_syllables: int = 10

class CoWriteStartResponse(BaseModel):
    session_id: str

class SuggestRequest(BaseModel):
    session_id: str
    n: int = Field(3, ge=1, le=5)

class SuggestResponse(BaseModel):
    suggestions: list[dict]  # [{text, phonetic_score, syllable_ok, total_score}]

class AcceptRequest(BaseModel):
    session_id: str
    line: str

class AcceptResponse(BaseModel):
    accepted: str
    total_lines: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("LYRICS_MODEL_PATH", "gpt2")}


@app.get("/genres")
def list_genres():
    return [
        {"name": g, "description": GENRE_DESCRIPTIONS.get(g, "")}
        for g in GENRES
    ]


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    from src.inference.engine import SongMemory, LyricsEngine
    from src.data.phoneme_annotator import annotate_line, annotation_to_dict
    from src.data.rhyme_labeler import detect_scheme
    import numpy as np

    if req.genre not in GENRES:
        raise HTTPException(400, f"Unknown genre '{req.genre}'. Valid: {GENRES}")

    engine = get_engine()

    # Load style vector if requested
    style_vec = None
    if req.artist_style:
        from src.data.style_extractor import load_style_vector
        style_vec = load_style_vector(req.artist_style)

    memory = SongMemory(
        genre=req.genre,
        style_vec=style_vec,
        rhyme_scheme=req.rhyme_scheme,
        target_syllables=req.target_syllables,
    )

    # Seed with fixed anchor lines
    for line in req.seed_lines:
        memory.add_line(line)

    lines = engine.generate_verse(
        memory,
        num_lines=req.num_lines,
        section=req.section,
        arc_token=req.arc_token,
    )

    annotations = [annotation_to_dict(annotate_line(l)) for l in lines]
    scheme_info = detect_scheme(lines) if lines else {}

    return GenerateResponse(
        lines=lines,
        phoneme_annotations=annotations,
        rhyme_scheme_detected=scheme_info.get("scheme_type", "free"),
    )


@app.post("/cowrite/start", response_model=CoWriteStartResponse)
def cowrite_start(req: CoWriteStartRequest):
    from src.inference.engine import CoWriteSession

    engine = get_engine()
    session = CoWriteSession(engine, genre=req.genre, rhyme_scheme=req.rhyme_scheme)
    session.memory.target_syllables = req.target_syllables

    session_id = str(uuid.uuid4())
    _sessions[session_id] = session
    return CoWriteStartResponse(session_id=session_id)


@app.post("/cowrite/suggest", response_model=SuggestResponse)
def cowrite_suggest(req: SuggestRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    candidates = session.suggest(n=req.n)
    return SuggestResponse(
        suggestions=[
            {
                "text": c.text,
                "phonetic_score": round(c.phonetic_score, 3),
                "syllable_ok": c.syllable_ok,
                "novelty_score": round(c.novelty_score, 3),
                "valence_fit": round(c.valence_fit, 3),
                "total_score": round(c.total_score, 3),
            }
            for c in candidates
        ]
    )


@app.post("/cowrite/accept", response_model=AcceptResponse)
def cowrite_accept(req: AcceptRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    session.accept(req.line)
    return AcceptResponse(
        accepted=req.line,
        total_lines=len(session.memory.accepted_lines),
    )


@app.get("/cowrite/song/{session_id}")
def get_song(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {"lyrics": session.get_song()}


@app.delete("/cowrite/session/{session_id}")
def end_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"deleted": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

# Lyric Engine

A phonetic-aware lyrics generation system built on top of large language models. Unlike generic text generators, Lyric Engine treats lyrics as *music* — with stress patterns, rhyme density, and emotional arc baked into the generation process itself.

## What makes it different

Most LLM-based lyric tools generate text and check rhymes afterward. This system enforces phonetic constraints *during* generation via a constrained beam search that scores candidates on rhyme match, syllable count, and emotional fit before emitting a single line.

```
Input:  genre=trap, rhyme_scheme=AABB, section=VERSE, arc=[BUILD]
Output: 8 lines where every end-rhyme is phonetically verified,
        syllable counts are within ±3 of target, and valence
        trajectory matches the requested emotional arc
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   LyricsModel                       │
│                                                     │
│  ┌─────────────┐    ┌──────────────────────────┐   │
│  │ StyleVector │───▶│  Base LLM (Llama 3.1 8B) │   │
│  │  Projector  │    │  + Genre LoRA Adapter     │   │
│  │ (128d→768d) │    │  + General Music Adapter  │   │
│  └─────────────┘    └────────────┬─────────────┘   │
│                                  │ hidden states    │
│                          ┌───────▼──────────┐      │
│                          │  Phonetic Head   │      │
│                          │  (2-layer MLP)   │      │
│                          │  → phoneme logits│      │
│                          └──────────────────┘      │
└─────────────────────────────────────────────────────┘
```

**Dual tokenizer** — every input is tokenized twice in parallel:
- Semantic stream: standard BPE tokens (meaning)
- Phoneme stream: CMU Pronouncing Dictionary ARPAbet tokens (sound)

**Genre LoRA adapters** — one lightweight adapter (~4M params) per genre, merged at inference with weighted blending. Blend two genres: `60% trap + 40% R&B` = interpolated adapter weights, single forward pass.

**Artist style encoder** — 128-dim vector capturing an artist's statistical fingerprint: average syllables per line, rhyme density, unique vocabulary ratio, metaphor cluster centroid (via sentence-transformers), emotional valence distribution. Injected as a prefix token into the embedding space. No verbatim lyric memorization — legally safe.

**Emotional arc modeling** — songs are tagged with section-level arc tokens `[SETUP] → [BUILD] → [RELEASE] → [REFRAME] → [PEAK]`. Valence and arousal are scored per line (TextBlob / fine-tuned RoBERTa). The arc is enforced as a constraint during beam scoring, not just a prompt instruction.

**Constrained beam search** — generates 8 candidate lines, scores each on:
1. Phonetic rhyme match (end-phoneme edit distance)
2. Syllable count match (hard filter ±3)
3. Novelty vs accepted lines (Jaccard distance)
4. Emotional valence fit to arc target

Emits top-1 in auto mode, top-3 in co-write mode.

## Project structure

```
src/
├── data/
│   ├── scraper.py          # Genius API lyrics collection (~5M songs)
│   ├── phoneme_annotator.py # CMU dict + rule-based stress/syllable tagging
│   ├── rhyme_labeler.py    # Phoneme edit-distance rhyme scheme detection
│   ├── valence_scorer.py   # Per-line valence/arousal + arc token assignment
│   └── style_extractor.py  # 128-dim artist style vector extraction
├── model/
│   ├── dual_tokenizer.py   # BPE + ARPAbet parallel token streams
│   ├── phonetic_head.py    # Auxiliary MLP for constrained decoding
│   └── lyrics_model.py     # Full model: LLM + LoRA + style projector + phonetic head
├── training/
│   ├── dataset.py          # Training format assembly + DataLoader
│   ├── sft.py              # Stage 1 (general SFT) + Stage 2 (genre LoRAs)
│   └── rlhf.py             # Stage 3: reward model + PPO via TRL
├── inference/
│   └── engine.py           # Constrained beam search + CoWriteSession
└── api/
    └── server.py           # FastAPI: /generate, /cowrite/*, /health
```

## Supported genres

| Genre | Style |
|---|---|
| Trap | 808 cadence, triplet flow, flex imagery |
| R&B | Melisma-friendly, vulnerability, gospel callbacks |
| Indie | Concrete imagery, emotional restraint, non-obvious metaphors |
| Pop | Hook-first, universal themes, ear-worm phrasing |
| Drill | UK/Chicago variants, street-specific slang |
| Alt/Emo | Stream of consciousness, distorted metaphors |
| Hip-Hop | Wordplay, storytelling, boom-bap cadence |
| Country | Narrative, rural imagery, heartbreak/pride |
| Rock | Anthemic hooks, rebellion |
| Latin | Reggaeton flow, Spanglish, rhythmic wordplay |

## Training pipeline

| Stage | What | Compute |
|---|---|---|
| 1 | General music SFT on ~800M token annotated corpus | 4× A100, ~$800 |
| 2 | Per-genre LoRA adapters (rank 16, 1 epoch each) | 1× A100 per genre, ~$150 total |
| 3 | RLHF/PPO with human preference ratings | 1× A100, ~$200 |
| 4 | Phonetic head training (frozen base) | 1× A100, ~$50 |

Free alternative: Kaggle (30hr/week free GPU) + Google Colab.

## Local development (no GPU needed)

```bash
git clone https://github.com/SMXFREEZE/lyric-engine
cd lyric-engine
pip install -r requirements.txt

# Run smoke tests (uses GPT-2, CPU only)
python scripts/smoke_test.py

# Start API server (dev mode with GPT-2)
python src/api/server.py
```

API runs at `http://localhost:8000`. Try it:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"genre": "trap", "section": "VERSE", "arc_token": "[BUILD]", "num_lines": 8}'
```

## Co-write mode

```bash
# Start a session
curl -X POST http://localhost:8000/cowrite/start \
  -d '{"genre": "rnb", "rhyme_scheme": "ABAB"}'
# → {"session_id": "abc-123"}

# Get 3 suggestions for the next line
curl -X POST http://localhost:8000/cowrite/suggest \
  -d '{"session_id": "abc-123", "n": 3}'

# Accept a line (model updates context for next generation)
curl -X POST http://localhost:8000/cowrite/accept \
  -d '{"session_id": "abc-123", "line": "I been moving in silence"}'
```

## Production serving

```
vLLM on Modal Labs serverless
├── Scale to zero when idle
├── Auto-scale to 10 replicas under load
├── 4-bit quantization (bitsandbytes NF4)
├── Speculative decoding with 1B draft model
└── Target: full verse (8 lines) < 3s on 1× A10G
    Cost: ~$0.002 per song generated
```

## Environment variables

```bash
GENIUS_TOKEN=          # Genius API token for data collection
LYRICS_MODEL_PATH=gpt2 # Path to trained model (defaults to GPT-2 for dev)
VALENCE_MODEL_PATH=    # Optional: fine-tuned RoBERTa for emotion scoring
BEAM_SIZE=8            # Beam search width
HUGGING_FACE_HUB_TOKEN= # Required to download Llama 3.1
```

## Key technical decisions

**Why not just prompt GPT-4?** Prompting can't enforce phonetic constraints at the token level. You can ask GPT-4 to rhyme — it often doesn't, and when it does it's because it got lucky with its training data. This system *cannot* emit a line that violates the rhyme scheme because constraint checking happens before a token is accepted.

**Why LoRA per genre instead of one big model?** Storage efficiency (~32MB per adapter vs retraining 8B params), composability (blend adapters at runtime), and the ability to add new genres without touching the base model.

**Why style vectors instead of fine-tuning on artist lyrics?** Copyright. A style vector is a statistical fingerprint — it captures *how* someone writes, not *what* they wrote. The generated output is original and legally defensible.

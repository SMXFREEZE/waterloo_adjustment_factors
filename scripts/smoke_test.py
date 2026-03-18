"""
Smoke test — runs locally with no GPU, no API keys.
Tests the full pipeline with GPT-2 as the base model.

Run: python scripts/smoke_test.py
"""

import sys
sys.path.insert(0, ".")

def test_phoneme_annotator():
    print("\n-- Phoneme Annotator --")
    from src.data.phoneme_annotator import annotate_line
    ann = annotate_line("I been movin' in silence, they can't feel my weight")
    print(f"  Syllables   : {ann.total_syllables}")
    print(f"  End phoneme : {ann.end_phoneme}")
    print(f"  Stress      : {ann.stress_pattern}")
    assert ann.total_syllables > 0
    print("  PASS")


def test_rhyme_labeler():
    print("\n-- Rhyme Labeler --")
    from src.data.rhyme_labeler import detect_scheme
    lines = [
        "I been movin' in silence, they can't feel my weight",
        "Every step I take, yeah I'm moving with fate",
        "They say the game is cold but I turn up the heat",
        "Diamonds on my wrist while I dance to the beat",
    ]
    result = detect_scheme(lines)
    print(f"  Scheme  : {result['scheme_str']} ({result['scheme_type']})")
    print(f"  Density : {result['rhyme_density']}")
    assert result["rhyme_density"] > 0
    print("  PASS")


def test_valence_scorer():
    print("\n--Valence Scorer --")
    from src.data.valence_scorer import score_line
    em = score_line("I been movin' in silence, they can't feel my weight")
    print(f"  Valence : {em.valence:+.3f}")
    print(f"  Arousal : {em.arousal:.3f}")
    assert -1.0 <= em.valence <= 1.0
    print("  PASS")


def test_dual_tokenizer():
    print("\n--Dual Tokenizer (offline, GPT-2) --")
    from src.model.dual_tokenizer import OfflineDualTokenizer
    tok = OfflineDualTokenizer()
    enc = tok.encode("I been movin' in silence, they can't feel my weight")
    print(f"  Semantic IDs (first 8) : {enc.semantic_ids[:8]}")
    print(f"  Phoneme IDs  (first 8) : {enc.phoneme_ids[:8]}")
    assert len(enc.semantic_ids) == len(enc.phoneme_ids)
    print("  PASS")


def test_phonetic_head():
    print("\n--Phonetic Head --")
    import torch
    from src.model.phonetic_head import PhoneticHead, PhoneticConstraintScorer
    from src.model.dual_tokenizer import PHONEME_TO_ID

    head = PhoneticHead(d_model=64, hidden=32)
    scorer = PhoneticConstraintScorer(head, device="cpu")

    dummy = torch.randn(4, 5, 64)  # batch=4, seq=5, d_model=64
    target_id = PHONEME_TO_ID.get("EY1", 10)
    beam_scores = torch.tensor([-1.0, -1.5, -0.8, -2.0])
    ranked = scorer.rerank_beams(dummy, beam_scores, target_id)
    print(f"  Reranked indices: {ranked}")
    assert len(ranked) == 4
    print("  PASS")


def test_lyrics_model():
    print("\n--Lyrics Model (GPT-2) --")
    import torch
    from src.model.lyrics_model import load_base_model, LyricsModel

    base, tok = load_base_model("gpt2")
    model = LyricsModel(base, d_model=base.config.hidden_size)
    model.eval()

    ids = tok("[VERSE] I been movin in silence", return_tensors="pt")
    with torch.no_grad():
        out = model(
            input_ids=ids["input_ids"],
            attention_mask=ids["attention_mask"],
            style_vec=torch.randn(1, 128),
        )
    print(f"  LM logits shape     : {out['lm_logits'].shape}")
    print(f"  Phoneme logits shape: {out['phoneme_logits'].shape}")
    print("  PASS")


def test_inference_engine():
    print("\n--Inference Engine (GPT-2, 3 beams) --")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.inference.engine import LyricsEngine, SongMemory

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained("gpt2")

    engine = LyricsEngine(mdl, tok, device="cpu", beam_size=3)
    memory = SongMemory(genre="hip_hop", rhyme_scheme="AABB", target_syllables=10)
    memory.sections.append(("[SETUP]", "VERSE"))

    candidates = engine.generate_line(memory, top_n=3)
    print(f"  Got {len(candidates)} candidates:")
    for c in candidates:
        print(f"    [{c.total_score:.2f}] {c.text[:60]}")
    assert len(candidates) > 0
    print("  PASS")


if __name__ == "__main__":
    import traceback
    tests = [
        test_phoneme_annotator,
        test_rhyme_labeler,
        test_valence_scorer,
        test_dual_tokenizer,
        test_phonetic_head,
        test_lyrics_model,
        test_inference_engine,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)

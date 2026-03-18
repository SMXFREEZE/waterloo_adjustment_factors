"""Genre definitions and their LoRA adapter configs."""

GENRES = [
    "trap",
    "rnb",
    "indie",
    "pop",
    "drill",
    "alt_emo",
    "hip_hop",
    "country",
    "rock",
    "latin",
]

GENRE_DESCRIPTIONS = {
    "trap":     "Drug refs coded, 808 cadence, triplet flow patterns, flex imagery",
    "rnb":      "Melisma-friendly lines, vulnerability, second-person intimacy, gospel callbacks",
    "indie":    "Concrete imagery, lowercase aesthetic, non-obvious metaphors, emotional restraint",
    "pop":      "Hook-first structure, universal themes, short syllabic lines, ear-worm phrasing",
    "drill":    "UK/Chicago variants, street-specific slang, threat posturing, slide references",
    "alt_emo":  "Self-referential, stream of consciousness, distorted metaphors, introspective arcs",
    "hip_hop":  "Wordplay, cultural references, storytelling, boom-bap cadence",
    "country":  "Narrative storytelling, rural imagery, heartbreak/pride themes, twang phrasing",
    "rock":     "Power chords energy, rebellion, anthemic hooks, distortion imagery",
    "latin":    "Spanish/Spanglish, reggaeton flow, romantic themes, rhythmic word-play",
}

# Emotional arc control tokens — injected at section boundaries
ARC_TOKENS = ["[SETUP]", "[BUILD]", "[RELEASE]", "[REFRAME]", "[PEAK]", "[OUTRO]"]

# Song structure tokens
SECTION_TOKENS = ["[VERSE]", "[PRECHORUS]", "[CHORUS]", "[BRIDGE]", "[HOOK]", "[OUTRO]"]

# All special tokens combined
SPECIAL_TOKENS = ARC_TOKENS + SECTION_TOKENS + [
    "[GENRE_START]", "[GENRE_END]",
    "[STYLE_START]", "[STYLE_END]",
    "[RHYME_AABB]", "[RHYME_ABAB]", "[RHYME_ABCB]", "[RHYME_FREE]",
]

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

BASE_MODEL_LORA_CONFIG = {
    **LORA_CONFIG,
    "r": 64,  # Wider for the general music SFT adapter
    "lora_alpha": 128,
}

"""
Core lyrics model.

Architecture:
  - Base: Llama 3.1 8B (or GPT-2 for local dev) loaded in 4-bit via bitsandbytes
  - LoRA adapters injected via PEFT
  - Style vector prefix injected via a learned projection into the embedding space
  - PhoneticHead attached to the last hidden state
  - Special structure/arc tokens added to vocabulary
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from configs.genres import LORA_CONFIG, BASE_MODEL_LORA_CONFIG, SPECIAL_TOKENS
from src.model.phonetic_head import PhoneticHead
from src.model.dual_tokenizer import PHONEME_VOCAB_SIZE


DEV_MODEL = "gpt2"          # used when LYRICS_MODEL_PATH not set (no GPU needed)
PROD_MODEL = "meta-llama/Llama-3.1-8B"


def get_model_name() -> str:
    return os.getenv("LYRICS_MODEL_PATH", DEV_MODEL)


def load_base_model(
    model_name: Optional[str] = None,
    use_4bit: bool = False,
    device: str = "cpu",
) -> tuple[PreTrainedModel, AutoTokenizer]:
    name = model_name or get_model_name()
    print(f"[lyrics_model] Loading base model: {name}")

    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens
    new_tokens = [t for t in SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def apply_lora(
    model: PreTrainedModel,
    rank: int = 16,
    alpha: int = 32,
    genre_adapter: bool = True,
) -> PreTrainedModel:
    cfg = LORA_CONFIG.copy()
    cfg["r"] = rank
    cfg["lora_alpha"] = alpha

    # Filter target modules to those that exist in this model
    all_names = {name for name, _ in model.named_modules()}
    valid_targets = [t for t in cfg["target_modules"] if any(t in n for n in all_names)]
    if not valid_targets:
        # GPT-2 fallback modules
        valid_targets = ["c_attn", "c_proj"]
    cfg["target_modules"] = valid_targets

    lora_cfg = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias=cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_cfg)


# ── Style vector projection ───────────────────────────────────────────────────

class StyleProjector(nn.Module):
    """
    Projects a 128-dim artist style vector into d_model-dim space.
    The output is prepended as a prefix token to the input embeddings.
    """
    def __init__(self, style_dim: int = 128, d_model: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(style_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, style_vec: torch.Tensor) -> torch.Tensor:
        """style_vec: (batch, 128) → (batch, 1, d_model)"""
        return self.proj(style_vec).unsqueeze(1)


# ── Full lyrics model wrapper ─────────────────────────────────────────────────

class LyricsModel(nn.Module):
    """
    Combines:
      - Base LLM (with LoRA)
      - StyleProjector (128d → d_model prefix token)
      - PhoneticHead (hidden → phoneme logits)
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        d_model: int = 768,
        style_dim: int = 128,
        phonetic_hidden: int = 256,
    ):
        super().__init__()
        self.base = base_model
        self.style_projector = StyleProjector(style_dim, d_model)
        self.phonetic_head = PhoneticHead(d_model, phonetic_hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        phoneme_ids: Optional[torch.Tensor] = None,
        style_vec: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        # If style vector provided, inject as prefix
        inputs_embeds = None
        if style_vec is not None:
            word_embeds = self.base.get_input_embeddings()(input_ids)  # (B, S, D)
            style_prefix = self.style_projector(style_vec)              # (B, 1, D)
            inputs_embeds = torch.cat([style_prefix, word_embeds], dim=1)
            # Extend attention mask for the prefix token
            prefix_mask = torch.ones(attention_mask.shape[0], 1, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            if labels is not None:
                prefix_label = torch.full((labels.shape[0], 1), -100, device=labels.device)
                labels = torch.cat([prefix_label, labels], dim=1)

        out = self.base(
            input_ids=None if inputs_embeds is not None else input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        result = {
            "lm_loss": out.loss,
            "lm_logits": out.logits,
        }

        # Phonetic head
        if out.hidden_states:
            last_hidden = out.hidden_states[-1]  # (B, S, D)
            phoneme_logits = self.phonetic_head(last_hidden)
            result["phoneme_logits"] = phoneme_logits
            result["last_hidden"] = last_hidden

        return result

    def generate_line(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_vec: Optional[torch.Tensor] = None,
        max_new_tokens: int = 60,
        num_beams: int = 8,
        **kwargs,
    ) -> torch.Tensor:
        """Standard generation — constrained reranking happens in inference engine."""
        return self.base.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            early_stopping=True,
            **kwargs,
        )

    def save(self, path: str):
        self.base.save_pretrained(path)

    @classmethod
    def from_pretrained(
        cls,
        base_name: str = DEV_MODEL,
        lora_path: Optional[str] = None,
        use_4bit: bool = False,
        device: str = "cpu",
    ) -> "LyricsModel":
        base, _ = load_base_model(base_name, use_4bit=use_4bit, device=device)
        if lora_path:
            base = PeftModel.from_pretrained(base, lora_path)
        d_model = base.config.hidden_size
        return cls(base, d_model=d_model)


if __name__ == "__main__":
    print("Loading dev model (GPT-2)...")
    base, tok = load_base_model("gpt2")
    model = LyricsModel(base, d_model=base.config.hidden_size)
    model.eval()

    ids = tok("[VERSE] I been movin' in silence", return_tensors="pt")
    out = model(
        input_ids=ids["input_ids"],
        attention_mask=ids["attention_mask"],
        style_vec=torch.randn(1, 128),
    )
    print(f"LM logits shape     : {out['lm_logits'].shape}")
    print(f"Phoneme logits shape: {out['phoneme_logits'].shape}")
    print("Model OK")

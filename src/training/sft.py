"""
Stage 1 + 2: Supervised Fine-Tuning (SFT)
  - Stage 1: General music SFT on full annotated corpus (LoRA rank 64)
  - Stage 2: Per-genre LoRA adapters (LoRA rank 16, genre-specific subsets)

Run on cloud GPU (4x A100 80GB for Stage 1, 1x A100 for Stage 2):
  python -m src.training.sft --stage 1 --data data/raw/all_songs.jsonl
  python -m src.training.sft --stage 2 --genre trap --data data/raw/trap.jsonl

Estimated cost: ~$800 Stage 1, ~$150 Stage 2 (all genres) on RunPod.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm

from src.model.lyrics_model import load_base_model, apply_lora, LyricsModel
from src.model.phonetic_head import phonetic_head_loss
from src.training.dataset import create_dataloaders


def train_sft(
    stage: int = 1,
    genre: Optional[str] = None,
    data_path: str = "data/raw/all_songs.jsonl",
    val_path: Optional[str] = None,
    base_model: str = "meta-llama/Llama-3.1-8B",
    output_dir: str = "checkpoints",
    batch_size: int = 4,
    grad_accum_steps: int = 8,
    max_length: int = 512,
    epochs: int = 2,
    lr: float = 2e-4,
    lora_rank: int = 64,
    alpha: Optional[int] = None,
    use_4bit: bool = True,
    style_vec_dir: Optional[str] = "data/style_vectors",
    phonetic_loss_weight: float = 0.1,
    viral_conditioning: Optional["np.ndarray"] = None,  # 32-dim viral DNA vector
):
    accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps)

    lora_alpha = alpha if alpha is not None else lora_rank * 2

    if stage == 2:
        assert genre is not None, "Stage 2 requires --genre"
        output_subdir = f"{output_dir}/genre_{genre}"
    else:
        output_subdir = f"{output_dir}/stage1_general"

    # Log viral conditioning if provided
    if viral_conditioning is not None and np.linalg.norm(viral_conditioning) > 0:
        print(f"  Viral DNA conditioning: norm={np.linalg.norm(viral_conditioning):.3f}, "
              f"top dims={viral_conditioning[:4].round(3)}")

    Path(output_subdir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base, tokenizer = load_base_model(base_model, use_4bit=use_4bit and device == "cuda")
    base = apply_lora(base, rank=lora_rank, alpha=lora_alpha)
    base.gradient_checkpointing_enable()
    model = LyricsModel(base, d_model=base.config.hidden_size)

    if accelerator.is_main_process:
        model.base.print_trainable_parameters()

    train_dl, val_dl = create_dataloaders(
        data_path, val_path, tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        style_vec_dir=style_vec_dir,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_dl) // grad_accum_steps) * epochs
    warmup_steps = total_steps // 10
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model, optimizer, train_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, scheduler
    )

    global_step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", disable=not accelerator.is_main_process)

        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                # Inject viral DNA: add as a bias to the style vector
                style_vec = batch.get("style_vec")
                if viral_conditioning is not None and style_vec is not None:
                    viral_t = torch.tensor(viral_conditioning, dtype=style_vec.dtype, device=style_vec.device)
                    # Pad/trim viral vector to match style vec dim
                    sv_dim = style_vec.shape[-1]
                    v_dim  = viral_t.shape[0]
                    if v_dim < sv_dim:
                        viral_t = torch.nn.functional.pad(viral_t, (0, sv_dim - v_dim))
                    else:
                        viral_t = viral_t[:sv_dim]
                    style_vec = style_vec + 0.1 * viral_t.unsqueeze(0)

                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    phoneme_ids=batch.get("phoneme_ids"),
                    style_vec=style_vec,
                    labels=batch["labels"],
                )

                loss = out["lm_loss"]

                # Add phonetic head loss if available
                if "phoneme_logits" in out and "phoneme_ids" in batch:
                    ph_loss = phonetic_head_loss(out["phoneme_logits"], batch["phoneme_ids"])
                    loss = loss + phonetic_loss_weight * ph_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                total_loss += loss.item()

                if accelerator.is_main_process:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

        # Validation
        if val_dl:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dl:
                    out = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    val_loss += out["lm_loss"].item()
            val_loss /= len(val_dl)
            if accelerator.is_main_process:
                print(f"  Val loss: {val_loss:.4f}")

        # Save checkpoint
        if accelerator.is_main_process:
            ckpt_path = f"{output_subdir}/epoch_{epoch+1}"
            accelerator.unwrap_model(model).save(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

    if accelerator.is_main_process:
        print(f"\nTraining complete. Final checkpoint: {output_subdir}/epoch_{epochs}")


if __name__ == "__main__":
    import typer
    app = typer.Typer()

    @app.command()
    def main(
        stage: int = typer.Option(1),
        genre: Optional[str] = typer.Option(None),
        data: str = typer.Option("data/raw/all_songs.jsonl"),
        val: Optional[str] = typer.Option(None),
        base_model: str = typer.Option("meta-llama/Llama-3.1-8B"),
        output_dir: str = typer.Option("checkpoints"),
        batch_size: int = typer.Option(4),
        epochs: int = typer.Option(2),
        lr: float = typer.Option(2e-4),
        lora_rank: int = typer.Option(64),
        no_4bit: bool = typer.Option(False),
    ):
        train_sft(
            stage=stage, genre=genre, data_path=data, val_path=val,
            base_model=base_model, output_dir=output_dir,
            batch_size=batch_size, epochs=epochs, lr=lr,
            lora_rank=lora_rank, use_4bit=not no_4bit,
        )

    app()

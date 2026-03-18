"""
Stage 3: RLHF with PPO via TRL.

Flow:
  1. Load 10k generated lyrics pairs rated by humans (crowdsourced via Prolific)
  2. Train a reward model (Bradley-Terry) on preference pairs
  3. PPO fine-tune the SFT model using the reward model

This is what separates "good" from "god tier" output.

Run (requires GPU):
  python -m src.training.rlhf --sft_path checkpoints/stage1_general/epoch_2
"""

import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import Dataset


# ── Reward model ──────────────────────────────────────────────────────────────

def train_reward_model(
    preferences_jsonl: str,       # [{prompt, chosen, rejected}]
    base_model: str = "meta-llama/Llama-3.1-8B",
    output_dir: str = "checkpoints/reward_model",
    epochs: int = 2,
    lr: float = 1e-5,
):
    """
    Train a Bradley-Terry reward model on human preference pairs.
    preferences_jsonl: path to JSONL with {prompt, chosen, rejected} records.
    """
    from transformers import Trainer, TrainingArguments
    from peft import get_peft_model, LoraConfig, TaskType

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load as sequence classifier (score = reward)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=1,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    # LoRA for reward model too
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_cfg)

    # Load data
    records = []
    with open(preferences_jsonl, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    def tokenize_pair(record):
        chosen_enc = tokenizer(
            record["prompt"] + "\n" + record["chosen"],
            max_length=512, truncation=True, padding="max_length",
        )
        rejected_enc = tokenizer(
            record["prompt"] + "\n" + record["rejected"],
            max_length=512, truncation=True, padding="max_length",
        )
        return {
            "input_ids_chosen": chosen_enc["input_ids"],
            "attention_mask_chosen": chosen_enc["attention_mask"],
            "input_ids_rejected": rejected_enc["input_ids"],
            "attention_mask_rejected": rejected_enc["attention_mask"],
        }

    dataset = Dataset.from_list(records).map(tokenize_pair)

    # Custom Trainer with Bradley-Terry loss
    class RewardTrainer(torch.nn.Module):
        pass  # TRL's RewardTrainer handles this — using here as a placeholder

    # Use TRL RewardTrainer if available
    try:
        from trl import RewardTrainer, RewardConfig
        training_args = RewardConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            max_length=512,
            report_to="none",
        )
        trainer = RewardTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        trainer.train()
        trainer.save_model(output_dir)
        print(f"Reward model saved → {output_dir}")
    except ImportError:
        print("TRL RewardTrainer not available — install trl>=0.9.0")


# ── PPO training ──────────────────────────────────────────────────────────────

def train_ppo(
    sft_path: str,
    reward_model_path: str,
    data_jsonl: str,
    output_dir: str = "checkpoints/rlhf",
    batch_size: int = 4,
    ppo_epochs: int = 1,
    lr: float = 1e-5,
    max_new_tokens: int = 80,
):
    """
    PPO fine-tune the SFT model using the reward model.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(sft_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        sft_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    reward_model.eval()

    ppo_config = PPOConfig(
        model_name=sft_path,
        learning_rate=lr,
        batch_size=batch_size,
        ppo_epochs=ppo_epochs,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        log_with=None,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,   # TRL handles ref model internally
        tokenizer=tokenizer,
    )

    # Load prompts
    prompts = []
    with open(data_jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            genre = r.get("genre", "hip_hop")
            prompts.append(f"[GENRE_START] {genre} [GENRE_END]\n[VERSE] [SETUP]\n")

    print(f"[rlhf] Starting PPO on {len(prompts)} prompts...")
    for epoch in range(ppo_epochs):
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            query_tensors = [
                tokenizer.encode(p, return_tensors="pt").squeeze(0).to(device)
                for p in batch_prompts
            ]
            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
            )
            responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

            # Score with reward model
            rewards = []
            for resp in responses:
                enc = tokenizer(resp, return_tensors="pt", max_length=512, truncation=True)
                with torch.no_grad():
                    score = reward_model(**enc).logits.squeeze().item()
                rewards.append(torch.tensor(score))

            ppo_trainer.step(query_tensors, response_tensors, rewards)

        print(f"[rlhf] Epoch {epoch+1}/{ppo_epochs} complete")

    ppo_trainer.save_pretrained(output_dir)
    print(f"PPO model saved → {output_dir}")


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command("reward")
    def cmd_reward(
        preferences: str = typer.Option("data/preferences.jsonl"),
        base_model: str = typer.Option("meta-llama/Llama-3.1-8B"),
        output: str = typer.Option("checkpoints/reward_model"),
    ):
        train_reward_model(preferences, base_model, output)

    @app.command("ppo")
    def cmd_ppo(
        sft_path: str = typer.Argument(...),
        reward_path: str = typer.Argument(...),
        data: str = typer.Option("data/raw/all_songs.jsonl"),
        output: str = typer.Option("checkpoints/rlhf"),
    ):
        train_ppo(sft_path, reward_path, data, output)

    app()

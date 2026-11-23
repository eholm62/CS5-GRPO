"""
Train a small model with TRL's GRPOTrainer to answer character-count questions.

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_grpo.py --csv letter_counting_dataset.csv
    CUDA_VISIBLE_DEVICES=0 python train_grpo.py --csv data.csv --max_steps 2000 --temperature 1.2
"""

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer


# --------------------------------------------------------------------------- #
# Reward logging callback
# --------------------------------------------------------------------------- #
class RewardLoggerCallback(TrainerCallback):
    """Appends a JSON line per logging step for later plotting."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._fh = open(log_path, "w", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch is not None else None,
            "reward": logs.get("reward"),
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "kl": logs.get("kl"),
            "clip_ratio": logs.get("clip_ratio"),
            "entropy": logs.get("entropy"),
        }
        if entry["reward"] is not None:
            self._fh.write(json.dumps(entry) + "\n")
            self._fh.flush()

    def on_train_end(self, args, state, control, **kwargs):
        self._fh.close()


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        reader = csv.reader(sample)
        first_line = next(reader, None)
        has_header = first_line and first_line[0].strip().lower() == "prompt"

        f.seek(0)
        reader = csv.reader(f)
        if has_header:
            next(reader, None)
        for line in reader:
            if not line or len(line) < 4:
                continue
            prompt, target_count, difficulty, text_used = (c.strip() for c in line[:4])
            if not prompt or not prompt.isdigit() and not _try_int(target_count):
                continue
            rows.append({
                "prompt": prompt,
                "target_count": target_count,
                "difficulty_level": difficulty,
                "text_used": text_used,
            })
    return rows


def _try_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# Reward function (strictly binary)
# --------------------------------------------------------------------------- #
def reward_accuracy(
    completions: List[List[Dict[str, str]]],
    target_count: List[str],
    **_: Any,
) -> List[float]:
    """
    1.0 if the model's completion contains the correct count as its last number.
    0.0 otherwise. No partial credit.
    """
    rewards = []
    for completion, expected in zip(completions, target_count):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        found = re.findall(r"-?\d+", text)
        try:
            expected_int = int(str(expected).strip())
        except (ValueError, TypeError):
            rewards.append(0.0)
            continue
        if found and int(found[-1]) == expected_int:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# --------------------------------------------------------------------------- #
# Final checkpoint saving
# --------------------------------------------------------------------------- #
def is_peft_model(model) -> bool:
    return hasattr(model, "peft_config") and model.peft_config is not None


def save_final_checkpoint(trainer, tokenizer, args, interrupted: bool = False) -> str:
    """Save the final trained weights to <output_dir>/final/ on this machine."""
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    model = trainer.model
    tag = " (partial - training interrupted)" if interrupted else ""

    if is_peft_model(model):
        if args.save_merged:
            print(f"Merging LoRA adapter into base model{tag}...")
            merged = model.merge_and_unload()
            merged.save_pretrained(final_dir, safe_serialization=True)
            print(f"Saved MERGED full model (LoRA folded in) to {final_dir}")
        else:
            trainer.save_model(final_dir)
            print(f"Saved LoRA adapter to {final_dir} "
                  f"(use --save_merged for a full standalone model)")
    else:
        trainer.save_model(final_dir)
        print(f"Saved full model weights to {final_dir}{tag}")

    # Always save the tokenizer so the checkpoint is loadable on its own
    tokenizer.save_pretrained(final_dir)

    # Record the training arguments for reproducibility
    with open(os.path.join(final_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Files in {final_dir}:")
    for name in sorted(os.listdir(final_dir)):
        size_mb = os.path.getsize(os.path.join(final_dir, name)) / (1024 * 1024)
        print(f"  - {name:<40s} {size_mb:10.2f} MB")
    return final_dir


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to the CSV data file.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output_dir", default="./grpo_checkpoints")

    # Training duration
    p.add_argument("--num_epochs", type=int, default=3,
                   help="Number of full passes over the dataset (ignored if --max_steps > 0).")
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Stop after this many optimizer steps (-1 = use --num_epochs).")

    # Batch / sampling
    p.add_argument("--batch_size", type=int, default=8,
                   help="Prompts per training step.")
    p.add_argument("--num_generations", type=int, default=8,
                   help="Completions sampled per prompt (GRPO group size).")
    p.add_argument("--max_completion_length", type=int, default=32,
                   help="Max tokens in a generated completion.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature. Higher = more diverse generations "
                        "(GRPO needs diversity to compute meaningful advantages). "
                        "Try 1.0-1.3.")
    p.add_argument("--top_p", type=float, default=0.95,
                   help="Nucleus sampling cutoff. 0.9-1.0 typical for GRPO.")

    # Optimization
    p.add_argument("--learning_rate", type=float, default=5e-6,
                   help="Peak LR. 5e-6 to 1e-5 typical for GRPO on small models.")
    p.add_argument("--logging_steps", type=int, default=10,
                   help="Log reward every N optimizer steps.")

    # Saving
    p.add_argument("--save_total_limit", type=int, default=2,
                   help="Number of per-epoch checkpoints to keep under output_dir "
                        "(final model is always saved separately). -1 keeps all.")
    p.add_argument("--save_merged", action="store_true",
                   help="With --use_lora: merge the adapter into the base model "
                        "and save a full standalone checkpoint (no PEFT needed at inference).")

    # Misc
    p.add_argument("--use_lora", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    # --- Data ----------------------------------------------------------------- #
    rows = load_csv(args.csv)
    if not rows:
        raise SystemExit("No rows found in CSV.")
    print(f"Loaded {len(rows)} rows from {args.csv}")
    dataset = datasets.Dataset.from_list(rows)

    # --- Model & tokenizer ------------------------------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
    ).to(device)

    peft_config = None
    if args.use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

    # --- GRPO config ------------------------------------------------------------ #
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,          # -1 → use epochs
        beta=0.0,
        report_to=[],
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
    )

    # --- Trainer ------------------------------------------------------------------ #
    reward_log_path = "reward_log.jsonl"
    reward_logger = RewardLoggerCallback(reward_log_path)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        eval_dataset=None,
        reward_funcs=reward_accuracy,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[reward_logger],
    )

    # --- Train (with best-effort save on interrupt) -------------------------------- #
    interrupted = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted — saving final weights before exiting...")

    # --- Save final weights to <output_dir>/final --------------------------------- #
    save_final_checkpoint(trainer, tokenizer, args, interrupted=interrupted)
    print(f"\nReward log:   {reward_log_path}")


if __name__ == "__main__":
    main()
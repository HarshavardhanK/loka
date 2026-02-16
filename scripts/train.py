#!/usr/bin/env python3
"""
Loka Training Script

Train the Loka astrophysics navigation model.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from accelerate import Accelerator
from datasets import load_dataset
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    get_scheduler,
)


# ── Validated training config ────────────────────────────────────────


class TrainConfig(BaseModel):
    """Validated training configuration loaded from YAML.

    Pydantic handles type coercion, defaults, and error messages
    automatically — no more ``config.get("lr", 2e-5)`` everywhere.
    """

    base_model: str = "mistralai/Mistral-7B-v0.1"
    mixed_precision: str = "fp16"
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    num_workers: int = 4
    report_to: List[str] = Field(default_factory=lambda: ["wandb"])
    run_name: str = "loka-training"

    model_config = {"extra": "allow"}

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        """Load and validate config from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Train Loka model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig.from_yaml(args.config)

    # Initialize accelerator for distributed training
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    # Set output directory
    output_dir = args.output_dir or os.environ.get(
        "LOKA_MODEL_DIR", "./models"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator.print("Training Loka model")
    accelerator.print(f"Output directory: {output_dir}")
    accelerator.print(f"Config: {config.model_dump_json(indent=2)}")

    # Load tokenizer and model
    accelerator.print(f"Loading base model: {config.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32,
        device_map="auto" if not accelerator.distributed_type else None,
    )

    # TODO: Load and preprocess astrophysics dataset
    # This would include:
    # - Celestial body positions and ephemeris data
    # - Trajectory planning examples
    # - Mission planning dialogues
    # - Orbital mechanics calculations

    accelerator.print("Loading training dataset...")
    # Placeholder: Replace with actual dataset loading
    # dataset = load_dataset("path/to/loka_dataset")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=config.mixed_precision == "fp16",
        bf16=config.mixed_precision == "bf16",
        dataloader_num_workers=config.num_workers,
        report_to=config.report_to,
        run_name=config.run_name,
    )

    accelerator.print("Training arguments configured")
    accelerator.print(f"Training will use {accelerator.num_processes} processes")

    # TODO: Initialize trainer with actual dataset
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset.get("validation"),
    #     tokenizer=tokenizer,
    # )

    # trainer.train(resume_from_checkpoint=args.resume_from)

    accelerator.print("Training setup complete. Dataset loading not yet implemented.")
    accelerator.print("See src/loka/data/ for dataset preparation utilities.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate Verl-format parquet training data for orbital-mechanics GRPO.

Each row represents one episode prompt with randomised initial conditions.
The output contains ``data_source``, ``prompt`` (chat messages), and
``reward_model`` (ground-truth JSON for the reward function).

Usage
-----
    python scripts/generate_training_data.py \\
        --n-train 10000 --n-val 1000 \\
        --output-dir data/orbital
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the package is importable when running from the repo root.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loka.envs.orbital_transfer import OrbitalTransferEnv  # noqa: E402
from loka.rl.bridge import OrbitalObservationWrapper  # noqa: E402
from loka.rl.curriculum import CurriculumScheduler  # noqa: E402


def _generate_rows(n_episodes: int, seed: int = 0) -> list:
    """Create *n_episodes* prompt rows with randomised initial states."""
    wrapper = OrbitalObservationWrapper()
    env = OrbitalTransferEnv()
    rows = []

    for i in range(n_episodes):
        obs, _info = env.reset(seed=seed + i)
        messages = wrapper.build_messages(
            obs, step=0, max_steps=env.max_steps,
        )
        ground_truth = json.dumps({
            "initial_state": obs.tolist(),
            "a_target": env.a_target,
            "e_target": env.e_target,
            "dv_hohmann": env.dv_hohmann,
            "max_steps": env.max_steps,
        })
        rows.append({
            "data_source": "orbital_transfer",
            "prompt": messages,
            "reward_model": {"ground_truth": ground_truth},
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate Verl-format training parquet for orbital RL.",
    )
    parser.add_argument(
        "--n-train", type=int, default=10000,
        help="Number of training episodes (default: 10 000).",
    )
    parser.add_argument(
        "--n-val", type=int, default=1000,
        help="Number of validation episodes (default: 1 000).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/orbital",
        help="Output directory for parquet files.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed.",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_train} training episodes ...")
    train_rows = _generate_rows(args.n_train, seed=args.seed)
    train_path = out / "train.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path)
    print(f"  -> {train_path}  ({len(train_rows)} rows)")

    print(f"Generating {args.n_val} validation episodes ...")
    val_rows = _generate_rows(args.n_val, seed=args.seed + args.n_train)
    val_path = out / "val.parquet"
    pd.DataFrame(val_rows).to_parquet(val_path)
    print(f"  -> {val_path}  ({len(val_rows)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()

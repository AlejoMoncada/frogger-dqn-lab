"""Train a DQN agent on ALE/Frogger-v5 (RAM observations)."""

import argparse
from dataclasses import replace
from pathlib import Path

from dqn_frogger.config import DQNConfig
from dqn_frogger.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on Frogger RAM.")
    parser.add_argument("--total-steps", type=int, default=None, help="Override total env steps.")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"), help="Directory for CSVs and checkpoint."
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Use tiny values to check the pipeline quickly."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DQNConfig.smoke() if args.smoke else DQNConfig()
    overrides = {}
    if args.total_steps is not None:
        overrides["total_steps"] = args.total_steps
    if args.seed is not None:
        overrides["seed"] = args.seed
    config = replace(config, **overrides)

    metrics = train(config, args.output_dir)
    print(f"done: {metrics}")


if __name__ == "__main__":
    main()

"""Train the pixel DQN (CNN, AtariVectorEnv, Double DQN, n-step) on ALE/Frogger-v5.

Example: uv run python train_pixels.py --smoke --device cuda --output-dir runs/px_smoke_cuda
"""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from dqn_frogger.pixels.config import REPLAY_IMPLS, REWARD_MODES, PixelDQNConfig, default_device
from dqn_frogger.pixels.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pixel DQN on Frogger.")
    parser.add_argument("--total-steps", type=int, default=None, help="Total env steps (all envs).")
    parser.add_argument("--num-envs", type=int, default=None, help="Parallel Frogger envs.")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device for the CNN; auto picks cuda when available.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pixels"))
    parser.add_argument(
        "--smoke", action="store_true", help="Tiny values to check the pipeline quickly."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the run already in --output-dir (clears only the known run artifacts).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    parser.add_argument(
        "--updates-per-vector-step",
        type=int,
        default=None,
        help="Learner updates after each vector step (default 1).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Replay capacity in transitions (default 100000, about 5.6 GB of host RAM).",
    )
    parser.add_argument(
        "--replay-impl",
        choices=list(REPLAY_IMPLS),
        default=None,
        help="Replay layout: frame (default, one stored frame per step) or stacked (legacy, "
        "about 8x more RAM per transition).",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Env steps between resumable checkpoints (default 250000).",
    )
    parser.add_argument(
        "--reward-mode",
        choices=list(REWARD_MODES),
        default=None,
        help="Reward seen by the learner: sign (default), scale (raw * --reward-scale), raw.",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=None,
        help="Multiplier for --reward-mode scale (default 0.1). Episode returns stay raw.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Learner batch size (default 128)."
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="CPU threads for the vector env step (default 8).",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=None,
        help="Env steps between target-network syncs (default 8000).",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=None,
        help="Env steps of warmup before learner updates (default 20000).",
    )
    parser.add_argument(
        "--eps-decay-steps",
        type=int,
        default=None,
        help="Env steps over which epsilon decays to its end value (default 1000000).",
    )
    parser.add_argument("--lr", type=float, default=None, help="Adam learning rate (default 1e-4).")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from <output-dir>/checkpoints/latest.pt (replay is refilled, not restored).",
    )
    parser.add_argument(
        "--force-memory",
        action="store_true",
        help="Start even when the replay estimate plus 3 GB exceeds the RAM free now.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    if choice == "auto":
        return default_device()
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("error: --device cuda requested but CUDA is not available")
    return choice


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    config = PixelDQNConfig.smoke(device=device) if args.smoke else PixelDQNConfig(device=device)
    overrides = {}
    if args.total_steps is not None:
        overrides["total_steps"] = args.total_steps
    if args.num_envs is not None:
        overrides["num_envs"] = args.num_envs
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.updates_per_vector_step is not None:
        overrides["updates_per_vector_step"] = args.updates_per_vector_step
    if args.buffer_size is not None:
        overrides["buffer_size"] = args.buffer_size
    if args.save_interval is not None:
        overrides["save_interval"] = args.save_interval
    if args.reward_mode is not None:
        overrides["reward_mode"] = args.reward_mode
    if args.reward_scale is not None:
        overrides["reward_scale"] = args.reward_scale
    if args.replay_impl is not None:
        overrides["replay_impl"] = args.replay_impl
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.num_threads is not None:
        overrides["num_threads"] = args.num_threads
    if args.target_update_interval is not None:
        overrides["target_update_interval"] = args.target_update_interval
    if args.learning_starts is not None:
        overrides["learning_starts"] = args.learning_starts
    if args.eps_decay_steps is not None:
        overrides["eps_decay_steps"] = args.eps_decay_steps
    if args.lr is not None:
        overrides["lr"] = args.lr
    config = replace(config, **overrides)

    try:
        metrics = train(
            config,
            args.output_dir,
            overwrite=args.overwrite,
            resume=args.resume,
            force_memory=args.force_memory,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(f"done: {metrics}")


if __name__ == "__main__":
    main()

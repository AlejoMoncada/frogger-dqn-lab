"""Hyperparameters for the pixel DQN. Step counts are env steps summed over all vector envs."""

from dataclasses import dataclass

import numpy as np

from dqn_frogger.config import DEFAULT_CHECKPOINT_STEPS
from dqn_frogger.pixels.device import default_device  # re-exported for existing imports

REWARD_MODES = ("sign", "scale", "raw")
REPLAY_IMPLS = ("frame", "stacked")  # frame: shared frame store (default); stacked: legacy layout


def transform_reward(reward: float, mode: str, scale: float) -> float:
    """Reward the learner sees: `sign` clips to -1/0/+1, `scale` multiplies, `raw` passes through."""
    if mode == "sign":
        return float(np.sign(reward))
    if mode == "scale":
        return float(reward) * scale
    if mode == "raw":
        return float(reward)
    raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {mode!r}")


def upgrade_saved_config(saved: dict) -> dict:
    """Map a config saved before `reward_mode` existed (`reward_clip` bool) to the new fields."""
    if "reward_clip" not in saved:
        return saved
    upgraded = {key: value for key, value in saved.items() if key != "reward_clip"}
    upgraded["reward_mode"] = "sign" if saved["reward_clip"] else "raw"
    upgraded.setdefault("reward_scale", 0.1)
    return upgraded


@dataclass
class PixelDQNConfig:
    """All pixel-training hyperparameters, with the defaults used for the home challenge."""

    total_steps: int = 2_000_000
    num_envs: int = 8
    num_threads: int = 8
    buffer_size: int = 100_000  # transitions; ~0.7 GB with replay_impl "frame", ~5.6 GB "stacked"
    replay_impl: str = "frame"  # "frame" (one stored frame per step) or "stacked" (legacy layout)
    batch_size: int = 128
    gamma: float = 0.99
    n_step: int = 3
    lr: float = 1e-4
    learning_starts: int = 20_000
    updates_per_vector_step: int = 1
    target_update_interval: int = 8_000
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 1_000_000
    reward_mode: str = "sign"  # "sign" (clip to -1/0/+1), "scale" (raw * reward_scale), "raw"
    reward_scale: float = 0.1  # used only by reward_mode="scale"; episode returns stay raw
    seed: int = 42
    eval_episodes: int = 5
    eval_interval: int = 100_000
    eval_epsilon: float = 0.01
    checkpoint_steps: tuple[int, ...] = DEFAULT_CHECKPOINT_STEPS
    save_interval: int = 250_000  # periodic resumable checkpoint + checkpoints/latest.pt
    # Life loss ends the TD bootstrap (done=1, n-step window flushed) but not the game: logging,
    # evaluation and milestones still see full 5-life games. Missing in old configs means False.
    life_loss_terminal: bool = False
    device: str = ""  # "" or "auto": cuda, else mps, else cpu (see __post_init__)

    def __post_init__(self) -> None:
        if not self.device or self.device == "auto":
            self.device = default_device()
        if self.total_steps % self.num_envs != 0:
            raise ValueError("total_steps must be a multiple of num_envs")
        if self.num_envs <= 0 or self.n_step < 1:
            raise ValueError("num_envs and n_step must be positive")
        if self.updates_per_vector_step < 0:
            raise ValueError("updates_per_vector_step cannot be negative")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")
        if self.reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {self.reward_mode!r}")
        if self.replay_impl not in REPLAY_IMPLS:
            raise ValueError(f"replay_impl must be one of {REPLAY_IMPLS}, got {self.replay_impl!r}")

    def active_checkpoint_steps(self) -> tuple[int, ...]:
        """Checkpoint steps not beyond `total_steps`, always including the final step."""
        steps = {s for s in self.checkpoint_steps if 0 < s <= self.total_steps}
        steps.add(self.total_steps)
        return tuple(sorted(steps))

    @classmethod
    def smoke(cls, device: str | None = None) -> "PixelDQNConfig":
        """Tiny configuration that exercises every code path in well under two minutes."""
        return cls(
            total_steps=2_000,
            num_envs=4,
            num_threads=4,
            buffer_size=1_000,
            batch_size=32,
            learning_starts=200,
            target_update_interval=400,
            eps_decay_steps=1_000,
            eval_interval=1_000,
            eval_episodes=1,
            checkpoint_steps=(1_000, 2_000),
            device=device or "",
        )

"""Hyperparameters for the pixel DQN. Step counts are env steps summed over all vector envs."""

from dataclasses import dataclass

import torch

from dqn_frogger.config import DEFAULT_CHECKPOINT_STEPS


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PixelDQNConfig:
    """All pixel-training hyperparameters, with the defaults used for the home challenge."""

    total_steps: int = 2_000_000
    num_envs: int = 8
    num_threads: int = 8
    buffer_size: int = 100_000  # ~5.6 GB of host RAM (obs + next_obs as uint8)
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
    reward_clip: bool = True
    seed: int = 42
    eval_episodes: int = 5
    eval_interval: int = 100_000
    eval_epsilon: float = 0.01
    checkpoint_steps: tuple[int, ...] = DEFAULT_CHECKPOINT_STEPS
    save_interval: int = 250_000  # periodic resumable checkpoint + checkpoints/latest.pt
    device: str = ""  # "" resolves to cuda when available, else cpu (see __post_init__)

    def __post_init__(self) -> None:
        if not self.device:
            self.device = default_device()
        if self.total_steps % self.num_envs != 0:
            raise ValueError("total_steps must be a multiple of num_envs")
        if self.num_envs <= 0 or self.n_step < 1:
            raise ValueError("num_envs and n_step must be positive")
        if self.updates_per_vector_step < 0:
            raise ValueError("updates_per_vector_step cannot be negative")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")

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

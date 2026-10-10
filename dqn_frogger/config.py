"""Hyperparameters for DQN training."""

from dataclasses import dataclass


@dataclass
class DQNConfig:
    """All training hyperparameters, with the defaults used for the full run."""

    total_steps: int = 300_000
    buffer_size: int = 100_000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-4
    learning_starts: int = 10_000
    train_freq: int = 4
    target_update_interval: int = 2_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 150_000
    reward_clip: bool = True
    seed: int = 42
    eval_episodes: int = 10
    eval_interval: int = 25_000
    eval_epsilon: float = 0.01
    device: str = "cpu"

    @classmethod
    def smoke(cls) -> "DQNConfig":
        """Tiny configuration that exercises every code path in a few seconds."""
        return cls(
            total_steps=2_000,
            learning_starts=200,
            buffer_size=1_000,
            eval_interval=1_000,
            eval_episodes=1,
            target_update_interval=200,
        )

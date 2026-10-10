"""Fixed-capacity circular replay buffer."""

from typing import NamedTuple

import numpy as np
import torch


class Batch(NamedTuple):
    """A sampled mini-batch of transitions, as torch tensors."""

    obs: torch.Tensor  # float32 (B, obs_dim)
    actions: torch.Tensor  # int64 (B,)
    rewards: torch.Tensor  # float32 (B,)
    next_obs: torch.Tensor  # float32 (B, obs_dim)
    dones: torch.Tensor  # float32 (B,), 1.0 when the episode terminated


class ReplayBuffer:
    """Stores transitions in preallocated numpy arrays, overwriting the oldest first."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._next_index = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition, overwriting the oldest one when full."""
        i = self._next_index
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self._next_index = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        """Draw a uniform random batch (with replacement) as torch tensors."""
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = rng.integers(0, self._size, size=batch_size)
        return Batch(
            obs=torch.from_numpy(self.obs[idx]),
            actions=torch.from_numpy(self.actions[idx]),
            rewards=torch.from_numpy(self.rewards[idx]),
            next_obs=torch.from_numpy(self.next_obs[idx]),
            dones=torch.from_numpy(self.dones[idx]),
        )

    def __len__(self) -> int:
        return self._size

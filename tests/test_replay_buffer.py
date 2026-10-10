"""Tests for the fixed-capacity circular replay buffer."""

import numpy as np
import torch

from dqn_frogger.replay_buffer import ReplayBuffer


def _fill(buffer: ReplayBuffer, n: int, obs_dim: int) -> None:
    for i in range(n):
        obs = np.full(obs_dim, float(i), dtype=np.float32)
        next_obs = np.full(obs_dim, float(i) + 0.5, dtype=np.float32)
        buffer.add(obs, action=i % 3, reward=float(i), next_obs=next_obs, done=(i % 2 == 0))


def test_len_grows_until_capacity_then_stays_fixed() -> None:
    buffer = ReplayBuffer(capacity=4, obs_dim=2)
    _fill(buffer, 3, obs_dim=2)
    assert len(buffer) == 3
    _fill(buffer, 10, obs_dim=2)
    assert len(buffer) == 4


def test_wraparound_keeps_only_most_recent_transitions() -> None:
    buffer = ReplayBuffer(capacity=3, obs_dim=2)
    _fill(buffer, 5, obs_dim=2)  # rewards 0..4 written; only 2, 3, 4 should survive
    rng = np.random.default_rng(0)
    seen: set[float] = set()
    for _ in range(50):
        batch = buffer.sample(batch_size=64, rng=rng)
        seen.update(batch.rewards.tolist())
    assert seen == {2.0, 3.0, 4.0}


def test_sample_shapes_and_dtypes() -> None:
    obs_dim = 128
    buffer = ReplayBuffer(capacity=50, obs_dim=obs_dim)
    _fill(buffer, 20, obs_dim=obs_dim)
    batch = buffer.sample(batch_size=8, rng=np.random.default_rng(1))

    assert batch.obs.shape == (8, obs_dim) and batch.obs.dtype == torch.float32
    assert batch.actions.shape == (8,) and batch.actions.dtype == torch.int64
    assert batch.rewards.shape == (8,) and batch.rewards.dtype == torch.float32
    assert batch.next_obs.shape == (8, obs_dim) and batch.next_obs.dtype == torch.float32
    assert batch.dones.shape == (8,) and batch.dones.dtype == torch.float32


def test_done_is_stored_as_float_mask() -> None:
    buffer = ReplayBuffer(capacity=10, obs_dim=1)
    buffer.add(np.zeros(1, np.float32), 0, 1.0, np.zeros(1, np.float32), done=True)
    batch = buffer.sample(batch_size=4, rng=np.random.default_rng(0))
    assert set(batch.dones.tolist()) == {1.0}

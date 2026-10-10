"""N-step accumulation (termination flush, truncation is not terminal) and replay sampling."""

import numpy as np
import pytest
import torch

from dqn_frogger.pixels.replay import NStepAccumulator, PixelReplayBuffer


class RecordingSink:
    """Stands in for the replay buffer and records every emitted transition."""

    def __init__(self) -> None:
        self.rows: list[tuple] = []

    def add(self, obs, action, reward, next_obs, done, discount) -> None:
        self.rows.append((int(obs[0, 0, 0]), action, reward, int(next_obs[0, 0, 0]), done, discount))


def frame(value: int) -> np.ndarray:
    """A (4,84,84) observation whose first pixel encodes its identity."""
    out = np.zeros((4, 84, 84), dtype=np.uint8)
    out[0, 0, 0] = value
    return out


def test_nstep_emits_discounted_sum_once_window_is_full() -> None:
    sink = RecordingSink()
    acc = NStepAccumulator(n=3, gamma=0.5, sink=sink)
    acc.push(frame(0), 0, 1.0, frame(1), terminated=False, truncated=False)
    acc.push(frame(1), 1, 2.0, frame(2), terminated=False, truncated=False)
    assert sink.rows == []
    acc.push(frame(2), 2, 4.0, frame(3), terminated=False, truncated=False)
    assert sink.rows == [(0, 0, 1.0 + 0.5 * 2.0 + 0.25 * 4.0, 3, False, 0.125)]


def test_nstep_termination_flushes_remaining_as_terminal() -> None:
    sink = RecordingSink()
    acc = NStepAccumulator(n=3, gamma=0.5, sink=sink)
    acc.push(frame(0), 0, 1.0, frame(1), terminated=False, truncated=False)
    acc.push(frame(1), 1, 2.0, frame(2), terminated=False, truncated=False)
    acc.push(frame(2), 2, 4.0, frame(3), terminated=True, truncated=False)
    assert sink.rows == [
        (0, 0, 1.0 + 0.5 * 2.0 + 0.25 * 4.0, 3, True, 0.125),
        (1, 1, 2.0 + 0.5 * 4.0, 3, True, 0.25),
        (2, 2, 4.0, 3, True, 0.5),
    ]


def test_nstep_truncation_is_not_terminal_and_bootstraps_from_final_obs() -> None:
    sink = RecordingSink()
    acc = NStepAccumulator(n=3, gamma=0.5, sink=sink)
    acc.push(frame(0), 0, 1.0, frame(1), terminated=False, truncated=False)
    acc.push(frame(1), 1, 2.0, frame(2), terminated=False, truncated=False)
    acc.push(frame(2), 2, 4.0, frame(3), terminated=False, truncated=True)
    dones = [row[4] for row in sink.rows]
    discounts = [row[5] for row in sink.rows]
    assert dones == [False, False, False]
    assert discounts == pytest.approx([0.125, 0.25, 0.5])
    assert all(row[3] == 3 for row in sink.rows)


def test_nstep_accumulator_is_per_episode_after_flush() -> None:
    sink = RecordingSink()
    acc = NStepAccumulator(n=2, gamma=1.0, sink=sink)
    acc.push(frame(0), 0, 1.0, frame(1), terminated=True, truncated=False)
    # A new episode starts with an empty window: nothing from the previous one leaks in.
    acc.push(frame(5), 1, 1.0, frame(6), terminated=False, truncated=False)
    assert [row[0] for row in sink.rows] == [0]
    acc.push(frame(6), 0, 3.0, frame(7), terminated=False, truncated=False)
    assert sink.rows[-1] == (5, 1, 4.0, 7, False, 1.0)


def test_replay_sampling_shapes_and_dtypes_without_pinning() -> None:
    buffer = PixelReplayBuffer(capacity=16)
    rng = np.random.default_rng(0)
    for i in range(10):
        buffer.add(frame(i), i % 5, float(i), frame(i + 1), done=False, discount=0.97)
    batch = buffer.sample(8, rng, pin_memory=False)
    assert batch.obs.shape == (8, 4, 84, 84) and batch.obs.dtype == torch.uint8
    assert batch.next_obs.shape == (8, 4, 84, 84) and batch.next_obs.dtype == torch.uint8
    assert batch.actions.dtype == torch.int64
    assert batch.rewards.dtype == torch.float32
    assert batch.dones.dtype == torch.float32
    assert batch.discounts.dtype == torch.float32


def test_replay_overwrites_oldest_when_full() -> None:
    buffer = PixelReplayBuffer(capacity=4)
    for i in range(6):
        buffer.add(frame(i), 0, 0.0, frame(i), done=False, discount=1.0)
    assert len(buffer) == 4
    assert sorted(buffer.obs[:, 0, 0, 0].tolist()) == [2, 3, 4, 5]


def test_replay_estimated_bytes_counts_obs_and_next_obs() -> None:
    assert PixelReplayBuffer.estimated_bytes(200_000) >= 2 * 200_000 * 4 * 84 * 84

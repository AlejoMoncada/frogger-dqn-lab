"""Uint8 replay buffer for stacked frames, plus per-env n-step accumulation.

Memory: each stored transition keeps obs and next_obs as uint8 (4, 84, 84), i.e.
2 * 28 224 bytes. Capacity 200 000 therefore needs about 11.3 GB of host RAM (not 5.6 GB:
the next observation is stored explicitly, not reconstructed from neighbouring rows).

N-step rule: a window of up to n rewards is summed with gamma^k weights and stored with
discount gamma^m, where m is the number of rewards actually summed. A terminal ends the
episode and flushes every pending start state with done=1. A truncation (time limit) flushes
with done=0, so the learner still bootstraps from the final observation.
"""

from collections import deque
from typing import Any, NamedTuple, Protocol

import numpy as np
import torch

from dqn_frogger.pixels.network import FRAME_SIZE, FRAME_STACK

OBS_SHAPE = (FRAME_STACK, FRAME_SIZE, FRAME_SIZE)


class PixelBatch(NamedTuple):
    """A sampled mini-batch. Observations stay uint8; the network scales them."""

    obs: torch.Tensor  # uint8 (B, 4, 84, 84)
    actions: torch.Tensor  # int64 (B,)
    rewards: torch.Tensor  # float32 (B,), n-step discounted sum of (clipped) rewards
    next_obs: torch.Tensor  # uint8 (B, 4, 84, 84)
    dones: torch.Tensor  # float32 (B,), 1.0 when the episode terminated
    discounts: torch.Tensor  # float32 (B,), gamma^m for the m rewards summed


class Sink(Protocol):
    def add(self, obs, action, reward, next_obs, done, discount) -> None: ...


class PixelReplayBuffer:
    """Fixed-capacity ring buffer of n-step transitions in host RAM."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.obs = np.zeros((capacity, *OBS_SHAPE), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *OBS_SHAPE), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._next_index = 0
        self._size = 0

    @staticmethod
    def estimated_bytes(capacity: int) -> int:
        """Host memory held by the arrays for a given capacity."""
        per_transition = 2 * int(np.prod(OBS_SHAPE)) + 8 + 4 + 4 + 4
        return capacity * per_transition

    def add(self, obs, action: int, reward: float, next_obs, done: bool, discount: float) -> None:
        """Store one transition (copied), overwriting the oldest one when full."""
        i = self._next_index
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.discounts[i] = discount
        self.dones[i] = float(done)
        self._next_index = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def stream(self, env_id: int) -> "StackedStream":
        """Per-env handle used by the trainer; its references are the stacks themselves."""
        return StackedStream()

    def sample(self, batch_size: int, rng: np.random.Generator, pin_memory: bool = False) -> PixelBatch:
        """Uniform batch with replacement. Pinned host tensors make the GPU copy asynchronous."""
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = rng.integers(0, self._size, size=batch_size)
        tensors = [
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.dones[idx]),
            torch.from_numpy(self.discounts[idx]),
        ]
        if pin_memory:
            tensors = [t.pin_memory() for t in tensors]
        return PixelBatch(*tensors)

    def __len__(self) -> int:
        return self._size


class StackedStream:
    """Reference handle for PixelReplayBuffer: a reference is a private copy of the stack."""

    def start(self, stack) -> np.ndarray:
        """Begin an episode with its reset observation."""
        return np.array(stack, copy=True)

    def step(self, stack) -> np.ndarray:
        """Reference for the observation after one real step."""
        return np.array(stack, copy=True)


def _own(payload):
    """Copy array payloads so the window never aliases the caller's buffers. Refs pass through."""
    return np.array(payload, copy=True) if isinstance(payload, np.ndarray) else payload


class NStepAccumulator:
    """Turns one environment's step stream into n-step transitions for a sink.

    Call `push` once per real environment step. After a terminal or truncation the window is
    flushed, so the next `push` belongs to a new episode. Observations are whatever the sink
    accepts: stacks for PixelReplayBuffer, frame references for FrameReplay.
    """

    def __init__(self, n: int, gamma: float, sink: Sink) -> None:
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.gamma = gamma
        self.sink = sink
        self._pending: deque[tuple[Any, int, float]] = deque()

    def push(self, obs, action: int, reward: float, next_obs, terminated: bool, truncated: bool) -> None:
        self._pending.append((_own(obs), int(action), float(reward)))
        if terminated or truncated:
            while self._pending:
                self._emit(next_obs, done=terminated)
        elif len(self._pending) == self.n:
            self._emit(next_obs, done=False)

    def _emit(self, next_obs, done: bool) -> None:
        obs0, action0, _ = self._pending[0]
        total = 0.0
        for k, (_, _, reward) in enumerate(self._pending):
            total += (self.gamma**k) * reward
        discount = self.gamma ** len(self._pending)
        self._pending.popleft()
        self.sink.add(obs0, action0, total, next_obs, done, discount)

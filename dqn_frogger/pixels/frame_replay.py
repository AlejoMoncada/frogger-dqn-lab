"""Frame-sharing replay: each 84x84 frame is stored once per env stream; transitions keep indices.

A stacked transition (PixelReplayBuffer) keeps obs and next_obs as four frames each, so almost
every frame is stored eight times over. Here each env has a circular frame store with one slot per
env step, and a transition keeps only the absolute stream positions of its newest obs frame and
newest next_obs frame, plus the episode start. Consecutive transitions share frames.

Exactness: AtariVectorEnv's stack is a zero-filled shift register. The reset stack is
[0, 0, 0, f] and each step's first three channels equal the previous step's last three
(checked on 4488 steps). So a stack is fully determined by its newest position p and the episode
start q: channel k holds frame p-3+k, or zeros when p-3+k < q. Frames never cross episodes.

Lifecycle: a transition is valid while every frame it reads is still in its env's store. When
a slot is about to be overwritten, the transitions that read that frame are evicted (they leave
the sample pool; their ring slot is reused later). Each env's transitions form a FIFO in
insertion order, so both eviction paths always take the oldest transition of that env.

Use `buffer.stream(env_id)`: `start(reset_obs)` at every episode start, including the NextStep
reset step that the trainer does not store, and `step(next_obs)` after every real step. Both
return references that NStepAccumulator passes back to `add`.
"""

from collections import deque
from typing import NamedTuple

import numpy as np
import torch

from dqn_frogger.pixels.network import FRAME_SIZE, FRAME_STACK
from dqn_frogger.pixels.replay import OBS_SHAPE, PixelBatch

FRAME_SHAPE = (FRAME_SIZE, FRAME_SIZE)
FRAME_BYTES = FRAME_SIZE * FRAME_SIZE
LOOKBACK = FRAME_STACK - 1  # frames before the newest one in a stack
# env int32; pos_obs, pos_next, episode_start, action int64; reward, discount, done float32; alive.
INDEX_BYTES_PER_TRANSITION = 4 + 3 * 8 + 8 + 3 * 4 + 1


class FrameRef(NamedTuple):
    """Where a stack lives: its env, the stream position of its newest frame, the episode start."""

    env: int
    pos: int
    episode_start: int


def default_frames_per_env(capacity: int, num_envs: int, n_step: int) -> int:
    """Frame slots per env: one per transition each env holds, plus n-step and stack slack."""
    return -(-capacity // num_envs) + n_step + 8


class FrameReplay:
    """Fixed-capacity ring of n-step transitions that references a shared per-env frame store."""

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        n_step: int,
        frames_per_env: int | None = None,
    ) -> None:
        if capacity <= 0 or num_envs <= 0:
            raise ValueError("capacity and num_envs must be positive")
        if n_step < 1:
            raise ValueError("n_step must be at least 1")
        slots = default_frames_per_env(capacity, num_envs, n_step) if frames_per_env is None else frames_per_env
        if slots < n_step + 4:
            raise ValueError(
                f"frames_per_env must be at least n_step + 4 = {n_step + 4} so pending n-step "
                "windows keep their frames"
            )
        self.capacity = capacity
        self.num_envs = num_envs
        self.n_step = n_step
        self.frames_per_env = slots
        self.frames = np.zeros((num_envs, slots, *FRAME_SHAPE), dtype=np.uint8)
        self._env = np.zeros(capacity, dtype=np.int32)
        self._pos_obs = np.zeros(capacity, dtype=np.int64)
        self._pos_next = np.zeros(capacity, dtype=np.int64)
        self._episode_start = np.zeros(capacity, dtype=np.int64)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._discounts = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._alive = np.zeros(capacity, dtype=bool)
        self._newest = np.full(num_envs, -1, dtype=np.int64)
        self._current_start = np.zeros(num_envs, dtype=np.int64)
        self._fifos: list[deque[int]] = [deque() for _ in range(num_envs)]
        self._next_index = 0
        self._size = 0  # ring slots in use, alive or evicted
        self._live = 0  # transitions that can be sampled

    @staticmethod
    def estimated_bytes(capacity: int, num_envs: int, n_step: int) -> int:
        """Host memory held by the frame store and the index arrays for a default store."""
        frames = num_envs * default_frames_per_env(capacity, num_envs, n_step) * FRAME_BYTES
        return frames + capacity * INDEX_BYTES_PER_TRANSITION

    @property
    def nbytes(self) -> int:
        arrays = (
            self.frames,
            self._env,
            self._pos_obs,
            self._pos_next,
            self._episode_start,
            self._actions,
            self._rewards,
            self._discounts,
            self._dones,
            self._alive,
        )
        return sum(array.nbytes for array in arrays)

    def stream(self, env_id: int) -> "FrameStream":
        if not 0 <= env_id < self.num_envs:
            raise ValueError(f"env_id must be in [0, {self.num_envs}), got {env_id}")
        return FrameStream(self, env_id)

    def add(self, obs: FrameRef, action: int, reward: float, next_obs: FrameRef, done: bool, discount: float) -> None:
        """Store one transition by reference, overwriting the oldest slot when full."""
        if obs.env != next_obs.env or obs.episode_start != next_obs.episode_start or obs.pos > next_obs.pos:
            raise ValueError("obs and next_obs must come from one episode stream, in order")
        env = obs.env
        newest = int(self._newest[env])
        if next_obs.pos > newest:
            raise ValueError("next_obs refers to a frame that was never stored")
        if self._oldest_needed(obs.pos, obs.episode_start) <= newest - self.frames_per_env:
            raise ValueError("frames for this transition were already overwritten; raise frames_per_env")

        slot = self._next_index
        if self._size == self.capacity:
            if self._alive[slot]:
                self._unlink_oldest(slot)
        else:
            self._size += 1
        self._env[slot] = env
        self._pos_obs[slot] = obs.pos
        self._pos_next[slot] = next_obs.pos
        self._episode_start[slot] = obs.episode_start
        self._actions[slot] = action
        self._rewards[slot] = reward
        self._discounts[slot] = discount
        self._dones[slot] = float(done)
        self._alive[slot] = True
        self._live += 1
        self._fifos[env].append(slot)
        self._next_index = (slot + 1) % self.capacity

    def sample(self, batch_size: int, rng: np.random.Generator, pin_memory: bool = False) -> PixelBatch:
        """Uniform batch over the transitions that are still valid, with replacement."""
        if self._live == 0:
            raise ValueError("cannot sample from an empty buffer")
        if self._live == self._size:
            idx = rng.integers(0, self._size, size=batch_size)
        else:
            alive = np.flatnonzero(self._alive[: self._size])
            idx = alive[rng.integers(0, len(alive), size=batch_size)]
        tensors = [
            torch.from_numpy(self._gather(idx, self._pos_obs)),
            torch.from_numpy(self._actions[idx]),
            torch.from_numpy(self._rewards[idx]),
            torch.from_numpy(self._gather(idx, self._pos_next)),
            torch.from_numpy(self._dones[idx]),
            torch.from_numpy(self._discounts[idx]),
        ]
        if pin_memory:
            tensors = [t.pin_memory() for t in tensors]
        return PixelBatch(*tensors)

    def __len__(self) -> int:
        return self._live

    # Internal -----------------------------------------------------------------

    @staticmethod
    def _oldest_needed(pos: int, episode_start: int) -> int:
        """Oldest frame position a stack reads from the store (earlier channels are zeros)."""
        return max(episode_start, pos - LOOKBACK)

    def _write_frame(self, env: int, frame: np.ndarray) -> int:
        """Append the newest frame of env's stream and evict the transitions that read the slot."""
        pos = int(self._newest[env]) + 1
        replaced = pos - self.frames_per_env
        fifo = self._fifos[env]
        while fifo and self._oldest_needed(int(self._pos_obs[fifo[0]]), int(self._episode_start[fifo[0]])) <= replaced:
            self._evict(fifo.popleft())
        self.frames[env, pos % self.frames_per_env] = frame
        self._newest[env] = pos
        return pos

    def _evict(self, slot: int) -> None:
        self._alive[slot] = False
        self._live -= 1

    def _unlink_oldest(self, slot: int) -> None:
        """The ring is about to reuse `slot`, the globally oldest transition, which is also the
        oldest live transition of its env."""
        fifo = self._fifos[int(self._env[slot])]
        if not fifo or fifo[0] != slot:
            raise RuntimeError("replay bookkeeping out of order")
        fifo.popleft()
        self._evict(slot)

    def _gather(self, idx: np.ndarray, newest: np.ndarray) -> np.ndarray:
        """Rebuild the (B, 4, 84, 84) stacks whose newest frames are `newest[idx]`."""
        env = self._env[idx]
        episode_start = self._episode_start[idx]
        positions = newest[idx][:, None] + np.arange(-LOOKBACK, 1)
        out = self.frames[env[:, None], positions % self.frames_per_env]
        out[positions < episode_start[:, None]] = 0
        assert out.shape[1:] == OBS_SHAPE
        return out


class FrameStream:
    """One env's frame history inside a FrameReplay."""

    def __init__(self, replay: FrameReplay, env: int) -> None:
        self._replay = replay
        self._env = env

    def start(self, stack) -> FrameRef:
        """Begin an episode with its reset observation (stack (4, 84, 84), newest last)."""
        pos = self._replay._write_frame(self._env, stack[-1])
        self._replay._current_start[self._env] = pos
        return FrameRef(self._env, pos, pos)

    def step(self, stack) -> FrameRef:
        """Reference for the observation after one real step of the current episode."""
        pos = self._replay._write_frame(self._env, stack[-1])
        return FrameRef(self._env, pos, int(self._replay._current_start[self._env]))

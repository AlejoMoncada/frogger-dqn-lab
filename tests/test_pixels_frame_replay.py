"""Frame-sharing replay: equivalence with the stacked buffer, eviction safety, trainer wiring.

The scripted env mimics AtariVectorEnv's stack semantics exactly (zero-filled shift register
at reset, NextStep autoreset step after every done). The real-env test checks the same thing
against ale-py's AtariVectorEnv.
"""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from dqn_frogger.pixels import trainer as pixel_trainer
from dqn_frogger.pixels.config import REPLAY_IMPLS, PixelDQNConfig
from dqn_frogger.pixels.frame_replay import FrameReplay, default_frames_per_env
from dqn_frogger.pixels.replay import NStepAccumulator, PixelReplayBuffer
from dqn_frogger.pixels.safety import replay_estimated_bytes
from dqn_frogger.trainer import CONFIG_JSON
from test_pixels_trainer import FakeVectorEnv, tiny_config

FRAME_BYTES = 84 * 84


class ScriptedStackEnv:
    """Mimics AtariVectorEnv: zero-filled shift-register stacks, NextStep autoreset.

    Frames are random, so every stack is distinguishable. Episodes last 1..max_len steps (short
    ones exercise the zero fill) and end with terminated or, sometimes, truncated.
    """

    def __init__(self, num_envs: int, seed: int, max_len: int = 9) -> None:
        self.n = num_envs
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)
        self.hist = np.zeros((num_envs, 4, 84, 84), dtype=np.uint8)
        self.length = np.ones(num_envs, dtype=np.int64)
        self.k = np.zeros(num_envs, dtype=np.int64)
        self.pending = np.zeros(num_envs, dtype=bool)

    def _frames(self) -> np.ndarray:
        return self.rng.integers(0, 256, size=(self.n, 84, 84), dtype=np.uint8)

    def _new_episode_length(self) -> int:
        return int(self.rng.integers(1, self.max_len + 1))

    def reset(self, *, seed=None, options=None):
        self.hist[:] = 0
        self.hist[:, 3] = self._frames()
        self.k[:] = 0
        self.pending[:] = False
        self.length = np.array([self._new_episode_length() for _ in range(self.n)])
        return self.hist.copy(), {}

    def step(self, actions):
        new = self._frames()
        rewards = np.zeros(self.n, dtype=np.float32)
        terminated = np.zeros(self.n, dtype=bool)
        truncated = np.zeros(self.n, dtype=bool)
        for i in range(self.n):
            if self.pending[i]:  # NextStep: the reset observation, action ignored
                self.pending[i] = False
                self.k[i] = 0
                self.length[i] = self._new_episode_length()
                self.hist[i] = 0
                self.hist[i, 3] = new[i]
                continue
            self.k[i] += 1
            self.hist[i, :3] = self.hist[i, 1:].copy()
            self.hist[i, 3] = new[i]
            rewards[i] = float(self.rng.integers(0, 2))
            if self.k[i] == self.length[i]:
                if self.rng.random() < 0.3:
                    truncated[i] = True
                else:
                    terminated[i] = True
                self.pending[i] = True
        return self.hist.copy(), rewards, terminated, truncated, {}

    def close(self) -> None:
        pass


def drive(env, replay, num_envs: int, n_step: int, steps: int, seed: int) -> int:
    """Feed one env's stream into `replay` the way the trainer does. Returns terminal count."""
    streams = [replay.stream(i) for i in range(num_envs)]
    accumulators = [NStepAccumulator(n_step, 0.99, replay) for _ in range(num_envs)]
    obs, _ = env.reset(seed=np.arange(seed, seed + num_envs))
    refs = [streams[i].start(obs[i]) for i in range(num_envs)]
    skip = np.zeros(num_envs, dtype=bool)
    action_rng = np.random.default_rng(seed)
    terminals = 0
    for _ in range(steps):
        actions = action_rng.integers(0, 5, size=num_envs)
        next_obs, rewards, terminated, truncated, _ = env.step(actions)
        for i in range(num_envs):
            if skip[i]:
                skip[i] = False
                refs[i] = streams[i].start(next_obs[i])
                continue
            next_ref = streams[i].step(next_obs[i])
            accumulators[i].push(
                refs[i],
                int(actions[i]),
                float(rewards[i]),
                next_ref,
                bool(terminated[i]),
                bool(truncated[i]),
            )
            refs[i] = next_ref
            if terminated[i] or truncated[i]:
                terminals += 1
                skip[i] = True
    return terminals


def assert_batches_equal(a, b) -> None:
    for name in ("obs", "actions", "rewards", "next_obs", "dones", "discounts"):
        left, right = getattr(a, name), getattr(b, name)
        assert left.dtype == right.dtype, name
        assert torch.equal(left, right), name


@pytest.mark.parametrize(
    ("num_envs", "n_step", "capacity"),
    [
        (3, 3, 1000),  # no wraparound
        (3, 3, 40),  # wraparound, many overwrites
        (2, 1, 17),  # one-step returns, odd capacity
        (4, 5, 64),  # longer n-step windows
    ],
)
def test_frame_replay_samples_identically_to_stacked_on_scripted_stream(
    num_envs, n_step, capacity
) -> None:
    stacked = PixelReplayBuffer(capacity)
    frame = FrameReplay(capacity, num_envs, n_step)
    drive(ScriptedStackEnv(num_envs, seed=3), stacked, num_envs, n_step, steps=300, seed=3)
    drive(ScriptedStackEnv(num_envs, seed=3), frame, num_envs, n_step, steps=300, seed=3)

    assert len(frame) == len(stacked)
    for sample_seed in range(3):
        a = stacked.sample(64, np.random.default_rng(sample_seed))
        b = frame.sample(64, np.random.default_rng(sample_seed))
        assert_batches_equal(a, b)


def test_frame_replay_matches_real_vector_env_stacks() -> None:
    from dqn_frogger.pixels.vector_env import make_vector_env

    num_envs, n_step, steps = 3, 3, 400
    stacked = PixelReplayBuffer(2000)
    frame = FrameReplay(2000, num_envs, n_step)
    terminals = drive(
        make_vector_env(num_envs=num_envs, num_threads=2),
        stacked,
        num_envs,
        n_step,
        steps,
        seed=7,
    )
    drive(make_vector_env(num_envs=num_envs, num_threads=2), frame, num_envs, n_step, steps, 7)
    assert terminals > 0, "the stream must include at least one episode end"

    assert len(frame) == len(stacked)
    assert_batches_equal(
        stacked.sample(128, np.random.default_rng(1)), frame.sample(128, np.random.default_rng(1))
    )


def test_eviction_never_samples_a_transition_whose_frames_were_overwritten() -> None:
    num_envs, n_step = 2, 3
    reference = PixelReplayBuffer(10_000)  # never wraps: every transition is kept here
    drive(ScriptedStackEnv(num_envs, seed=11), reference, num_envs, n_step, 300, seed=11)

    # A frame store far too small for the ring: most transitions lose frames and must be dropped.
    frame = FrameReplay(60, num_envs, n_step, frames_per_env=n_step + 4)
    drive(ScriptedStackEnv(num_envs, seed=11), frame, num_envs, n_step, 300, seed=11)
    assert 0 < len(frame) < 60

    rng = np.random.default_rng(0)
    batch = frame.sample(20, rng)
    reference_obs = torch.from_numpy(
        np.stack([reference.obs[i] for i in range(len(reference))])
    )
    reference_next = torch.from_numpy(
        np.stack([reference.next_obs[i] for i in range(len(reference))])
    )
    for j in range(batch.obs.shape[0]):
        match = (
            (reference_obs == batch.obs[j]).flatten(1).all(dim=1)
            & (reference_next == batch.next_obs[j]).flatten(1).all(dim=1)
        )
        assert bool(match.any()), f"sampled transition {j} is not a real transition"


def test_eviction_keeps_sampling_consistent_when_frames_are_overwritten() -> None:
    frame = FrameReplay(capacity=10, num_envs=1, n_step=1, frames_per_env=6)
    stream = frame.stream(0)
    rng = np.random.default_rng(5)
    ref = stream.start(np.zeros((4, 84, 84), dtype=np.uint8))
    for _ in range(20):
        nxt = stream.step(rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8))
        frame.add(ref, 0, 0.0, nxt, False, 0.99)
        ref = nxt
    assert 1 <= len(frame) <= 3  # only transitions whose frames are still in the 6-slot store
    batch = frame.sample(8, np.random.default_rng(0))
    assert batch.obs.shape == (8, 4, 84, 84)


def test_zero_fill_at_episode_start_never_reads_previous_episode() -> None:
    frame = FrameReplay(capacity=8, num_envs=1, n_step=1)
    stream = frame.stream(0)

    def stack(*values: int) -> np.ndarray:
        """Newest channels hold `values` (oldest first); earlier channels are zero."""
        out = np.zeros((4, 84, 84), dtype=np.uint8)
        for k, value in enumerate(values):
            out[4 - len(values) + k] = value
        return out

    # Episode A (frame value 9) ends after one step.
    a_ref = stream.start(stack(9))
    a_next = stream.step(stack(9, 9))
    frame.add(a_ref, 0, 0.0, a_next, True, 0.99)
    # Episode B starts right after; its first stack must not borrow A's frames.
    b_ref = stream.start(stack(5))
    b_next = stream.step(stack(5, 6))
    frame.add(b_ref, 1, 1.0, b_next, False, 0.99)

    batch = frame.sample(64, np.random.default_rng(0))
    for j in range(64):
        if int(batch.actions[j]) == 1:
            assert torch.equal(batch.obs[j], torch.from_numpy(stack(5)))
            assert torch.equal(batch.next_obs[j], torch.from_numpy(stack(5, 6)))
    assert (batch.actions == 1).any()


def test_autoreset_steps_are_not_stored(tmp_path, monkeypatch) -> None:
    stored: list[FrameReplay] = []

    class RecordingReplay(FrameReplay):
        def __init__(self, capacity, num_envs, n_step):
            super().__init__(capacity, num_envs, n_step)
            stored.append(self)

    monkeypatch.setattr(pixel_trainer, "FrameReplay", RecordingReplay)
    monkeypatch.setattr(
        pixel_trainer,
        "make_vector_env",
        lambda num_envs, num_threads=1, **_: FakeVectorEnv(num_envs, num_threads),
    )
    out = tmp_path / "run"
    config = replace(
        tiny_config(out),
        num_envs=2,
        total_steps=20,
        learning_starts=10**6,
        eval_interval=10**6,
        checkpoint_steps=(),
        n_step=1,
        replay_impl="frame",
    )
    pixel_trainer.train(config, out)

    # Episodes last 5 steps; the 6th step of each env is the NextStep reset step: 10 vector steps
    # give 9 real transitions per env.
    assert len(stored[0]) == 18


def test_config_records_replay_impl(tmp_path) -> None:
    out = tmp_path / "run"
    config = replace(
        tiny_config(out),
        total_steps=40,
        learning_starts=10**6,
        eval_interval=10**6,
        checkpoint_steps=(),
    )
    pixel_trainer.train(config, out)
    saved = json.loads((out / CONFIG_JSON).read_text(encoding="utf-8"))
    assert saved["config"]["replay_impl"] == "frame"


def test_config_default_and_validation() -> None:
    assert PixelDQNConfig(device="cpu").replay_impl == "frame"
    assert REPLAY_IMPLS == ("frame", "stacked")
    with pytest.raises(ValueError, match="replay_impl"):
        PixelDQNConfig(device="cpu", replay_impl="sparse")


def test_memory_estimate_is_one_frame_per_transition_and_matches_arrays() -> None:
    estimate = FrameReplay.estimated_bytes(200_000, 8, 3)
    assert estimate <= 1.5e9
    assert estimate / 200_000 < FRAME_BYTES * 1.1  # about one frame plus index overhead
    small = FrameReplay(1000, 4, 3)
    assert small.nbytes == FrameReplay.estimated_bytes(1000, 4, 3)
    assert default_frames_per_env(200_000, 8, 3) == 25_000 + 3 + 8


def test_safety_uses_the_selected_replay_estimate() -> None:
    frame = replay_estimated_bytes("frame", 200_000, 8, 3)
    stacked = replay_estimated_bytes("stacked", 200_000, 8, 3)
    assert frame == FrameReplay.estimated_bytes(200_000, 8, 3)
    assert stacked == PixelReplayBuffer.estimated_bytes(200_000)
    assert frame < stacked / 7
    with pytest.raises(ValueError, match="replay_impl"):
        replay_estimated_bytes("sparse", 1, 1, 1)


def test_frame_store_needs_room_for_pending_windows() -> None:
    with pytest.raises(ValueError, match="frames_per_env"):
        FrameReplay(10, 1, n_step=3, frames_per_env=6)

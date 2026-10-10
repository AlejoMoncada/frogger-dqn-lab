"""Pixel trainer smoke test on a tiny fake vector env (no ALE, CPU only)."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from dqn_frogger.pixels import trainer as pixel_trainer
from dqn_frogger.pixels.config import PixelDQNConfig
from dqn_frogger.trainer import EPISODE_HEADER


class FakeVectorEnv:
    """Mimics AtariVectorEnv's NextStep autoreset: the step after a done returns the reset
    observation with reward 0 and terminated False; the supplied action is ignored.

    Every episode lasts `episode_len` steps, ends with terminated=True and lives reaching 0.
    """

    def __init__(self, num_envs: int, num_threads: int = 1, episode_len: int = 5) -> None:
        self.num_envs = num_envs
        self.episode_len = episode_len
        self.single_action_space = gym.spaces.Discrete(5)
        self._steps = np.zeros(num_envs, dtype=int)
        self._pending_reset = np.zeros(num_envs, dtype=bool)
        self._lives = np.full(num_envs, 4, dtype=np.int64)

    def _obs(self) -> np.ndarray:
        return np.full((self.num_envs, 4, 84, 84), int(self._steps.sum()) % 256, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        self._steps[:] = 0
        self._pending_reset[:] = False
        self._lives[:] = 4
        return self._obs(), {"lives": self._lives.copy()}

    def step(self, actions):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            if self._pending_reset[i]:
                self._pending_reset[i] = False
                self._steps[i] = 0
                self._lives[i] = 4
                continue
            self._steps[i] += 1
            rewards[i] = 1.0 if self._steps[i] % 2 == 0 else 0.0
            if self._steps[i] >= self.episode_len:
                terminated[i] = True
                self._lives[i] = 0
                self._pending_reset[i] = True
        info = {"lives": self._lives.copy()}
        return self._obs(), rewards, terminated, np.zeros(self.num_envs, dtype=bool), info

    def close(self) -> None:
        pass


def tiny_config(output_dir: Path) -> PixelDQNConfig:
    """Smoke settings shrunk further so the CPU test stays in the seconds range."""
    return replace(
        PixelDQNConfig.smoke(device="cpu"),
        total_steps=400,
        learning_starts=40,
        batch_size=8,
        eval_interval=200,
        checkpoint_steps=(200,),
        target_update_interval=100,
    )


def test_smoke_run_writes_csv_timing_and_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        pixel_trainer,
        "make_vector_env",
        lambda num_envs, num_threads=1, **_: FakeVectorEnv(num_envs, num_threads),
    )
    out = tmp_path / "run"
    config = tiny_config(out)

    metrics = pixel_trainer.train(config, out)

    with (out / "episodes.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == EPISODE_HEADER
    assert len(rows) > 1
    assert int(rows[1][0]) == 1

    timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    assert timing["device"] == "cpu"
    assert timing["num_envs"] == config.num_envs
    assert timing["total_steps"] == config.total_steps
    assert timing["steps_per_second"] > 0
    breakdown = timing["time_breakdown_seconds"]
    assert set(breakdown) >= {"env_step", "action_selection", "learner_update", "eval"}

    run_config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert run_config["observation"] == "pixels"
    assert run_config["checkpoints"][-1] == f"checkpoints/step_{config.total_steps}.pt"
    assert (out / "checkpoints" / f"step_{config.total_steps}.pt").exists()
    assert (out / "eval.csv").exists()
    assert (out / "dqn_frogger.pt").exists()
    assert metrics["total_steps"] == config.total_steps


def test_existing_run_is_refused_without_overwrite(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        pixel_trainer,
        "make_vector_env",
        lambda num_envs, num_threads=1, **_: FakeVectorEnv(num_envs, num_threads),
    )
    out = tmp_path / "run"
    pixel_trainer.train(tiny_config(out), out)
    with pytest.raises(FileExistsError):
        pixel_trainer.train(tiny_config(out), out)
    pixel_trainer.train(tiny_config(out), out, overwrite=True)

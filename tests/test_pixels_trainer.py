"""Pixel trainer smoke test on a tiny fake vector env (no ALE, CPU only)."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from dqn_frogger.pixels import trainer as pixel_trainer
from dqn_frogger.pixels.agent import PixelDQNAgent
from dqn_frogger.pixels.config import PixelDQNConfig
from dqn_frogger.pixels.safety import MemoryBudgetError
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


def _fake_envs(monkeypatch) -> None:
    monkeypatch.setattr(
        pixel_trainer,
        "make_vector_env",
        lambda num_envs, num_threads=1, **_: FakeVectorEnv(num_envs, num_threads),
    )


def _csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_periodic_saves_write_numbered_bundles_and_latest(tmp_path, monkeypatch) -> None:
    _fake_envs(monkeypatch)
    out = tmp_path / "run"
    config = replace(tiny_config(out), save_interval=100, checkpoint_steps=())

    pixel_trainer.train(config, out)

    checkpoints = out / "checkpoints"
    for step in (100, 200, 300, 400):
        assert (checkpoints / f"step_{step}.pt").exists()
    latest = torch.load(checkpoints / "latest.pt", weights_only=False)
    assert latest["env_step"] == 400
    assert {"online_net", "target_net", "optimizer", "episode", "epsilon", "rng", "config"} <= set(
        latest
    )
    assert not (checkpoints / "latest.pt.tmp").exists()


def test_resume_restores_weights_optimizer_rng_and_counters(tmp_path, monkeypatch) -> None:
    _fake_envs(monkeypatch)
    out = tmp_path / "run"
    first = replace(tiny_config(out), total_steps=200, save_interval=100, checkpoint_steps=())
    pixel_trainer.train(first, out)
    latest_path = out / "checkpoints" / "latest.pt"
    saved = torch.load(latest_path, weights_only=False)

    agent = PixelDQNAgent(n_actions=5, lr=first.lr, gamma=first.gamma, device="cpu")
    rng = np.random.default_rng(0)
    restored = pixel_trainer._restore_checkpoint(latest_path, agent, rng)

    assert restored.env_step == 200
    assert restored.episode == saved["episode"]
    for key, tensor in saved["online_net"].items():
        assert torch.equal(agent.online_net.state_dict()[key], tensor)
    for key, tensor in saved["target_net"].items():
        assert torch.equal(agent.target_net.state_dict()[key], tensor)
    restored_opt = agent.optimizer.state_dict()["state"]
    for index, moments in saved["optimizer"]["state"].items():
        assert torch.equal(restored_opt[index]["exp_avg"], moments["exp_avg"])
        assert torch.equal(restored_opt[index]["exp_avg_sq"], moments["exp_avg_sq"])
    assert rng.bit_generator.state == saved["rng"]["numpy"]
    assert torch.equal(torch.get_rng_state(), saved["rng"]["torch_cpu"])


def test_resume_appends_csv_rows_and_records_resumed_step(tmp_path, monkeypatch) -> None:
    _fake_envs(monkeypatch)
    out = tmp_path / "run"
    first = replace(tiny_config(out), total_steps=200, save_interval=100, checkpoint_steps=())
    pixel_trainer.train(first, out)
    episodes_before = _csv_rows(out / "episodes.csv")
    eval_before = _csv_rows(out / "eval.csv")
    last_episode = int(episodes_before[-1][0])

    pixel_trainer.train(replace(first, total_steps=400), out, resume=True)

    episodes_after = _csv_rows(out / "episodes.csv")
    assert episodes_after[: len(episodes_before)] == episodes_before
    assert episodes_after.count(EPISODE_HEADER) == 1
    assert int(episodes_after[len(episodes_before)][0]) == last_episode + 1
    assert len(_csv_rows(out / "eval.csv")) > len(eval_before)
    timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    run_config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert timing["resumed_from_step"] == 200
    assert run_config["resumed_from_step"] == 200


def test_resume_without_a_checkpoint_fails_clearly(tmp_path, monkeypatch) -> None:
    _fake_envs(monkeypatch)
    out = tmp_path / "run"
    with pytest.raises(FileNotFoundError, match="latest.pt"):
        pixel_trainer.train(tiny_config(out), out, resume=True)


class InterruptingEnv(FakeVectorEnv):
    """Raises KeyboardInterrupt on the call after `interrupt_after_calls` training steps."""

    def __init__(self, num_envs: int, num_threads: int, interrupt_after_calls: int) -> None:
        super().__init__(num_envs, num_threads)
        self.calls = 0
        self.interrupt_after_calls = interrupt_after_calls

    def step(self, actions):
        self.calls += 1
        if self.calls > self.interrupt_after_calls:
            raise KeyboardInterrupt
        return super().step(actions)


def test_keyboard_interrupt_saves_latest_before_exiting(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        pixel_trainer,
        "make_vector_env",
        lambda num_envs, num_threads=1, **_: InterruptingEnv(num_envs, num_threads, 30)
        if num_envs > 1
        else FakeVectorEnv(num_envs, num_threads),
    )
    out = tmp_path / "run"
    config = replace(tiny_config(out), save_interval=10**9, checkpoint_steps=())

    with pytest.raises(KeyboardInterrupt):
        pixel_trainer.train(config, out)

    latest = torch.load(out / "checkpoints" / "latest.pt", weights_only=False)
    assert latest["env_step"] == 30 * config.num_envs


def test_train_refuses_before_touching_disk_when_replay_does_not_fit(
    tmp_path, monkeypatch
) -> None:
    _fake_envs(monkeypatch)
    monkeypatch.setattr(pixel_trainer, "available_memory_bytes", lambda: 1_000_000_000)
    out = tmp_path / "run"

    with pytest.raises(MemoryBudgetError, match="--force-memory"):
        pixel_trainer.train(tiny_config(out), out)
    assert not out.exists()


def test_force_memory_starts_anyway(tmp_path, monkeypatch) -> None:
    _fake_envs(monkeypatch)
    monkeypatch.setattr(pixel_trainer, "available_memory_bytes", lambda: 1_000_000_000)
    out = tmp_path / "run"

    pixel_trainer.train(tiny_config(out), out, force_memory=True)

    assert (out / "episodes.csv").exists()

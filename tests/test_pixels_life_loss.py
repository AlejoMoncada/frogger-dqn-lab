"""Life loss as a terminal for bootstrapping only (episodic life for the TD target).

The fake env mirrors AtariVectorEnv: 4 lives, NextStep autoreset, lives reported in info. A game
lasts 4 * LIFE_LEN steps; a life is lost every LIFE_LEN steps and the last loss ends the game.
Every step pays reward 1, so a game's return is 4 * LIFE_LEN regardless of the actions.
"""

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from dqn_frogger.pixels import trainer as pixel_trainer
from dqn_frogger.pixels.config import PixelDQNConfig
from dqn_frogger.pixels.frame_replay import FrameReplay
from dqn_frogger.pixels.replay import PixelReplayBuffer
from dqn_frogger.trainer import EPISODE_HEADER

LIFE_LEN = 3
GAME_LEN = 4 * LIFE_LEN


class FakeLifeEnv:
    def __init__(self, num_envs: int, num_threads: int = 1) -> None:
        self.num_envs = num_envs
        self.single_action_space = gym.spaces.Discrete(5)
        self._steps = np.zeros(num_envs, dtype=np.int64)
        self._lives = np.full(num_envs, 4, dtype=np.int64)
        self._pending_reset = np.zeros(num_envs, dtype=bool)

    def _obs(self) -> np.ndarray:
        return np.zeros((self.num_envs, 4, 84, 84), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        self._steps[:] = 0
        self._lives[:] = 4
        self._pending_reset[:] = False
        return self._obs(), {"lives": self._lives.copy()}

    def step(self, actions):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            if self._pending_reset[i]:  # NextStep reset step: reward 0, lives back to 4
                self._pending_reset[i] = False
                self._steps[i] = 0
                self._lives[i] = 4
                continue
            self._steps[i] += 1
            rewards[i] = 1.0
            if self._steps[i] % LIFE_LEN == 0:
                self._lives[i] -= 1
            if self._steps[i] == GAME_LEN:
                terminated[i] = True
                self._pending_reset[i] = True
        info = {"lives": self._lives.copy()}
        return self._obs(), rewards, terminated, np.zeros(self.num_envs, dtype=bool), info

    def close(self) -> None:
        pass


def _config(output_dir: Path, **overrides) -> PixelDQNConfig:
    base = replace(
        PixelDQNConfig.smoke(device="cpu"),
        num_envs=1,
        num_threads=1,
        total_steps=2 * GAME_LEN + 1,  # two games plus the NextStep reset step between them
        buffer_size=200,
        batch_size=4,
        n_step=3,
        gamma=0.5,
        reward_mode="raw",
        learning_starts=10_000,  # no learner updates: the recorded adds are what matters
        eval_interval=10_000,
        eval_episodes=1,
        checkpoint_steps=(),
    )
    return replace(base, **overrides)


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setattr(
        pixel_trainer,
        "make_vector_env",
        lambda num_envs, num_threads=1, **_: FakeLifeEnv(num_envs, num_threads),
    )


@pytest.fixture
def recorded_adds(monkeypatch):
    """Record (reward, done, discount) of every transition the trainer stores, both layouts."""
    adds: list[tuple[float, bool, float]] = []
    for cls in (PixelReplayBuffer, FrameReplay):
        original = cls.add

        def recording_add(self, obs, action, reward, next_obs, done, discount, _original=original):
            adds.append((round(float(reward), 6), bool(done), round(float(discount), 6)))
            return _original(self, obs, action, reward, next_obs, done, discount)

        monkeypatch.setattr(cls, "add", recording_add)
    return adds


def _episode_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


# Windows of n=3, gamma=0.5 inside one LIFE_LEN segment: full, then cut at the death.
SEGMENT = [(1.75, True, 0.125), (1.5, True, 0.25), (1.0, True, 0.5)]


def test_flag_on_stores_each_life_loss_as_done_and_flushes_windows(
    tmp_path, fake_env, recorded_adds
) -> None:
    pixel_trainer.train(_config(tmp_path / "on", life_loss_terminal=True), tmp_path / "on")

    # Each game has 4 segments; every segment ends with a terminal (life loss or game over).
    assert recorded_adds == SEGMENT * 4 * 2


def test_flag_off_keeps_only_game_over_as_terminal(tmp_path, fake_env, recorded_adds) -> None:
    pixel_trainer.train(_config(tmp_path / "off", life_loss_terminal=False), tmp_path / "off")

    full_open = [(1.75, False, 0.125)] * (GAME_LEN - 3)  # windows closed by the window size
    per_game = full_open + [(1.75, True, 0.125), (1.5, True, 0.25), (1.0, True, 0.5)]
    assert recorded_adds == per_game * 2


def test_episode_log_keeps_one_row_per_full_game(tmp_path, fake_env, recorded_adds) -> None:
    out = tmp_path / "on"
    pixel_trainer.train(_config(out, life_loss_terminal=True), out)

    rows = _episode_rows(out / "episodes.csv")
    assert rows[0] == EPISODE_HEADER
    games = rows[1:]
    assert len(games) == 2
    assert [float(row[EPISODE_HEADER.index("return")]) for row in games] == [GAME_LEN, GAME_LEN]
    assert [int(row[EPISODE_HEADER.index("length")]) for row in games] == [GAME_LEN, GAME_LEN]


def test_episode_log_is_identical_with_and_without_the_flag(tmp_path, fake_env) -> None:
    off, on = tmp_path / "off", tmp_path / "on"
    pixel_trainer.train(_config(off, life_loss_terminal=False), off)
    pixel_trainer.train(_config(on, life_loss_terminal=True), on)

    assert _episode_rows(on / "episodes.csv") == _episode_rows(off / "episodes.csv")


def test_lives_are_tracked_across_the_autoreset_into_the_next_game(
    tmp_path, fake_env, recorded_adds
) -> None:
    # If lives_prev kept the game-over value 0 across the reset step, the first loss of game 2
    # (4 -> 3) would be missed and the second game would not match the first.
    pixel_trainer.train(_config(tmp_path / "on", life_loss_terminal=True), tmp_path / "on")

    game_one, game_two = recorded_adds[: 4 * 3], recorded_adds[4 * 3 :]
    assert game_one == game_two == SEGMENT * 4


@pytest.mark.parametrize("replay_impl", ["frame", "stacked"])
def test_both_replay_layouts_train_with_life_loss_terminal(tmp_path, fake_env, replay_impl) -> None:
    out = tmp_path / replay_impl
    pixel_trainer.train(
        _config(out, life_loss_terminal=True, replay_impl=replay_impl), out
    )

    assert len(_episode_rows(out / "episodes.csv")) == 3  # header + two games


def test_config_round_trip_and_missing_key_default(tmp_path) -> None:
    config = _config(tmp_path, life_loss_terminal=True)
    saved = json.loads(json.dumps(asdict(config)))
    assert PixelDQNConfig(**saved).life_loss_terminal is True

    old = {key: value for key, value in saved.items() if key != "life_loss_terminal"}
    assert PixelDQNConfig(**old).life_loss_terminal is False
    assert PixelDQNConfig().life_loss_terminal is False


def test_resume_keeps_the_saved_flag_unless_it_is_passed(tmp_path, fake_env) -> None:
    out = tmp_path / "run"
    pixel_trainer.train(_config(out, life_loss_terminal=True), out)

    resumed = replace(_config(out, life_loss_terminal=False), total_steps=2 * GAME_LEN + 1 + 6)
    pixel_trainer.train(resumed, out, resume=True)

    saved = json.loads((out / "config.json").read_text(encoding="utf-8"))["config"]
    assert saved["life_loss_terminal"] is True


def test_cli_flag_sets_life_loss_terminal(monkeypatch) -> None:
    import sys

    import train_pixels

    captured = {}
    monkeypatch.setattr(train_pixels, "train", lambda config, output_dir, **_: captured.update(c=config) or {})
    monkeypatch.setattr(sys, "argv", ["train_pixels.py", "--device", "cpu", "--life-loss-terminal"])
    train_pixels.main()
    assert captured["c"].life_loss_terminal is True

    monkeypatch.setattr(sys, "argv", ["train_pixels.py", "--device", "cpu"])
    train_pixels.main()
    assert captured["c"].life_loss_terminal is False

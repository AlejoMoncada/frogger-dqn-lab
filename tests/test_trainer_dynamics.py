"""Trainer dynamics on a scripted environment: done flags, next observations, target sync,
replay reward clipping and raw logged returns."""

import csv
from pathlib import Path

import gymnasium as gym
import numpy as np

import dqn_frogger.trainer as trainer_module
from dqn_frogger.config import DQNConfig
from dqn_frogger.replay_buffer import ReplayBuffer

# (reward, terminated, truncated, lives) for each global step. Step 2 truncates, step 5 terminates,
# step 11 truncates. Once the script ends, every step truncates so evaluation episodes finish.
SCRIPT = [
    (5.0, False, False, 3),
    (-3.0, False, False, 3),
    (0.0, False, True, 3),
    (1.0, False, False, 3),
    (0.0, False, False, 2),
    (0.0, True, False, 2),
    (0.0, False, False, 2),
    (0.0, False, False, 2),
    (0.0, False, False, 2),
    (0.0, False, False, 2),
    (0.0, False, False, 2),
    (0.0, False, True, 2),
]
RESET_OBS = 7.0
TERMINAL_OBS = 200.0
MIDSTEP_OBS = 50.0


class ScriptedEnv(gym.Env):
    observation_space = gym.spaces.Box(low=0, high=255, shape=(128,), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self) -> None:
        self._global_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.full(128, RESET_OBS, dtype=np.uint8), {}

    def step(self, action):
        index = min(self._global_step, len(SCRIPT) - 1)
        reward, terminated, truncated, lives = SCRIPT[index]
        if self._global_step >= len(SCRIPT):
            truncated = True
        self._global_step += 1
        if terminated or truncated:
            obs_value = TERMINAL_OBS
        else:
            obs_value = MIDSTEP_OBS
        obs = np.full(128, obs_value, dtype=np.uint8)
        return obs, reward, terminated, truncated, {"lives": lives}


def _config(tmp_path: Path) -> DQNConfig:
    return DQNConfig(
        total_steps=len(SCRIPT),
        learning_starts=4,
        buffer_size=100,
        batch_size=2,
        train_freq=1,
        target_update_interval=4,
        eval_interval=1_000,
        eval_episodes=1,
        checkpoint_steps=(),
    )


def test_trainer_dynamics_on_scripted_env(tmp_path: Path, monkeypatch) -> None:
    transitions: list[tuple] = []
    sync_steps: list[int] = []
    update_steps: list[int] = []

    class SpyBuffer(ReplayBuffer):
        def add(self, obs, action, reward, next_obs, done):
            transitions.append(
                (float(reward), bool(done), np.array(next_obs, copy=True))
            )
            super().add(obs, action, reward, next_obs, done)

    class SpyAgent(trainer_module.DQNAgent):
        def sync_target(self) -> None:
            sync_steps.append(len(transitions))
            super().sync_target()

        def update(self, batch) -> float:
            update_steps.append(len(transitions))
            return super().update(batch)

    monkeypatch.setattr(trainer_module, "make_env", lambda seed, render_mode=None: ScriptedEnv())
    monkeypatch.setattr(trainer_module, "ReplayBuffer", SpyBuffer)
    monkeypatch.setattr(trainer_module, "DQNAgent", SpyAgent)

    trainer_module.train(_config(tmp_path), tmp_path)

    total = len(SCRIPT)
    assert len(transitions) == total

    # Truncation at step 2 is not a terminal state; termination at step 5 is.
    assert transitions[2][1] is False
    assert transitions[5][1] is True
    assert transitions[11][1] is False

    # On termination the stored next observation is the pre-reset one, not the reset observation.
    assert np.all(transitions[5][2] == TERMINAL_OBS)
    assert np.all(transitions[2][2] == TERMINAL_OBS)

    # Target sync: at construction (step 0), then exactly at each target_update_interval boundary.
    assert sync_steps == [0, 4, 8, 12]

    # Updates start at learning_starts and happen every train_freq steps.
    assert update_steps and min(update_steps) >= 4

    # Replay rewards are clipped to their sign; the logged episode returns stay raw.
    assert transitions[0][0] == 1.0
    assert transitions[1][0] == -1.0
    with (tmp_path / "episodes.csv").open(encoding="utf-8") as f:
        returns = [float(row["return"]) for row in csv.DictReader(f)]
    assert returns == [2.0, 1.0, 0.0]

"""Evaluation policy RNG must depend only on the episode seed, never on earlier evaluations."""

import gymnasium as gym
import numpy as np

from dqn_frogger.evaluation import run_episodes


class _ScriptedEnv(gym.Env):
    """Three steps per episode, obs is the step counter; actions are never used by the env."""

    observation_space = gym.spaces.Box(low=0.0, high=3.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        return np.array([self._t], dtype=np.float32), 1.0, self._t >= 3, False, {}


def _recording_policy(log: list[int]):
    def act(obs, rng: np.random.Generator) -> int:
        action = int(rng.integers(2))
        log.append(action)
        return action

    return act


def test_episode_actions_do_not_depend_on_previous_evaluations() -> None:
    env = _ScriptedEnv()
    first_log: list[int] = []
    run_episodes(env, _recording_policy(first_log), episodes=2, seed=100)

    # An unrelated evaluation in between must not shift the random stream of a later one.
    run_episodes(env, _recording_policy([]), episodes=5, seed=7)

    second_log: list[int] = []
    run_episodes(env, _recording_policy(second_log), episodes=2, seed=100)

    assert first_log == second_log
    assert len(first_log) == 6


def test_each_episode_gets_its_own_seeded_stream() -> None:
    env = _ScriptedEnv()
    log: list[int] = []
    run_episodes(env, _recording_policy(log), episodes=2, seed=100)

    expected = list(np.random.default_rng(100).integers(2, size=3))
    expected += list(np.random.default_rng(101).integers(2, size=3))
    assert log == [int(a) for a in expected]

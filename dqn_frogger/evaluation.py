"""Greedy and random policy evaluation helpers."""

from collections.abc import Callable

import gymnasium as gym
import numpy as np

Act = Callable[[np.ndarray, np.random.Generator], int]


def run_episodes(
    env: gym.Env,
    act: Act,
    episodes: int,
    seed: int,
) -> list[float]:
    """Run `episodes` full episodes and return the raw (unclipped) score of each.

    Episode i resets the environment with seed + i and gives the policy its own generator
    default_rng(seed + i), so results do not depend on earlier evaluations.
    """
    returns: list[float] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        policy_rng = np.random.default_rng(seed + episode)
        done = False
        total = 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(act(obs, policy_rng))
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    return returns

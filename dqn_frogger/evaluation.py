"""Greedy and random policy evaluation helpers."""

from collections.abc import Callable

import gymnasium as gym
import numpy as np


def run_episodes(
    env: gym.Env,
    act: Callable[[np.ndarray], int],
    episodes: int,
    seed: int,
) -> list[float]:
    """Run `episodes` full episodes and return the raw (unclipped) score of each."""
    returns: list[float] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total = 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(act(obs))
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    return returns

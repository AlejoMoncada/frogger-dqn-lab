"""Gymnasium environment factory for ALE/Frogger-v5 with RAM observations."""

import ale_py
import gymnasium as gym
import numpy as np

ENV_ID = "ALE/Frogger-v5"
RAM_SIZE = 128


class RamFloatObservation(gym.ObservationWrapper):
    """Convert the uint8 RAM vector (128,) into float32 values in [0, 1]."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(RAM_SIZE,), dtype=np.float32
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return (observation.astype(np.float32) / 255.0).astype(np.float32)


def make_env(seed: int, render_mode: str | None = None) -> gym.Env:
    """Create Frogger with RAM observations, v5 defaults, and a seeded action space."""
    gym.register_envs(ale_py)
    env = gym.make(ENV_ID, obs_type="ram", render_mode=render_mode)
    env = RamFloatObservation(env)
    env.action_space.seed(seed)
    return env

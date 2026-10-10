"""Leaderboard media path on a tiny fake episode: first-medal PNGs and a readable GIF."""

from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from milestones import play_episode, write_gif, write_screenshots

HEIGHT, WIDTH = 64, 80


class FakeRenderEnv(gym.Env):
    """Six forward hops (rows 1-6, reward +1 each) on a frame that changes every step."""

    observation_space = gym.spaces.Box(low=0, high=255, shape=(128,), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros(128, dtype=np.uint8), {}

    def step(self, action):
        self._t += 1
        reward = 1.0 if self._t <= 6 else 0.0
        done = self._t >= 20
        return np.zeros(128, dtype=np.uint8), reward, False, done, {"lives": 4}

    def render(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :] = (self._t * 10) % 256
        return frame


def test_medal_screenshots_and_gif_are_readable(tmp_path: Path) -> None:
    episode = play_episode(FakeRenderEnv(), lambda _obs, _rng: 0, seed=900_000, keep_gif=True)

    assert {"first_hop", "road_crossed"} <= set(episode.medal_frames)
    (tmp_path / "screenshots").mkdir()  # main() creates this directory before evaluating
    saved = write_screenshots(tmp_path, 100, [episode])
    assert saved["first_hop"] == "screenshots/step_100_first_hop.png"
    for relative in saved.values():
        image = imageio.imread(tmp_path / relative)
        assert image.shape[:2] == (HEIGHT, WIDTH)

    assert len(episode.gif_frames) > 1
    gif_path = tmp_path / "gif.gif"
    write_gif(gif_path, episode.gif_frames)
    frames = imageio.mimread(gif_path)
    assert len(frames) > 1
    assert frames[0].shape[:2] == episode.gif_frames[0].shape[:2]

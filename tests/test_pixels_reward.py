"""Reward transform for the pixel trainer: sign, scale and raw modes, legacy reward_clip mapping."""

import csv
from dataclasses import asdict, replace

import numpy as np
import pytest

from dqn_frogger.pixels import trainer as pixel_trainer
from dqn_frogger.pixels.config import PixelDQNConfig, transform_reward, upgrade_saved_config
from dqn_frogger.pixels.replay import PixelReplayBuffer
from test_pixels_trainer import FakeVectorEnv, tiny_config


class HomeRewardEnv(FakeVectorEnv):
    """FakeVectorEnv autoreset, but the +1 steps become a home-sized +55 reward."""

    def step(self, actions):
        obs, rewards, terminated, truncated, info = super().step(actions)
        return obs, rewards * 55.0, terminated, truncated, info


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("sign", {0: 0.0, 1: 1.0, 55: 1.0}),
        ("scale", {0: 0.0, 1: 0.1, 55: 5.5}),
        ("raw", {0: 0.0, 1: 1.0, 55: 55.0}),
    ],
)
def test_transform_reward_values_per_mode(mode, expected) -> None:
    for raw, value in expected.items():
        assert transform_reward(float(raw), mode, 0.1) == pytest.approx(value)


def test_transform_reward_sign_keeps_negative_sign() -> None:
    assert transform_reward(-3.0, "sign", 0.1) == -1.0


def test_transform_reward_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="reward_mode"):
        transform_reward(1.0, "clip", 0.1)
    with pytest.raises(ValueError, match="reward_mode"):
        PixelDQNConfig(reward_mode="clip")


def test_legacy_reward_clip_maps_to_reward_mode() -> None:
    saved_true = upgrade_saved_config({"reward_clip": True, "seed": 1})
    saved_false = upgrade_saved_config({"reward_clip": False, "seed": 1})

    assert saved_true["reward_mode"] == "sign"
    assert saved_false["reward_mode"] == "raw"
    assert "reward_clip" not in saved_true
    assert saved_true["seed"] == 1


def test_resumed_legacy_config_is_sign_and_reports_no_reward_change() -> None:
    current = PixelDQNConfig(device="cpu")
    saved = asdict(current)
    saved.pop("reward_mode")
    saved.pop("reward_scale")
    saved["reward_clip"] = True

    assert upgrade_saved_config(saved)["reward_mode"] == "sign"
    changed = pixel_trainer._changed_hyperparameters(saved, current)
    assert "reward_mode" not in changed


def test_nstep_return_uses_transformed_rewards_not_raw(tmp_path, monkeypatch) -> None:
    stored: list[PixelReplayBuffer] = []

    class RecordingBuffer(PixelReplayBuffer):
        def __init__(self, capacity: int) -> None:
            super().__init__(capacity)
            stored.append(self)

    monkeypatch.setattr(pixel_trainer, "PixelReplayBuffer", RecordingBuffer)
    monkeypatch.setattr(
        pixel_trainer,
        "make_vector_env",
        lambda num_envs, num_threads=1, **_: HomeRewardEnv(num_envs, num_threads),
    )
    out = tmp_path / "run"
    config = replace(
        tiny_config(out),
        total_steps=20,
        learning_starts=10**6,
        eval_interval=10**6,
        checkpoint_steps=(),
        reward_mode="scale",
        reward_scale=0.1,
        n_step=3,
    )

    pixel_trainer.train(config, out)

    rewards = stored[0].rewards[: len(stored[0])]
    # Episode rewards are 0, 55, 0, 55, 0. The first 3-step window is 0 + 0.99 * 55 + 0 once
    # scaled by 0.1 (5.445); raw rewards would give 54.45 instead.
    assert np.any(np.isclose(rewards, 0.99 * 5.5, atol=1e-4))
    assert rewards.max() < 20.0  # raw 3-step sums would reach ~109

    with (out / "episodes.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert float(rows[1][2]) == pytest.approx(110.0)  # logged episode return stays raw

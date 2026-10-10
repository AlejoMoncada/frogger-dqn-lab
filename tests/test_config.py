"""Tests for the checkpoint schedule derived from DQNConfig."""

from dqn_frogger.config import DQNConfig


def test_checkpoint_steps_beyond_total_are_dropped_and_final_step_added() -> None:
    config = DQNConfig(total_steps=400, checkpoint_steps=(100, 300, 500))
    assert config.active_checkpoint_steps() == (100, 300, 400)


def test_default_checkpoint_steps_are_kept_for_long_runs() -> None:
    config = DQNConfig(total_steps=3_000_000)
    assert config.active_checkpoint_steps() == (
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        3_000_000,
    )


def test_smoke_config_uses_small_checkpoint_steps() -> None:
    assert DQNConfig.smoke().active_checkpoint_steps() == (1_000, 2_000)

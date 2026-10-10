"""Seed separation between training and the milestone leaderboard evaluation."""

import pytest

from dqn_frogger.seeds import EVAL_SEED_BASE, EVAL_SEED_SPAN, validate_training_seed


def test_training_seed_inside_leaderboard_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="leaderboard"):
        validate_training_seed(EVAL_SEED_BASE)
    with pytest.raises(ValueError, match="leaderboard"):
        validate_training_seed(EVAL_SEED_BASE + EVAL_SEED_SPAN - 1)


def test_training_seed_outside_leaderboard_range_is_accepted() -> None:
    validate_training_seed(42)
    validate_training_seed(EVAL_SEED_BASE - 1)
    validate_training_seed(EVAL_SEED_BASE + EVAL_SEED_SPAN)

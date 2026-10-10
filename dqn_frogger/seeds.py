"""Seed ranges shared by training and the milestone leaderboard.

The leaderboard evaluates checkpoints on seeds EVAL_SEED_BASE + i. Training must never use a
seed in that range, otherwise evaluation and training share episodes.
"""

EVAL_SEED_BASE = 900_000
EVAL_SEED_SPAN = 1_000  # leaderboard seeds are [EVAL_SEED_BASE, EVAL_SEED_BASE + EVAL_SEED_SPAN)


def validate_training_seed(seed: int) -> None:
    """Raise ValueError when the training seed falls inside the leaderboard evaluation range."""
    if EVAL_SEED_BASE <= seed < EVAL_SEED_BASE + EVAL_SEED_SPAN:
        raise ValueError(
            f"training seed {seed} is inside the leaderboard evaluation seed range "
            f"[{EVAL_SEED_BASE}, {EVAL_SEED_BASE + EVAL_SEED_SPAN}); choose another seed"
        )

"""Smoke test: a tiny end-to-end training run writes metrics and a checkpoint."""

import csv
from pathlib import Path

from dqn_frogger.config import DQNConfig
from dqn_frogger.trainer import CHECKPOINT_NAME, EPISODES_CSV, EVAL_CSV, train


def test_train_smoke_writes_metrics_and_checkpoint(tmp_path: Path) -> None:
    config = DQNConfig(
        total_steps=400,
        learning_starts=50,
        buffer_size=500,
        batch_size=16,
        eval_interval=200,
        eval_episodes=1,
        target_update_interval=100,
    )

    metrics = train(config, tmp_path)

    assert (tmp_path / EPISODES_CSV).exists()
    assert (tmp_path / EVAL_CSV).exists()
    assert (tmp_path / CHECKPOINT_NAME).exists()
    assert metrics["total_steps"] == 400

    with (tmp_path / EVAL_CSV).open(encoding="utf-8") as f:
        eval_rows = list(csv.DictReader(f))
    assert [int(r["global_step"]) for r in eval_rows] == [200, 400]

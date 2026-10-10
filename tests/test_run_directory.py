"""Run-directory reuse protection and checkpoint discovery for the leaderboard."""

import json
from pathlib import Path

import pytest

from dqn_frogger.config import DQNConfig
from dqn_frogger.trainer import CHECKPOINT_NAME, EPISODES_CSV, train
from milestones import find_run_checkpoints


def _tiny_config(**overrides: object) -> DQNConfig:
    values = dict(
        total_steps=200,
        learning_starts=50,
        buffer_size=500,
        batch_size=16,
        eval_interval=200,
        eval_episodes=1,
        target_update_interval=100,
        checkpoint_steps=(100,),
    )
    values.update(overrides)
    return DQNConfig(**values)  # type: ignore[arg-type]


def test_train_refuses_a_directory_that_already_holds_a_run(tmp_path: Path) -> None:
    train(_tiny_config(), tmp_path)

    with pytest.raises(FileExistsError, match="--overwrite"):
        train(_tiny_config(), tmp_path)


def test_overwrite_clears_only_known_run_artifacts(tmp_path: Path) -> None:
    train(_tiny_config(), tmp_path)
    (tmp_path / "notes.txt").write_text("keep me", encoding="utf-8")

    train(_tiny_config(), tmp_path, overwrite=True)

    assert (tmp_path / CHECKPOINT_NAME).exists()
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_empty_checkpoint_directory_does_not_block_a_new_run(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir()
    train(_tiny_config(), tmp_path)
    assert (tmp_path / EPISODES_CSV).exists()


def test_leaderboard_discovers_checkpoints_from_config_manifest(tmp_path: Path) -> None:
    train(_tiny_config(), tmp_path)
    stale = tmp_path / "checkpoints" / "step_50.pt"
    stale.write_bytes(b"left over from another run")

    found = find_run_checkpoints(tmp_path)

    assert [step for step, _ in found] == [100, 200]


def test_leaderboard_falls_back_to_glob_without_config(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "step_300.pt").write_bytes(b"x")
    (tmp_path / "checkpoints" / "step_100.pt").write_bytes(b"x")

    assert [step for step, _ in find_run_checkpoints(tmp_path)] == [100, 300]

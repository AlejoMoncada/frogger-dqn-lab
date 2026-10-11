"""Checkpoint discovery for the leaderboard: killed runs and latest.pt (no ALE, no evaluation)."""

import json

import pytest
import torch

import milestones


def _checkpoint(path, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"online_net": {}, "env_step": step}, path)


def test_missing_manifest_checkpoints_are_skipped_with_a_warning(tmp_path) -> None:
    run = tmp_path / "run"
    _checkpoint(run / "checkpoints" / "step_100.pt", 100)
    (run / "config.json").write_text(
        json.dumps({"checkpoints": ["checkpoints/step_100.pt", "checkpoints/step_200.pt"]}),
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="step_200.pt"):
        found = milestones.find_run_checkpoints(run)

    assert [step for step, _ in found] == [100]


def test_latest_checkpoint_is_not_a_leaderboard_row(tmp_path) -> None:
    run = tmp_path / "run"
    _checkpoint(run / "checkpoints" / "step_100.pt", 100)
    _checkpoint(run / "checkpoints" / "latest.pt", 150)

    found = milestones.find_run_checkpoints(run)

    assert [step for step, _ in found] == [100]


def test_latest_checkpoint_is_ignored_with_a_manifest_too(tmp_path) -> None:
    run = tmp_path / "run"
    _checkpoint(run / "checkpoints" / "step_100.pt", 100)
    _checkpoint(run / "checkpoints" / "latest.pt", 150)
    (run / "config.json").write_text(
        json.dumps({"checkpoints": ["checkpoints/step_100.pt"]}), encoding="utf-8"
    )

    assert [step for step, _ in milestones.find_run_checkpoints(run)] == [100]

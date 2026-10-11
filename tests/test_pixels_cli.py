"""CLI flags of train_pixels.py land in PixelDQNConfig and in the run's config.json."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

import train_pixels
from dqn_frogger.pixels import trainer as pixel_trainer
from dqn_frogger.pixels.config import PixelDQNConfig

TUNING_ARGS = [
    "--batch-size",
    "64",
    "--num-threads",
    "2",
    "--target-update-interval",
    "500",
    "--learning-starts",
    "300",
    "--eps-decay-steps",
    "5000",
    "--lr",
    "0.0005",
]


def _run_cli(monkeypatch, argv: list[str]) -> PixelDQNConfig:
    """Run main() with `train` replaced by a recorder and return the config it received."""
    captured: dict[str, PixelDQNConfig] = {}

    def fake_train(config, output_dir, **kwargs):
        captured["config"] = config
        return {}

    monkeypatch.setattr(train_pixels, "train", fake_train)
    monkeypatch.setattr(sys, "argv", ["train_pixels.py", "--device", "cpu", *argv])
    train_pixels.main()
    return captured["config"]


def test_tuning_flags_land_in_config(monkeypatch) -> None:
    config = _run_cli(monkeypatch, TUNING_ARGS)

    assert config.batch_size == 64
    assert config.num_threads == 2
    assert config.target_update_interval == 500
    assert config.learning_starts == 300
    assert config.eps_decay_steps == 5000
    assert config.lr == pytest.approx(0.0005)


def test_defaults_are_unchanged_without_flags(monkeypatch) -> None:
    config = _run_cli(monkeypatch, [])

    assert config == PixelDQNConfig(device="cpu")


def test_tuning_flags_are_written_to_config_json(monkeypatch, tmp_path: Path) -> None:
    config = _run_cli(monkeypatch, TUNING_ARGS)
    path = tmp_path / "config.json"

    pixel_trainer._write_run_config(path, config)
    saved = json.loads(path.read_text(encoding="utf-8"))["config"]

    assert saved["batch_size"] == 64
    assert saved["num_threads"] == 2
    assert saved["target_update_interval"] == 500
    assert saved["learning_starts"] == 300
    assert saved["eps_decay_steps"] == 5000
    assert saved["lr"] == pytest.approx(0.0005)
    assert saved == json.loads(json.dumps(asdict(config)))  # tuples become lists in JSON

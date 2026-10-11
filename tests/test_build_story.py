"""Tests for tools/build_story.py: data structure, copied assets, idempotence (tmp dirs only)."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILDER_PATH = REPO_ROOT / "tools" / "build_story.py"


@pytest.fixture
def builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_story", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _leaderboard(run_id: str) -> dict[str, Any]:
    medals_random = {"first_hop": 1.0, "road_crossed": 0.0, "first_log": 0.0, "mid_river": 0.0,
                     "frog_home": 0.0, "three_homes": 0.0, "level_cleared": 0.0}
    medals_ckpt = dict(medals_random, first_hop=1.0, road_crossed=0.9, frog_home=0.5)
    return {
        "schema_version": 1,
        "algorithm": "DQN-pixels (Double, n-step)",
        "env_id": "ALE/Frogger-v5",
        "run_id": run_id,
        "proxy_medals": ["frog_home", "three_homes", "level_cleared"],
        "rows": [
            {
                "label": "random", "step": None, "episodes": 20, "score_mean": 9.3,
                "score_std": 1.7, "max_rows_mean": 3.0, "max_rows_max": 5,
                "homes_mean": 0.0, "homes_max": 0, "medals": medals_random,
                "screenshots": {}, "gif": "gifs/random.gif",
            },
            {
                "label": "100,000", "step": 100000, "episodes": 20, "score_mean": 40.0,
                "score_std": 3.0, "max_rows_mean": 9.0, "max_rows_max": 10,
                "homes_mean": 0.5, "homes_max": 1, "medals": medals_ckpt,
                "screenshots": {"first_hop": "screenshots/step_100000_first_hop.png",
                                "frog_home": "screenshots/step_100000_frog_home.png"},
                "gif": "gifs/step_100000.gif",
            },
        ],
    }


def _make_fake_results(root: Path) -> Path:
    """A results folder with one complete run (config + timing) and one killed run."""
    results = root / "results"
    full = results / "run_fake_full"
    _write(full / "leaderboard.json", json.dumps(_leaderboard("run_fake_full")))
    _write(full / "config.json", json.dumps({
        "observation": "pixels",
        "algorithm": "DQN-pixels (Double, n-step)",
        "config": {
            "total_steps": 5000000, "buffer_size": 200000, "replay_impl": "frame",
            "reward_mode": "scale", "reward_scale": 0.1, "seed": 43, "device": "mps",
            "life_loss_terminal": True,
        },
    }))
    _write(full / "timing.json", json.dumps({
        "total_steps": 5000000, "steps_per_second": 379.5, "device": "mps",
        "resumed_from_step": None,
    }))
    _write(full / "gifs" / "random.gif", b"GIF89a-random")
    _write(full / "gifs" / "step_100000.gif", b"GIF89a-100k")
    _write(full / "screenshots" / "step_100000_first_hop.png", b"png-first-hop")
    _write(full / "screenshots" / "step_100000_frog_home.png", b"png-frog-home")

    killed = results / "run_fake_killed"
    board = _leaderboard("run_fake_killed")
    board["rows"][1]["gif"] = "gifs/missing.gif"
    _write(killed / "leaderboard.json", json.dumps(board))
    _write(killed / "config.json", json.dumps({
        "observation": "pixels", "config": {"total_steps": 5000000, "reward_clip": True},
    }))
    _write(killed / "gifs" / "random.gif", b"GIF89a-killed-random")
    return results


def test_build_writes_structured_data_and_copies_assets(builder: ModuleType, tmp_path: Path) -> None:
    results = _make_fake_results(tmp_path)
    out = tmp_path / "historia"

    data = builder.build(results, out)
    builder.write_data_js(data, out)

    runs = {run["id"]: run for run in data["runs"]}
    assert set(runs) == {"run_fake_full", "run_fake_killed"}
    full = runs["run_fake_full"]
    assert full["status"] == "complete"
    assert "proxy_medals" not in full
    settings = full["settings"]
    assert settings["buffer_size"] == 200000
    assert settings["replay"] == "frame"
    assert settings["reward_mode"] == "scale"
    assert settings["reward_scale"] == 0.1
    assert settings["life_loss_terminal"] is True
    assert settings["seed"] == 43
    assert settings["device"] == "mps"
    assert settings["steps_per_second"] == 379.5
    assert settings["source"] == "config.json"

    random_row, ckpt = full["rows"]
    assert random_row["title"] == "Aleatoria"
    assert ckpt["title"] == "100 000"
    assert ckpt["medals"]["frog_home"] == 0.5
    assert set(ckpt["medals"]) == set(builder.MEDAL_ORDER)
    assert ckpt["gif"] == "assets/run_fake_full/gifs/step_100000.gif"
    assert ckpt["screenshots"]["first_hop"] == "assets/run_fake_full/screenshots/step_100000_first_hop.png"

    for run in data["runs"]:
        for row in run["rows"]:
            for path in [row["gif"], *row["screenshots"].values()]:
                if path is not None:
                    assert (out / path).is_file(), path

    killed = runs["run_fake_killed"]
    assert killed["settings"]["steps_per_second"] is None
    assert killed["settings"]["reward_mode"] == "sign"
    assert killed["rows"][1]["gif"] is None
    assert any(note.startswith("Falta") for note in killed["notes"])

    assert data["hero"]["run_id"] == "run_fake_full"
    assert data["hero"]["checkpoint"]["title"] == "100 000"
    assert data["hero"]["random"]["title"] == "Aleatoria"
    assert data["best_by_run"]["run_fake_full"]["frog_home"] == 0.5


def test_data_js_is_a_script_with_parseable_json(builder: ModuleType, tmp_path: Path) -> None:
    results = _make_fake_results(tmp_path)
    out = tmp_path / "historia"
    builder.write_data_js(builder.build(results, out), out)

    text = (out / "data.js").read_text(encoding="utf-8")
    prefix = "window.STORY_DATA = "
    assert text.startswith(prefix)
    assert text.rstrip().endswith(";")
    payload = json.loads(text[len(prefix):].rstrip().removesuffix(";"))
    assert [item["key"] for item in payload["medal_ladder"]] == list(builder.MEDAL_ORDER)


def test_home_detection_is_verified_and_no_home_proxy_is_emitted(builder: ModuleType, tmp_path: Path) -> None:
    results = _make_fake_results(tmp_path)
    out = tmp_path / "historia"
    builder.write_data_js(builder.build(results, out), out)

    text = (out / "data.js").read_text(encoding="utf-8")
    payload = json.loads(text[len("window.STORY_DATA = "):].rstrip().removesuffix(";"))
    detection = payload["home_detection"]
    assert detection["status"] == "verified"
    assert "+55" in detection["evidence"]
    assert "proxy" not in text
    for run in payload["runs"]:
        assert "proxy_medals" not in run


def test_rebuild_is_idempotent_and_removes_stale_assets(builder: ModuleType, tmp_path: Path) -> None:
    results = _make_fake_results(tmp_path)
    out = tmp_path / "historia"
    builder.main(["--results", str(results), "--out", str(out)])
    first = (out / "data.js").read_bytes()

    stale = out / "assets" / "run_old" / "gifs" / "stale.gif"
    _write(stale, b"old")
    builder.main(["--results", str(results), "--out", str(out)])

    assert (out / "data.js").read_bytes() == first
    assert not stale.exists()
    assert (out / "assets" / "run_fake_full" / "gifs" / "step_100000.gif").read_bytes() == b"GIF89a-100k"


def test_main_rejects_missing_results_folder(builder: ModuleType, tmp_path: Path) -> None:
    assert builder.main(["--results", str(tmp_path / "nope"), "--out", str(tmp_path / "out")]) == 1

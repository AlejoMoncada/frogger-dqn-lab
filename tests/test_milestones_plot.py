"""Tests for the medal plot: distinct line styles and the --plot-only rebuild path."""

import json
from pathlib import Path

import pytest

import milestones
from dqn_frogger.milestones import MEDAL_ORDER

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _write_leaderboard(run_dir: Path) -> None:
    def medals(fraction: float) -> dict[str, float]:
        return {name: fraction for name in MEDAL_ORDER}

    rows = [
        {"label": "random", "step": None, "medals": medals(0.0)},
        {"label": "100,000", "step": 100_000, "medals": medals(0.05)},
        {"label": "2,000,000", "step": 2_000_000, "medals": medals(1.0)},
    ]
    (run_dir / "leaderboard.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")


def test_every_medal_has_a_distinct_marker_and_line_style() -> None:
    styles = [milestones.MEDAL_STYLES[name] for name in MEDAL_ORDER]
    assert len(set(styles)) == len(MEDAL_ORDER)


def test_plot_only_rebuilds_milestones_png_from_leaderboard_json(tmp_path: Path) -> None:
    _write_leaderboard(tmp_path)

    output = milestones.rebuild_medal_plot(tmp_path)

    assert output == tmp_path / "milestones.png"
    assert output.read_bytes().startswith(PNG_SIGNATURE)


def test_plot_only_flag_does_not_run_any_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_leaderboard(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("--plot-only must not evaluate episodes")

    monkeypatch.setattr(milestones, "evaluate", forbidden)
    monkeypatch.setattr("sys.argv", ["milestones.py", "--run-dir", str(tmp_path), "--plot-only"])

    milestones.main()

    assert (tmp_path / "milestones.png").exists()


def test_plot_only_without_leaderboard_json_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        milestones.rebuild_medal_plot(tmp_path)

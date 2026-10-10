"""Tests for the leaderboard summaries and Markdown rendering (pure functions)."""

import pytest

from dqn_frogger.leaderboard import (
    EpisodeOutcome,
    MEDAL_EMOJI,
    medal_cell,
    render_markdown,
    summarize,
)
from dqn_frogger.milestones import MilestoneResult


def _outcome(score: float, max_rows: int, homes: int, medals: tuple[str, ...]) -> EpisodeOutcome:
    return EpisodeOutcome(
        score=score,
        milestones=MilestoneResult(
            max_rows=max_rows, homes=homes, medals={name: 0 for name in medals}
        ),
    )


def test_summarize_reports_scores_rows_homes_and_medal_percentages() -> None:
    outcomes = [
        _outcome(20.0, 8, 0, ("first_hop", "road_crossed", "first_log")),
        _outcome(10.0, 6, 0, ("first_hop", "road_crossed")),
    ]
    row = summarize("step 100", outcomes)

    assert row.episodes == 2
    assert row.score_mean == pytest.approx(15.0)
    assert row.score_std == pytest.approx(5.0)
    assert row.max_rows_mean == pytest.approx(7.0)
    assert row.max_rows_max == 8
    assert row.homes_max == 0
    assert row.medal_pct["first_hop"] == pytest.approx(100.0)
    assert row.medal_pct["road_crossed"] == pytest.approx(100.0)
    assert row.medal_pct["first_log"] == pytest.approx(50.0)
    assert row.medal_pct["level_cleared"] == pytest.approx(0.0)


def test_summarize_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        summarize("empty", [])


def test_medal_emoji_appears_only_from_half_of_the_episodes() -> None:
    assert medal_cell("first_hop", 50.0) == f"{MEDAL_EMOJI['first_hop']} 50%"
    assert medal_cell("first_hop", 49.9) == "49.9%"
    assert medal_cell("first_hop", 0.0) == "0%"


def test_render_markdown_has_header_and_one_line_per_row() -> None:
    row = summarize("step 100", [_outcome(5.0, 2, 0, ("first_hop",))])
    markdown = render_markdown([row])
    lines = markdown.splitlines()

    assert lines[0].startswith("| Checkpoint |")
    assert lines[1].startswith("|---")
    assert lines[2].startswith("| step 100 |")
    assert MEDAL_EMOJI["first_hop"] in lines[2]

"""Leaderboard summaries: per-checkpoint medal statistics and their Markdown/CSV rendering."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dqn_frogger.milestones import MEDAL_ORDER, PROXY_MEDALS, MilestoneResult

PROXY_MARK = "*"  # appended to proxy medal headers in Markdown
MEDAL_EMOJI: dict[str, str] = {
    "first_hop": "🐸",
    "road_crossed": "🥉",
    "first_log": "🪵",
    "mid_river": "🥈",
    "frog_home": "🥇",
    "three_homes": "🏆",
    "level_cleared": "👑",
}
MEDAL_SHOWN_FROM_PCT = 50.0


@dataclass(frozen=True)
class EpisodeOutcome:
    """Score and milestones of one evaluation episode."""

    score: float
    milestones: MilestoneResult


@dataclass(frozen=True)
class LeaderboardRow:
    """Aggregated statistics for one policy (a checkpoint or the random baseline)."""

    label: str
    episodes: int
    score_mean: float
    score_std: float
    max_rows_mean: float
    max_rows_max: int
    homes_mean: float
    homes_max: int
    medal_pct: dict[str, float]


def summarize(label: str, outcomes: Sequence[EpisodeOutcome]) -> LeaderboardRow:
    """Aggregate episode outcomes; raises ValueError when there are none."""
    if not outcomes:
        raise ValueError("cannot summarize zero episodes")
    scores = np.array([o.score for o in outcomes], dtype=float)
    max_rows = np.array([o.milestones.max_rows for o in outcomes], dtype=float)
    homes = np.array([o.milestones.homes for o in outcomes], dtype=float)
    n = len(outcomes)
    medal_pct = {
        name: 100.0 * sum(name in o.milestones.medals for o in outcomes) / n
        for name in MEDAL_ORDER
    }
    return LeaderboardRow(
        label=label,
        episodes=n,
        score_mean=float(scores.mean()),
        score_std=float(scores.std()),
        max_rows_mean=float(max_rows.mean()),
        max_rows_max=int(max_rows.max()),
        homes_mean=float(homes.mean()),
        homes_max=int(homes.max()),
        medal_pct=medal_pct,
    )


def medal_cell(name: str, pct: float) -> str:
    """Percentage of episodes with a medal, prefixed by its emoji from 50% upwards."""
    text = f"{pct:g}%"
    if pct >= MEDAL_SHOWN_FROM_PCT:
        return f"{MEDAL_EMOJI[name]} {text}"
    return text


def render_markdown(rows: Sequence[LeaderboardRow]) -> str:
    """Markdown table: one row per policy, medal columns in canonical order."""
    medal_headers = [
        f"{MEDAL_EMOJI[name]} {name}{PROXY_MARK if name in PROXY_MEDALS else ''}"
        for name in MEDAL_ORDER
    ]
    header = ["Checkpoint", "Episodes", "Score", "Max rows (mean / max)", "Homes (mean / max)"]
    lines = [
        "| " + " | ".join(header + medal_headers) + " |",
        "|" + "---|" * (len(header) + len(medal_headers)),
    ]
    for row in rows:
        cells = [
            row.label,
            str(row.episodes),
            f"{row.score_mean:.1f} ± {row.score_std:.1f}",
            f"{row.max_rows_mean:.1f} / {row.max_rows_max}",
            f"{row.homes_mean:.2f} / {row.homes_max}",
        ]
        cells += [medal_cell(name, row.medal_pct[name]) for name in MEDAL_ORDER]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LeaderboardEntry:
    """One leaderboard row plus its artifacts (paths are relative to the run directory)."""

    row: LeaderboardRow
    step: int | None  # None for the random baseline
    screenshots: dict[str, str]  # medal -> screenshot path
    gif: str


def leaderboard_document(
    *, algorithm: str, env_id: str, run_id: str, entries: Sequence[LeaderboardEntry]
) -> dict[str, object]:
    """Algorithm-agnostic JSON document; medal values are fractions in [0, 1]."""
    rows = [
        {
            "label": entry.row.label,
            "step": entry.step,
            "episodes": entry.row.episodes,
            "score_mean": entry.row.score_mean,
            "score_std": entry.row.score_std,
            "max_rows_mean": entry.row.max_rows_mean,
            "max_rows_max": entry.row.max_rows_max,
            "homes_mean": entry.row.homes_mean,
            "homes_max": entry.row.homes_max,
            "medals": {name: pct / 100.0 for name, pct in entry.row.medal_pct.items()},
            "screenshots": dict(entry.screenshots),
            "gif": entry.gif,
        }
        for entry in entries
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm": algorithm,
        "env_id": env_id,
        "run_id": run_id,
        "proxy_medals": [name for name in MEDAL_ORDER if name in PROXY_MEDALS],
        "rows": rows,
    }

"""Milestone tracker: turns per-step rewards and lives into rows, homes and medals.

Calibrated on ALE/Frogger-v5 (see docs/notas-de-desarrollo.md, section 4):
the frog starts on row 0 (safe strip), rows 1-5 are the road, row 6 is the median
strip, rows 7-11 are the river and row 12 is the home row. Each new row gives +1
reward, the row counter restarts when a life is lost, and reaching the home row
gives a home and sends the frog back to the start without losing a life.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

HOME_ROW = 12
HOMES_FOR_LEVEL_CLEARED = 5

# Medal name -> minimum rows reached within one life (max over the episode).
ROW_MEDALS: dict[str, int] = {
    "first_hop": 1,
    "road_crossed": 6,
    "first_log": 7,
    "mid_river": 9,
}
# Medal name -> minimum homes reached within one episode.
HOME_MEDALS: dict[str, int] = {
    "frog_home": 1,
    "three_homes": 3,
    "level_cleared": HOMES_FOR_LEVEL_CLEARED,
}
MEDAL_ORDER: tuple[str, ...] = (
    "first_hop",
    "road_crossed",
    "first_log",
    "mid_river",
    "frog_home",
    "three_homes",
    "level_cleared",
)


@dataclass(frozen=True)
class MilestoneResult:
    """Per-episode summary of what the agent achieved."""

    max_rows: int
    homes: int
    medals: dict[str, int] = field(default_factory=dict)  # medal -> step of first achievement


def format_medals(medals: Iterable[str]) -> str:
    """Join medal names with '|' in canonical order, for CSV output."""
    present = set(medals)
    return "|".join(name for name in MEDAL_ORDER if name in present)


class MilestoneTracker:
    """Feed one `observe` call per environment step; read the summary with `result`."""

    def __init__(self) -> None:
        self._rows = 0
        self._max_rows = 0
        self._homes = 0
        self._lives: int | None = None
        self._medals: dict[str, int] = {}

    def observe(self, step: int, reward: float, lives: int) -> None:
        """Process one step. A positive reward is one new row; a lost life resets the row count."""
        if reward > 0:
            self._rows += 1
            self._max_rows = max(self._max_rows, self._rows)
            if self._rows == HOME_ROW:
                self._homes += 1
                self._rows = 0
        if self._lives is not None and lives < self._lives:
            self._rows = 0
        self._lives = lives

        reached = [name for name, rows in ROW_MEDALS.items() if self._max_rows >= rows]
        reached += [name for name, homes in HOME_MEDALS.items() if self._homes >= homes]
        for name in reached:
            self._medals.setdefault(name, step)

    def result(self) -> MilestoneResult:
        return MilestoneResult(
            max_rows=self._max_rows, homes=self._homes, medals=dict(self._medals)
        )

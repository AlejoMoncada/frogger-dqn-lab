"""Tests for the pure milestone tracker: rows per life, homes and medal thresholds."""

from dqn_frogger.milestones import HOME_ROW, MilestoneTracker, format_medals

Event = tuple[float, int]  # (reward, lives after the step)


def _run(events: list[Event]):
    tracker = MilestoneTracker()
    for step, (reward, lives) in enumerate(events):
        tracker.observe(step, reward, lives)
    return tracker.result()


def _hops(n: int, lives: int = 4) -> list[Event]:
    return [(1.0, lives)] * n


def test_no_hops_means_no_progress() -> None:
    result = _run([(0.0, 4)] * 10)
    assert result.max_rows == 0
    assert result.homes == 0
    assert result.medals == {}


def test_first_hop_is_recorded_at_its_step() -> None:
    result = _run([(0.0, 4), (1.0, 4), (0.0, 4), (1.0, 4)])
    assert result.max_rows == 2
    assert result.medals["first_hop"] == 1


def test_road_crossed_after_six_hops() -> None:
    result = _run(_hops(6))
    assert result.medals["road_crossed"] == 5
    assert "first_log" not in result.medals


def test_river_medals_use_row_thresholds() -> None:
    result = _run(_hops(9))
    assert result.medals["first_log"] == 6
    assert result.medals["mid_river"] == 8


def test_death_resets_rows_but_max_rows_keeps_the_best_life() -> None:
    events = _hops(3) + [(0.0, 3)] + _hops(2, lives=3)
    result = _run(events)
    assert result.max_rows == 3
    assert "road_crossed" not in result.medals


def test_reward_on_the_same_step_as_a_death_still_counts() -> None:
    events = _hops(6) + [(1.0, 3)]  # seventh row reached and a life lost on the same step
    result = _run(events)
    assert result.max_rows == 7
    assert result.medals["first_log"] == 6


def test_reaching_home_row_counts_a_home_and_restarts_the_frog() -> None:
    events = _hops(HOME_ROW) + _hops(3)
    result = _run(events)
    assert result.homes == 1
    assert result.medals["frog_home"] == HOME_ROW - 1
    assert result.max_rows == HOME_ROW


def test_home_row_reward_on_a_death_step_does_not_count_as_home() -> None:
    events = _hops(HOME_ROW - 1) + [(1.0, 3)]  # row 12 reached and a life lost on the same step
    result = _run(events)
    assert result.homes == 0
    assert "frog_home" not in result.medals


def test_home_row_reward_without_death_still_counts_after_a_death_step() -> None:
    events = _hops(HOME_ROW - 1) + [(1.0, 3)] + _hops(HOME_ROW, lives=3)
    result = _run(events)
    assert result.homes == 1  # the second life reaches home cleanly


def test_three_homes_and_level_cleared_thresholds() -> None:
    result = _run(_hops(HOME_ROW) * 3)
    assert result.homes == 3
    assert result.medals["three_homes"] == 3 * HOME_ROW - 1
    assert "level_cleared" not in result.medals

    result = _run(_hops(HOME_ROW) * 5)
    assert result.homes == 5
    assert result.medals["level_cleared"] == 5 * HOME_ROW - 1


def test_medal_keeps_its_first_step_when_reached_again() -> None:
    events = _hops(6) + [(0.0, 3)] + _hops(6, lives=3)
    result = _run(events)
    assert result.medals["road_crossed"] == 5


def test_format_medals_orders_names_canonically() -> None:
    assert format_medals(["road_crossed", "first_hop"]) == "first_hop|road_crossed"
    assert format_medals([]) == ""

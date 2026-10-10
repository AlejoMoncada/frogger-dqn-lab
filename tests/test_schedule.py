"""Tests for the linear epsilon schedule."""

import pytest

from dqn_frogger.schedule import epsilon_at


def test_endpoints() -> None:
    assert epsilon_at(0, start=1.0, end=0.05, decay_steps=100) == pytest.approx(1.0)
    assert epsilon_at(100, start=1.0, end=0.05, decay_steps=100) == pytest.approx(0.05)


def test_stays_at_end_value_after_decay() -> None:
    assert epsilon_at(10_000, start=1.0, end=0.05, decay_steps=100) == pytest.approx(0.05)


def test_midpoint_is_linear() -> None:
    assert epsilon_at(50, start=1.0, end=0.0, decay_steps=100) == pytest.approx(0.5)


def test_monotonically_non_increasing() -> None:
    values = [epsilon_at(s, start=1.0, end=0.05, decay_steps=200) for s in range(0, 300)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_rejects_non_positive_decay_steps() -> None:
    with pytest.raises(ValueError):
        epsilon_at(0, start=1.0, end=0.05, decay_steps=0)

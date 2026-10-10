"""Exploration schedule."""


def epsilon_at(step: int, start: float = 1.0, end: float = 0.05, decay_steps: int = 150_000) -> float:
    """Linearly decay epsilon from `start` to `end` over `decay_steps`, then hold `end`."""
    if decay_steps <= 0:
        raise ValueError("decay_steps must be positive")
    fraction = min(max(step, 0) / decay_steps, 1.0)
    return start + (end - start) * fraction

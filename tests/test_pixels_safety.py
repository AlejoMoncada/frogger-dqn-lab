"""Memory guard decisions and atomic checkpoint writes for the pixel trainer (CPU only)."""

import pytest
import torch

from dqn_frogger.pixels.config import PixelDQNConfig
from dqn_frogger.pixels.replay import PixelReplayBuffer
from dqn_frogger.pixels.safety import (
    SAFETY_MARGIN_BYTES,
    MemoryBudgetError,
    check_memory_budget,
    enforce_memory_budget,
    save_atomic,
)

GB = 1_000_000_000


def test_fits_when_estimate_plus_margin_equals_available() -> None:
    decision = check_memory_budget(estimated_bytes=5 * GB, available_bytes=8 * GB)
    assert decision.required_bytes == 8 * GB
    assert decision.ok


def test_refuses_when_estimate_plus_margin_exceeds_available() -> None:
    decision = check_memory_budget(estimated_bytes=5 * GB + 1, available_bytes=8 * GB)
    assert not decision.ok


def test_default_replay_capacity_is_refused_on_a_small_machine() -> None:
    estimate = PixelReplayBuffer.estimated_bytes(PixelDQNConfig().buffer_size)
    # 100 000 transitions need about 5.6 GB; with the 3 GB margin 8 GB free is not enough.
    assert estimate < 6 * GB
    assert not check_memory_budget(estimate, available_bytes=8 * GB).ok
    assert check_memory_budget(estimate, available_bytes=10 * GB).ok


def test_margin_is_three_gigabytes_by_default() -> None:
    assert SAFETY_MARGIN_BYTES == 3 * GB
    decision = check_memory_budget(estimated_bytes=1 * GB, available_bytes=100 * GB)
    assert decision.margin_bytes == 3 * GB


def test_enforce_raises_with_clear_message_unless_forced() -> None:
    with pytest.raises(MemoryBudgetError, match="--force-memory"):
        enforce_memory_budget(estimated_bytes=6 * GB, available_bytes=4 * GB, force=False)
    decision = enforce_memory_budget(estimated_bytes=6 * GB, available_bytes=4 * GB, force=True)
    assert not decision.ok


def test_enforce_is_silent_when_it_fits() -> None:
    decision = enforce_memory_budget(estimated_bytes=1 * GB, available_bytes=10 * GB, force=False)
    assert decision.ok


def test_memory_budget_error_is_a_value_error() -> None:
    assert issubclass(MemoryBudgetError, ValueError)


def test_save_atomic_replaces_target_and_leaves_no_temp_file(tmp_path) -> None:
    target = tmp_path / "latest.pt"
    save_atomic({"env_step": 1}, target)
    save_atomic({"env_step": 2}, target)

    assert torch.load(target, weights_only=False) == {"env_step": 2}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["latest.pt"]

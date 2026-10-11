"""Memory guard and atomic writes for the pixel trainer.

The guard compares the replay buffer estimate against the RAM that is actually free now
(`psutil.virtual_memory().available`), not against total RAM: other applications may already
hold most of the machine. A run is refused when estimate + margin does not fit.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import torch

from dqn_frogger.pixels.frame_replay import FrameReplay
from dqn_frogger.pixels.replay import PixelReplayBuffer

SAFETY_MARGIN_BYTES = 3_000_000_000  # torch/CUDA/pinned memory and the OS, measured as 2-3 GB


def replay_estimated_bytes(replay_impl: str, capacity: int, num_envs: int, n_step: int) -> int:
    """Host memory of the replay buffer the run will allocate (the selected implementation)."""
    if replay_impl == "frame":
        return FrameReplay.estimated_bytes(capacity, num_envs, n_step)
    if replay_impl == "stacked":
        return PixelReplayBuffer.estimated_bytes(capacity)
    raise ValueError(f"replay_impl must be 'frame' or 'stacked', got {replay_impl!r}")


class MemoryBudgetError(ValueError):
    """The replay buffer plus the safety margin does not fit in the free RAM."""


@dataclass(frozen=True)
class MemoryDecision:
    estimated_bytes: int
    available_bytes: int
    margin_bytes: int

    @property
    def required_bytes(self) -> int:
        return self.estimated_bytes + self.margin_bytes

    @property
    def ok(self) -> bool:
        return self.required_bytes <= self.available_bytes

    def describe(self) -> str:
        return (
            f"replay estimate {self.estimated_bytes / 1e9:.2f} GB + margin "
            f"{self.margin_bytes / 1e9:.1f} GB = {self.required_bytes / 1e9:.2f} GB required; "
            f"available now {self.available_bytes / 1e9:.2f} GB"
        )


def check_memory_budget(
    estimated_bytes: int, available_bytes: int, margin_bytes: int = SAFETY_MARGIN_BYTES
) -> MemoryDecision:
    """Pure decision: the caller supplies both numbers."""
    return MemoryDecision(
        estimated_bytes=int(estimated_bytes),
        available_bytes=int(available_bytes),
        margin_bytes=int(margin_bytes),
    )


def available_memory_bytes() -> int:
    """RAM free right now, as reported by the OS."""
    return int(psutil.virtual_memory().available)


def enforce_memory_budget(
    estimated_bytes: int, available_bytes: int, force: bool = False
) -> MemoryDecision:
    """Raise MemoryBudgetError when the run does not fit, unless `force` is set."""
    decision = check_memory_budget(estimated_bytes, available_bytes)
    if not decision.ok and not force:
        raise MemoryBudgetError(
            f"refusing to start: {decision.describe()}. Lower --buffer-size, close other "
            "applications, or pass --force-memory to start anyway."
        )
    return decision


def save_atomic(payload: dict[str, Any], path: Path) -> None:
    """Write to a temp file in the same directory, then rename over `path`.

    A crash mid-write leaves the previous file intact; `os.replace` is atomic on one volume.
    """
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)

"""Device selection and device-specific helpers shared by the pixel trainer, agent and milestones.

Names: "cpu", "cuda" (NVIDIA), "mps" (Apple Silicon), "auto" (cuda, else mps, else cpu).
"""

import platform

import torch

DEVICE_CHOICES = ("cpu", "cuda", "mps", "auto")


class DeviceUnavailableError(ValueError):
    """The requested accelerator is not available on this machine."""


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(name: str) -> str:
    """Map a device name to a concrete device string, raising if the choice cannot run here."""
    if name == "auto":
        if cuda_available():
            return "cuda"
        if mps_available():
            return "mps"
        return "cpu"
    if name == "cpu":
        return "cpu"
    if name == "cuda":
        if not cuda_available():
            raise DeviceUnavailableError("device 'cuda' was requested but CUDA is not available")
        return "cuda"
    if name == "mps":
        if not mps_available():
            raise DeviceUnavailableError(
                "device 'mps' was requested but Apple MPS is not available "
                "(needs an Apple Silicon Mac and a macOS build of PyTorch)"
            )
        return "mps"
    raise ValueError(f"device must be one of {DEVICE_CHOICES}, got {name!r}")


def default_device() -> str:
    return resolve_device("auto")


def pin_memory_enabled(device: torch.device | str) -> bool:
    """Pinned host memory and non_blocking copies only pay off (and only work) on CUDA."""
    return torch.device(device).type == "cuda"


def sync_device(device: torch.device | str) -> None:
    """Wait for queued device work so timers measure real compute."""
    kind = torch.device(device).type
    if kind == "cuda":
        torch.cuda.synchronize(device)
    elif kind == "mps":
        torch.mps.synchronize()


def host_info() -> dict[str, str | None]:
    """Machine identity for config.json and timing.json; macOS adds the OS version."""
    is_mac = platform.system() == "Darwin"
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "macos_version": platform.mac_ver()[0] if is_mac else None,
    }

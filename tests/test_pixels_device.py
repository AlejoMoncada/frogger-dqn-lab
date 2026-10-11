"""Device selection, pinned-memory gating, device-aware sync and checkpoint portability (CPU only).

MPS and CUDA availability are monkeypatched; no accelerator is touched.
"""

import json

import numpy as np
import pytest
import torch

import train_pixels
from dqn_frogger.pixels import device as device_mod
from dqn_frogger.pixels import trainer as pixel_trainer
from dqn_frogger.pixels.agent import PixelDQNAgent
from dqn_frogger.pixels.config import PixelDQNConfig, default_device
from dqn_frogger.pixels.device import (
    DeviceUnavailableError,
    host_info,
    pin_memory_enabled,
    resolve_device,
    sync_device,
)
from dqn_frogger.pixels.frame_replay import FrameReplay
from dqn_frogger.pixels.replay import PixelReplayBuffer
from dqn_frogger.pixels.safety import save_atomic
from test_pixels_trainer import FakeVectorEnv, tiny_config


def _set_accelerators(monkeypatch, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


# --- resolve_device -------------------------------------------------------------------------


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("mps", [False, True])
def test_cpu_is_always_cpu(monkeypatch, cuda: bool, mps: bool) -> None:
    _set_accelerators(monkeypatch, cuda, mps)
    assert resolve_device("cpu") == "cpu"


def test_cuda_resolves_only_when_available(monkeypatch) -> None:
    _set_accelerators(monkeypatch, cuda=True, mps=False)
    assert resolve_device("cuda") == "cuda"
    _set_accelerators(monkeypatch, cuda=False, mps=True)
    with pytest.raises(DeviceUnavailableError, match="CUDA is not available"):
        resolve_device("cuda")


def test_mps_resolves_only_when_available(monkeypatch) -> None:
    _set_accelerators(monkeypatch, cuda=False, mps=True)
    assert resolve_device("mps") == "mps"
    _set_accelerators(monkeypatch, cuda=True, mps=False)
    with pytest.raises(DeviceUnavailableError, match="MPS is not available"):
        resolve_device("mps")


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch) -> None:
    _set_accelerators(monkeypatch, cuda=True, mps=True)
    assert resolve_device("auto") == "cuda"
    _set_accelerators(monkeypatch, cuda=False, mps=True)
    assert resolve_device("auto") == "mps"
    _set_accelerators(monkeypatch, cuda=False, mps=False)
    assert resolve_device("auto") == "cpu"


def test_default_device_follows_auto(monkeypatch) -> None:
    _set_accelerators(monkeypatch, cuda=False, mps=True)
    assert default_device() == "mps"
    assert PixelDQNConfig(device="auto").device == "mps"


def test_unknown_device_name_is_rejected(monkeypatch) -> None:
    _set_accelerators(monkeypatch, cuda=True, mps=True)
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_device("tpu")


def test_train_pixels_cli_exits_cleanly_when_device_missing(monkeypatch) -> None:
    _set_accelerators(monkeypatch, cuda=False, mps=False)
    with pytest.raises(SystemExit, match="error: device 'mps' was requested"):
        train_pixels.resolve_device("mps")
    assert train_pixels.resolve_device("cpu") == "cpu"


# --- pinned memory ----------------------------------------------------------------------------


def test_pin_memory_only_for_cuda() -> None:
    assert pin_memory_enabled("cuda") is True
    assert pin_memory_enabled(torch.device("cuda", 0)) is True
    assert pin_memory_enabled("mps") is False
    assert pin_memory_enabled("cpu") is False


def _record_pins(monkeypatch) -> list[torch.Tensor]:
    calls: list[torch.Tensor] = []

    def fake_pin(self):
        calls.append(self)
        return self

    monkeypatch.setattr(torch.Tensor, "pin_memory", fake_pin)
    return calls


def _filled_stacked(capacity: int = 16) -> PixelReplayBuffer:
    buffer = PixelReplayBuffer(capacity=capacity)
    for i in range(10):
        frame = np.full((4, 84, 84), i, dtype=np.uint8)
        buffer.add(frame, i % 5, float(i), frame, done=False, discount=0.97)
    return buffer


def _filled_frame(capacity: int = 16) -> FrameReplay:
    replay = FrameReplay(capacity, num_envs=1, n_step=1)
    stream = replay.stream(0)
    ref = stream.start(np.zeros((4, 84, 84), dtype=np.uint8))
    for i in range(10):
        nxt = stream.step(np.full((4, 84, 84), i + 1, dtype=np.uint8))
        replay.add(ref, i % 5, float(i), nxt, False, 0.97)
        ref = nxt
    return replay


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("make", [_filled_stacked, _filled_frame], ids=["stacked", "frame"])
def test_replay_pins_only_when_cuda_is_present(monkeypatch, make, cuda: bool) -> None:
    calls = _record_pins(monkeypatch)
    _set_accelerators(monkeypatch, cuda=cuda, mps=True)
    batch = make().sample(4, np.random.default_rng(0), pin_memory=True)
    assert batch.obs.shape == (4, 4, 84, 84)
    assert len(calls) == (6 if cuda else 0)


@pytest.mark.parametrize("make", [_filled_stacked, _filled_frame], ids=["stacked", "frame"])
def test_replay_never_pins_when_not_requested(monkeypatch, make) -> None:
    calls = _record_pins(monkeypatch)
    _set_accelerators(monkeypatch, cuda=True, mps=False)
    make().sample(4, np.random.default_rng(0), pin_memory=False)
    assert calls == []


# --- device-aware sync --------------------------------------------------------------------------


def test_sync_calls_cuda_synchronize_on_cuda(monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: seen.append("cuda"))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: seen.append("mps"), raising=False)
    sync_device("cuda")
    assert seen == ["cuda"]


def test_sync_calls_mps_synchronize_on_mps(monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: seen.append("cuda"))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: seen.append("mps"), raising=False)
    sync_device("mps")
    assert seen == ["mps"]


def test_sync_is_a_noop_on_cpu(monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: seen.append("cuda"))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: seen.append("mps"), raising=False)
    sync_device("cpu")
    pixel_trainer._sync(torch.device("cpu"))
    assert seen == []


def test_trainer_sync_dispatches_through_device_helper(monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: seen.append("mps"), raising=False)
    pixel_trainer._sync(torch.device("mps"))
    assert seen == ["mps"]


# --- checkpoints across devices ---------------------------------------------------------------


def test_bundle_saved_then_loaded_with_map_location_cpu(tmp_path, monkeypatch) -> None:
    source = PixelDQNAgent(n_actions=3, device="cpu")
    rng = np.random.default_rng(7)
    path = tmp_path / "latest.pt"
    save_atomic(pixel_trainer._bundle(source, PixelDQNConfig.smoke(device="cpu"), rng, 123, 4), path)

    requested: list[object] = []
    real_load = torch.load

    def spy_load(*args, **kwargs):
        requested.append(kwargs.get("map_location"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(pixel_trainer.torch, "load", spy_load)
    target = PixelDQNAgent(n_actions=3, device="cpu")
    state = pixel_trainer._restore_checkpoint(path, target, np.random.default_rng(0))

    assert requested == ["cpu"]
    assert state.env_step == 123 and state.episode == 4
    for a, b in zip(source.online_net.state_dict().values(), target.online_net.state_dict().values()):
        assert torch.equal(a, b)


def test_agent_load_reads_online_weights_with_map_location(tmp_path) -> None:
    source = PixelDQNAgent(n_actions=3, device="cpu")
    path = tmp_path / "dqn.pt"
    source.save(path)
    target = PixelDQNAgent(n_actions=3, device="cpu")
    target.load(path)
    for a, b in zip(source.online_net.state_dict().values(), target.online_net.state_dict().values()):
        assert torch.equal(a, b)


# --- run metadata ------------------------------------------------------------------------------


def test_host_info_reports_machine_and_mac_version_only_on_macos(monkeypatch) -> None:
    monkeypatch.setattr(device_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(device_mod.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(device_mod.platform, "mac_ver", lambda: ("15.1", ("", "", ""), "arm64"))
    info = host_info()
    assert info["machine"] == "arm64" and info["macos_version"] == "15.1"

    monkeypatch.setattr(device_mod.platform, "system", lambda: "Windows")
    assert host_info()["macos_version"] is None


def test_run_records_device_and_host(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        pixel_trainer,
        "make_vector_env",
        lambda num_envs, num_threads=1, **_: FakeVectorEnv(num_envs, num_threads),
    )
    out = tmp_path / "run"
    pixel_trainer.train(tiny_config(out), out)

    timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    run_config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert timing["device"] == "cpu" and run_config["device"] == "cpu"
    assert set(timing["host"]) == {"system", "machine", "processor", "macos_version"}
    assert set(run_config["host"]) == {"system", "machine", "processor", "macos_version"}
    assert timing["host"]["machine"] == run_config["host"]["machine"]

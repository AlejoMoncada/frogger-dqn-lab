"""CPU vs GPU benchmark for the DQN Frogger trainer (same torch build, same seed).

Needs a CUDA build of PyTorch, kept outside the project environment (which is CPU-only):

    uv venv .venv-cuda --python 3.11
    uv pip install --python .venv-cuda torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python .venv-cuda "ale-py==0.12.1" "gymnasium>=1.0,<2" numpy matplotlib imageio pillow
    PYTHONPATH=. .venv-cuda/Scripts/python benchmarks/cpu_vs_gpu.py runs/bench 100000
"""
import json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
from dqn_frogger.config import DQNConfig
from dqn_frogger.trainer import train
from dqn_frogger.env import make_env
from dqn_frogger.agent import DQNAgent
from dqn_frogger.replay_buffer import ReplayBuffer

out_root = Path(sys.argv[1]); steps = int(sys.argv[2])
res = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "steps": steps}

# 1) Emulator only: random actions, no network.
env = make_env(seed=0); env.reset(seed=0); rng = np.random.default_rng(0)
t = time.perf_counter(); n = 20000
for _ in range(n):
    _, _, term, trunc, _ = env.step(int(rng.integers(5)))
    if term or trunc: env.reset()
res["env_only_steps_per_s"] = round(n / (time.perf_counter() - t), 1)

# 2) Network update only: 2000 updates on a fixed batch of 64.
buf = ReplayBuffer(5000, 128); rng = np.random.default_rng(0)
for _ in range(5000):
    o = rng.random(128, dtype=np.float32); buf.add(o, int(rng.integers(5)), 1.0, o, False)
for dev in ["cpu", "cuda"]:
    agent = DQNAgent(obs_dim=128, n_actions=5, device=dev)
    for _ in range(50): agent.update(buf.sample(64, rng))
    if dev == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(2000): agent.update(buf.sample(64, rng))
    if dev == "cuda": torch.cuda.synchronize()
    res[f"update_ms_{dev}"] = round((time.perf_counter() - t) / 2000 * 1000, 3)

# 3) Full training, same seed and config, CPU vs CUDA.
for dev in ["cpu", "cuda"]:
    cfg = replace(DQNConfig(), total_steps=steps, device=dev)
    r = train(cfg, out_root / f"train_{dev}", overwrite=True)
    tj = json.loads((out_root / f"train_{dev}" / "timing.json").read_text())
    ev = (out_root / f"train_{dev}" / "eval.csv").read_text().strip().splitlines()[-1]
    res[f"train_{dev}"] = {"seconds": tj["total_seconds"], "steps_per_s": tj["steps_per_second"], "last_eval_row": ev}
print(json.dumps(res, indent=1))
(out_root / "bench.json").write_text(json.dumps(res, indent=1))

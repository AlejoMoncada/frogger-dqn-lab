"""Training loop: environment interaction, replay, TD learning, logging and checkpointing."""

import csv
import gymnasium
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, TextIO

import ale_py
import numpy as np
import torch

from dqn_frogger.agent import DQNAgent
from dqn_frogger.config import DQNConfig
from dqn_frogger.env import ENV_ID, make_env
from dqn_frogger.evaluation import run_episodes
from dqn_frogger.milestones import MilestoneTracker, format_medals
from dqn_frogger.replay_buffer import ReplayBuffer
from dqn_frogger.schedule import epsilon_at
from dqn_frogger.seeds import EVAL_SEED_BASE, EVAL_SEED_SPAN, validate_training_seed

CHECKPOINT_NAME = "dqn_frogger.pt"
CHECKPOINT_DIR = "checkpoints"
EPISODES_CSV = "episodes.csv"
EVAL_CSV = "eval.csv"
TIMING_JSON = "timing.json"
CONFIG_JSON = "config.json"
REPO_ROOT = Path(__file__).resolve().parent.parent
EPISODE_HEADER = [
    "episode",
    "global_step",
    "return",
    "length",
    "epsilon",
    "max_rows",
    "homes",
    "medals",
]
PROGRESS_EVERY = 10_000
EVAL_SEED_OFFSET = 10_000


class _CsvLog:
    """Append-only CSV writer that flushes every row so partial runs stay readable."""

    def __init__(self, path: Path, header: list[str]) -> None:
        self._file: TextIO = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(header)

    def write(self, row: list[Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _clip(reward: float, enabled: bool) -> float:
    return float(np.sign(reward)) if enabled else reward


def _write_timing(
    path: Path, total_steps: int, seconds: float, device: str
) -> dict[str, Any]:
    timing = {
        "total_steps": total_steps,
        "total_seconds": round(seconds, 2),
        "steps_per_second": round(total_steps / seconds, 1) if seconds > 0 else None,
        "device": device,
        "torch_version": torch.__version__,
        "cpu": platform.processor() or "unknown",
        "python_version": platform.python_version(),
    }
    path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return timing


def _git_commit() -> str | None:
    """Current HEAD commit of this repository, or None when git cannot tell."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _write_run_config(path: Path, config: DQNConfig) -> None:
    """Record everything needed to reproduce the run, before any training happens."""
    active_steps = config.active_checkpoint_steps()
    run_config = {
        "schema_version": 1,
        "config": asdict(config),
        "env": {"id": ENV_ID, "obs_type": "ram", "observation": "float32 in [0, 1], shape (128,)"},
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "ale_py": version("ale-py"),
            "numpy": np.__version__,
        },
        "git_commit": _git_commit(),
        "eval_seed_base": EVAL_SEED_BASE,
        "eval_seed_span": EVAL_SEED_SPAN,
        "train_eval_seed_offset": EVAL_SEED_OFFSET,
        "checkpoints": [f"{CHECKPOINT_DIR}/step_{step}.pt" for step in active_steps],
    }
    path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")


def train(config: DQNConfig, output_dir: Path | str) -> dict[str, Any]:
    """Train a DQN agent on Frogger RAM and return summary metrics."""
    validate_training_seed(config.seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out / CHECKPOINT_DIR
    checkpoint_dir.mkdir(exist_ok=True)
    _write_run_config(out / CONFIG_JSON, config)
    checkpoint_steps = set(config.active_checkpoint_steps())
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    env = make_env(config.seed)
    eval_env = make_env(config.seed + EVAL_SEED_OFFSET)
    eval_rng = np.random.default_rng(config.seed + EVAL_SEED_OFFSET)
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        lr=config.lr,
        gamma=config.gamma,
        device=config.device,
    )
    buffer = ReplayBuffer(config.buffer_size, obs_dim)
    episodes_log = _CsvLog(out / EPISODES_CSV, EPISODE_HEADER)
    eval_log = _CsvLog(out / EVAL_CSV, ["global_step", "mean_return", "std_return", "episodes"])

    obs, _ = env.reset(seed=config.seed)
    tracker = MilestoneTracker()
    episode, episode_return, episode_length = 0, 0.0, 0
    last_eval_mean = float("nan")
    last_loss = float("nan")
    eps = config.eps_start
    started = time.perf_counter()

    for step in range(1, config.total_steps + 1):
        eps = epsilon_at(step - 1, config.eps_start, config.eps_end, config.eps_decay_steps)
        action = agent.act(obs, eps, rng)
        next_obs, reward, terminated, truncated, info = env.step(action)
        tracker.observe(episode_length, float(reward), int(info["lives"]))

        buffer.add(obs, action, _clip(reward, config.reward_clip), next_obs, terminated)
        episode_return += float(reward)
        episode_length += 1
        obs = next_obs

        if terminated or truncated:
            episode += 1
            milestones = tracker.result()
            episodes_log.write(
                [
                    episode,
                    step,
                    episode_return,
                    episode_length,
                    eps,
                    milestones.max_rows,
                    milestones.homes,
                    format_medals(milestones.medals),
                ]
            )
            tracker = MilestoneTracker()
            obs, _ = env.reset()
            episode_return, episode_length = 0.0, 0

        if step >= config.learning_starts and step % config.train_freq == 0:
            if len(buffer) >= config.batch_size:
                last_loss = agent.update(buffer.sample(config.batch_size, rng))

        if step % config.target_update_interval == 0:
            agent.sync_target()

        if step % config.eval_interval == 0 or step == config.total_steps:
            returns = run_episodes(
                eval_env,
                lambda o: agent.act(o, config.eval_epsilon, eval_rng),
                config.eval_episodes,
                seed=config.seed + EVAL_SEED_OFFSET,
            )
            last_eval_mean = float(np.mean(returns))
            eval_log.write([step, last_eval_mean, float(np.std(returns)), len(returns)])
            print(f"[eval] step={step} mean_return={last_eval_mean:.1f}", flush=True)

        if step in checkpoint_steps:
            agent.save(checkpoint_dir / f"step_{step}.pt")

        if step % PROGRESS_EVERY == 0:
            print(
                f"step={step}/{config.total_steps} episodes={episode} "
                f"eps={eps:.3f} loss={last_loss:.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    episodes_log.close()
    eval_log.close()
    checkpoint = out / CHECKPOINT_NAME
    agent.save(checkpoint)
    _write_timing(out / TIMING_JSON, config.total_steps, elapsed, config.device)
    env.close()
    eval_env.close()

    return {
        "total_steps": config.total_steps,
        "episodes": episode,
        "last_eval_mean_return": last_eval_mean,
        "checkpoint": str(checkpoint),
        "seconds": elapsed,
    }

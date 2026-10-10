"""Pixel training loop: N vector envs on CPU threads, batched acting and learning on the device.

Each vector step: one batched epsilon-greedy action per env (device), one `env.step` (CPU
threads), per-env n-step accumulation into the replay buffer, then `updates_per_vector_step`
learner updates (device). Steps are counted as env steps summed over envs.

NextStep autoreset: the step after a done is a reset step (reward 0, action ignored). It is
counted in the step budget but never stored and never shown to the milestone tracker.
"""

import json
import platform
import random
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch

from dqn_frogger.env import ENV_ID
from dqn_frogger.milestones import MilestoneTracker, format_medals
from dqn_frogger.pixels.agent import PixelDQNAgent
from dqn_frogger.pixels.config import PixelDQNConfig
from dqn_frogger.pixels.replay import NStepAccumulator, PixelReplayBuffer
from dqn_frogger.pixels.vector_env import make_vector_env
from dqn_frogger.schedule import epsilon_at
from dqn_frogger.seeds import EVAL_SEED_BASE, EVAL_SEED_SPAN, validate_training_seed
from dqn_frogger.trainer import (
    CHECKPOINT_DIR,
    CHECKPOINT_NAME,
    CONFIG_JSON,
    EPISODE_HEADER,
    EPISODES_CSV,
    EVAL_CSV,
    EVAL_SEED_OFFSET,
    TIMING_JSON,
    _clear_run_artifacts,
    _CsvLog,
    _git_commit,
    _holds_run,
)

PROGRESS_EVERY = 10_000
OBSERVATION_KIND = "pixels"
ALGORITHM_LABEL = "DQN-pixels (Double, n-step)"
TIME_KEYS = ("env_step", "action_selection", "learner_update", "eval")


def _crossed(previous: int, now: int, every: int) -> bool:
    """True when the step counter passed a multiple of `every` during this vector step."""
    return now // every > previous // every


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_run_config(path: Path, config: PixelDQNConfig) -> None:
    """Record everything needed to reproduce the run, before any training happens."""
    run_config = {
        "schema_version": 1,
        "observation": OBSERVATION_KIND,
        "algorithm": ALGORITHM_LABEL,
        "config": asdict(config),
        "env": {
            "id": ENV_ID,
            "obs_type": "pixels",
            "observation": "uint8 stacked grayscale frames (4, 84, 84)",
            "backend": "ale_py.vector_env.AtariVectorEnv",
            "autoreset": "NextStep",
        },
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
        "checkpoints": [
            f"{CHECKPOINT_DIR}/step_{step}.pt" for step in config.active_checkpoint_steps()
        ],
    }
    path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")


def _write_timing(
    path: Path,
    config: PixelDQNConfig,
    seconds: float,
    breakdown: dict[str, float],
) -> dict[str, Any]:
    gpu = torch.cuda.get_device_name(0) if config.device == "cuda" else None
    accounted = sum(breakdown.values())
    timing = {
        "total_steps": config.total_steps,
        "total_seconds": round(seconds, 2),
        "steps_per_second": round(config.total_steps / seconds, 1) if seconds > 0 else None,
        "device": config.device,
        "gpu": gpu,
        "num_envs": config.num_envs,
        "num_threads": config.num_threads,
        "time_breakdown_seconds": {
            **{key: round(breakdown[key], 2) for key in TIME_KEYS},
            "other": round(max(seconds - accounted, 0.0), 2),
        },
        "torch_version": torch.__version__,
        "cpu": platform.processor() or "unknown",
        "python_version": platform.python_version(),
    }
    path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return timing


def _evaluate(
    agent: PixelDQNAgent,
    eval_env,
    config: PixelDQNConfig,
    step: int,
    eval_log: _CsvLog,
) -> float:
    """Greedy evaluation (epsilon = eval_epsilon) on one env with seeds from the shared range."""
    seed = config.seed + EVAL_SEED_OFFSET
    returns: list[float] = []
    for episode in range(config.eval_episodes):
        obs, _ = eval_env.reset(seed=np.array([seed + episode]))
        policy_rng = np.random.default_rng(seed + episode)
        done, total = False, 0.0
        while not done:
            action = agent.act(obs, config.eval_epsilon, policy_rng)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total += float(reward[0])
            done = bool(terminated[0] or truncated[0])
        returns.append(total)
    mean = float(np.mean(returns))
    eval_log.write(
        [
            step,
            mean,
            float(np.std(returns)),
            len(returns),
            seed,
            seed + len(returns) - 1,
        ]
    )
    print(f"[eval] step={step} mean_return={mean:.1f}", flush=True)
    return mean


def train(config: PixelDQNConfig, output_dir: Path | str, overwrite: bool = False) -> dict[str, Any]:
    """Train the pixel DQN on Frogger and return summary metrics.

    Refuses a directory that already holds a run unless `overwrite` is True, in which case only
    the known run artifacts are cleared first (same rule as the RAM trainer).
    """
    validate_training_seed(config.seed)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device 'cuda' was requested but CUDA is not available")
    out = Path(output_dir)
    if _holds_run(out):
        if not overwrite:
            raise FileExistsError(
                f"{out} already holds a run; choose another --output-dir or pass --overwrite"
            )
        _clear_run_artifacts(out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out / CHECKPOINT_DIR
    checkpoint_dir.mkdir(exist_ok=True)
    _write_run_config(out / CONFIG_JSON, config)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)

    env = make_vector_env(num_envs=config.num_envs, num_threads=config.num_threads)
    eval_env = make_vector_env(num_envs=1, num_threads=1)
    n_actions = int(env.single_action_space.n)
    num_envs = config.num_envs

    agent = PixelDQNAgent(n_actions=n_actions, lr=config.lr, gamma=config.gamma, device=config.device)
    buffer = PixelReplayBuffer(config.buffer_size)
    estimate_gb = PixelReplayBuffer.estimated_bytes(config.buffer_size) / 1e9
    print(
        f"replay capacity={config.buffer_size:,} transitions, ~{estimate_gb:.1f} GB host RAM; "
        f"device={config.device}, num_envs={num_envs}",
        flush=True,
    )
    accumulators = [NStepAccumulator(config.n_step, config.gamma, buffer) for _ in range(num_envs)]
    trackers = [MilestoneTracker() for _ in range(num_envs)]
    episode_return = np.zeros(num_envs, dtype=np.float64)
    episode_length = np.zeros(num_envs, dtype=np.int64)
    skip_next = np.zeros(num_envs, dtype=bool)

    episodes_log = _CsvLog(out / EPISODES_CSV, EPISODE_HEADER)
    eval_log = _CsvLog(
        out / EVAL_CSV,
        ["global_step", "mean_return", "std_return", "episodes", "first_seed", "last_seed"],
    )
    pending_checkpoints = list(config.active_checkpoint_steps())
    breakdown = {key: 0.0 for key in TIME_KEYS}

    obs, _ = env.reset(seed=np.arange(config.seed, config.seed + num_envs))
    episode = 0
    env_steps = 0
    last_eval_step = -1
    last_loss = float("nan")
    eps = config.eps_start
    started = time.perf_counter()

    while env_steps < config.total_steps:
        previous = env_steps
        now = previous + num_envs
        eps = epsilon_at(previous, config.eps_start, config.eps_end, config.eps_decay_steps)
        snapshot = np.array(obs, copy=True)

        t0 = time.perf_counter()
        actions = agent.act(snapshot, eps, rng)
        breakdown["action_selection"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        breakdown["env_step"] += time.perf_counter() - t0
        lives = info["lives"]

        for i in range(num_envs):
            if skip_next[i]:
                skip_next[i] = False  # NextStep reset step: not a transition
                continue
            reward = float(rewards[i])
            trackers[i].observe(int(episode_length[i]), reward, int(lives[i]))
            clipped = float(np.sign(reward)) if config.reward_clip else reward
            accumulators[i].push(
                snapshot[i], actions[i], clipped, next_obs[i], bool(terminated[i]), bool(truncated[i])
            )
            episode_return[i] += reward
            episode_length[i] += 1
            if terminated[i] or truncated[i]:
                episode += 1
                milestones = trackers[i].result()
                episodes_log.write(
                    [
                        episode,
                        now,
                        float(episode_return[i]),
                        int(episode_length[i]),
                        eps,
                        milestones.max_rows,
                        milestones.homes,
                        format_medals(milestones.medals),
                    ]
                )
                trackers[i] = MilestoneTracker()
                episode_return[i] = 0.0
                episode_length[i] = 0
                skip_next[i] = True
        obs = next_obs
        env_steps = now

        if now >= config.learning_starts:
            for _ in range(config.updates_per_vector_step):
                if len(buffer) >= config.batch_size:
                    t0 = time.perf_counter()
                    batch = buffer.sample(
                        config.batch_size, rng, pin_memory=device.type == "cuda"
                    )
                    last_loss = agent.update(batch)
                    _sync(device)
                    breakdown["learner_update"] += time.perf_counter() - t0

        if _crossed(previous, now, config.target_update_interval):
            agent.sync_target()

        if _crossed(previous, now, config.eval_interval):
            last_eval_step = now
            t0 = time.perf_counter()
            _evaluate(agent, eval_env, config, now, eval_log)
            breakdown["eval"] += time.perf_counter() - t0

        while pending_checkpoints and pending_checkpoints[0] <= now:
            step = pending_checkpoints.pop(0)
            agent.save(checkpoint_dir / f"step_{step}.pt")

        if _crossed(previous, now, PROGRESS_EVERY):
            print(
                f"step={now}/{config.total_steps} episodes={episode} "
                f"eps={eps:.3f} loss={last_loss:.4f}",
                flush=True,
            )

    if last_eval_step != env_steps:
        t0 = time.perf_counter()
        _evaluate(agent, eval_env, config, env_steps, eval_log)
        breakdown["eval"] += time.perf_counter() - t0
    elapsed = time.perf_counter() - started

    episodes_log.close()
    eval_log.close()
    checkpoint = out / CHECKPOINT_NAME
    agent.save(checkpoint)
    _write_timing(out / TIMING_JSON, config, elapsed, breakdown)
    env.close()
    eval_env.close()

    return {
        "total_steps": config.total_steps,
        "episodes": episode,
        "checkpoint": str(checkpoint),
        "seconds": elapsed,
        "steps_per_second": config.total_steps / elapsed if elapsed > 0 else None,
    }

"""Pixel training loop: N vector envs on CPU threads, batched acting and learning on the device.

Each vector step: one batched epsilon-greedy action per env (device), one `env.step` (CPU
threads), per-env n-step accumulation into the replay buffer, then `updates_per_vector_step`
learner updates (device). Steps are counted as env steps summed over envs.

NextStep autoreset: the step after a done is a reset step (reward 0, action ignored). It is
counted in the step budget but never stored and never shown to the milestone tracker.

Replay layout (`replay_impl`): "frame" (default) stores each 84x84 frame once per env step and
transitions keep frame indices (`frame_replay.py`, about 7.1 KB per transition); "stacked" keeps
obs and next_obs as full (4, 84, 84) stacks (`replay.py`, about 56.5 KB per transition).

Safety (added after a 5M run was killed at 4.6M steps by an out-of-memory condition):
- Memory guard: before anything is written, the replay estimate plus a 3 GB margin must fit in
  the RAM that is free now; otherwise the run is refused unless `force_memory` is set.
- Checkpoints: every `save_interval` env steps the trainer writes `checkpoints/step_<N>.pt` and
  atomically overwrites `checkpoints/latest.pt`. Named `checkpoint_steps` are saved the same way.
  Each bundle holds online and target networks, optimizer state, env step, episode counter,
  epsilon position, numpy/torch RNG states and the config. A KeyboardInterrupt saves `latest.pt`
  before the exception propagates.
- Resume (`resume=True`): restores all of the above from `checkpoints/latest.pt`, appends to
  episodes.csv and eval.csv, and records `resumed_from_step` in config.json and timing.json.
  The replay buffer is NOT checkpointed: it starts empty and `learning_starts` steps of warmup
  are applied again after the resume point before learner updates resume. Partial episodes
  at the resume point are discarded, and the environments are reset with seeds offset by the
  resume step.
"""

import csv
import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, TextIO

import gymnasium
import numpy as np
import torch

from dqn_frogger.env import ENV_ID
from dqn_frogger.milestones import MilestoneTracker, format_medals
from dqn_frogger.pixels.agent import PixelDQNAgent
from dqn_frogger.pixels.config import PixelDQNConfig, transform_reward, upgrade_saved_config
from dqn_frogger.pixels.frame_replay import FrameReplay
from dqn_frogger.pixels.replay import NStepAccumulator, PixelReplayBuffer
from dqn_frogger.pixels.safety import (
    available_memory_bytes,
    enforce_memory_budget,
    replay_estimated_bytes,
    save_atomic,
)
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
    _git_commit,
    _holds_run,
)

PROGRESS_EVERY = 10_000
OBSERVATION_KIND = "pixels"
ALGORITHM_LABEL = "DQN-pixels (Double, n-step)"
TIME_KEYS = ("env_step", "action_selection", "learner_update", "eval")
LATEST_NAME = "latest.pt"
BUNDLE_FORMAT = "pixel-dqn-bundle/v1"
# Settings that may change between the original run and a resume (extending or re-saving it).
RESUME_FREE_FIELDS = frozenset(
    {
        "total_steps",
        "save_interval",
        "checkpoint_steps",
        "buffer_size",
        "replay_impl",
        "device",
        "num_threads",
        "eval_interval",
        "eval_episodes",
    }
)


@dataclass(frozen=True)
class ResumedState:
    env_step: int
    episode: int
    epsilon: float
    config: dict[str, Any]


class _CsvFile:
    """Flushed CSV writer. With `append`, an existing non-empty file keeps its rows and header."""

    def __init__(self, path: Path, header: list[str], append: bool = False) -> None:
        keep_existing = append and path.exists() and path.stat().st_size > 0
        self._file: TextIO = path.open("a" if keep_existing else "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if not keep_existing:
            self._writer.writerow(header)

    def write(self, row: list[Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _crossed(previous: int, now: int, every: int) -> bool:
    """True when the step counter passed a multiple of `every` during this vector step."""
    return now // every > previous // every


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_run_config(
    path: Path, config: PixelDQNConfig, resumed_from_step: int | None = None
) -> None:
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
        "resumed_from_step": resumed_from_step,
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
    session_steps: int,
    resumed_from_step: int | None,
) -> dict[str, Any]:
    gpu = torch.cuda.get_device_name(0) if config.device == "cuda" else None
    accounted = sum(breakdown.values())
    timing = {
        "total_steps": config.total_steps,
        "session_steps": session_steps,
        "resumed_from_step": resumed_from_step,
        "total_seconds": round(seconds, 2),
        "steps_per_second": round(session_steps / seconds, 1) if seconds > 0 else None,
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
    eval_log: _CsvFile,
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


def _capture_rng(rng: np.random.Generator) -> dict[str, Any]:
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {"numpy": rng.bit_generator.state, "torch_cpu": torch.get_rng_state(), "torch_cuda": cuda}


def _restore_rng(state: dict[str, Any], rng: np.random.Generator) -> None:
    rng.bit_generator.state = state["numpy"]
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _bundle(
    agent: PixelDQNAgent,
    config: PixelDQNConfig,
    rng: np.random.Generator,
    env_step: int,
    episode: int,
) -> dict[str, Any]:
    """Everything a resume needs except the replay buffer (which is refilled, not saved)."""
    return {
        "format": BUNDLE_FORMAT,
        **agent.resumable_state(),
        "env_step": env_step,
        "episode": episode,
        "epsilon": epsilon_at(env_step, config.eps_start, config.eps_end, config.eps_decay_steps),
        "rng": _capture_rng(rng),
        "config": asdict(config),
    }


def _restore_checkpoint(path: Path, agent: PixelDQNAgent, rng: np.random.Generator) -> ResumedState:
    """Load a bundle into the agent and RNG. Tensors are loaded on CPU and moved by torch."""
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"{path} is not a resumable checkpoint (expected format {BUNDLE_FORMAT})")
    agent.restore_resumable_state(bundle)
    _restore_rng(bundle["rng"], rng)
    return ResumedState(
        env_step=int(bundle["env_step"]),
        episode=int(bundle["episode"]),
        epsilon=float(bundle["epsilon"]),
        config=bundle["config"],
    )


def _changed_hyperparameters(saved: dict[str, Any], current: PixelDQNConfig) -> list[str]:
    now = asdict(current)
    saved = upgrade_saved_config(saved)
    return sorted(
        key for key in now if key not in RESUME_FREE_FIELDS and saved.get(key) != now[key]
    )


def train(
    config: PixelDQNConfig,
    output_dir: Path | str,
    overwrite: bool = False,
    resume: bool = False,
    force_memory: bool = False,
) -> dict[str, Any]:
    """Train the pixel DQN on Frogger and return summary metrics.

    A directory that already holds a run is refused unless `overwrite` (clears the known run
    artifacts first) or `resume` (continues from `checkpoints/latest.pt`) is given.
    """
    validate_training_seed(config.seed)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device 'cuda' was requested but CUDA is not available")
    if resume and overwrite:
        raise ValueError("--resume and --overwrite cannot be combined")
    out = Path(output_dir)
    checkpoint_dir = out / CHECKPOINT_DIR
    latest_path = checkpoint_dir / LATEST_NAME

    decision = enforce_memory_budget(
        replay_estimated_bytes(config.replay_impl, config.buffer_size, config.num_envs, config.n_step),
        available_memory_bytes(),
        force=force_memory,
    )
    print(f"memory: {decision.describe()}" + ("  [forced]" if not decision.ok else ""), flush=True)

    if resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"cannot resume: {latest_path} does not exist")
    elif _holds_run(out):
        if not overwrite:
            raise FileExistsError(
                f"{out} already holds a run; choose another --output-dir, pass --resume, "
                "or pass --overwrite"
            )
        _clear_run_artifacts(out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

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
    resumed = _restore_checkpoint(latest_path, agent, rng) if resume else None
    start_step = resumed.env_step if resumed else 0
    learning_from = start_step + config.learning_starts
    _write_run_config(out / CONFIG_JSON, config, resumed_from_step=start_step if resumed else None)

    if config.replay_impl == "frame":
        buffer = FrameReplay(config.buffer_size, num_envs, config.n_step)
    else:
        buffer = PixelReplayBuffer(config.buffer_size)
    estimate_gb = (
        replay_estimated_bytes(config.replay_impl, config.buffer_size, num_envs, config.n_step) / 1e9
    )
    print(
        f"replay capacity={config.buffer_size:,} transitions ({config.replay_impl}), "
        f"~{estimate_gb:.2f} GB host RAM; device={config.device}, num_envs={num_envs}",
        flush=True,
    )
    if resumed:
        changed = _changed_hyperparameters(resumed.config, config)
        if changed:
            print(f"warning: resuming with changed settings: {', '.join(changed)}", flush=True)
        print(
            f"resumed_from_step={start_step} episode={resumed.episode} epsilon={resumed.epsilon:.3f}; "
            f"replay starts empty, learner waits until step {learning_from:,}",
            flush=True,
        )
    accumulators = [NStepAccumulator(config.n_step, config.gamma, buffer) for _ in range(num_envs)]
    streams = [buffer.stream(i) for i in range(num_envs)]
    trackers = [MilestoneTracker() for _ in range(num_envs)]
    episode_return = np.zeros(num_envs, dtype=np.float64)
    episode_length = np.zeros(num_envs, dtype=np.int64)
    skip_next = np.zeros(num_envs, dtype=bool)

    resuming = resumed is not None
    episodes_log = _CsvFile(out / EPISODES_CSV, EPISODE_HEADER, append=resuming)
    eval_log = _CsvFile(
        out / EVAL_CSV,
        ["global_step", "mean_return", "std_return", "episodes", "first_seed", "last_seed"],
        append=resuming,
    )
    pending_checkpoints = [s for s in config.active_checkpoint_steps() if s > start_step]
    breakdown = {key: 0.0 for key in TIME_KEYS}

    # Offset the env seeds on resume so the restarted episodes differ from the first run's.
    env_seed = config.seed + start_step
    obs, _ = env.reset(seed=np.arange(env_seed, env_seed + num_envs))
    refs = [streams[i].start(obs[i]) for i in range(num_envs)]  # references to the current obs
    episode = resumed.episode if resumed else 0
    env_steps = start_step
    last_eval_step = -1
    last_loss = float("nan")
    eps = resumed.epsilon if resumed else config.eps_start
    started = time.perf_counter()

    try:
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
                    skip_next[i] = False  # NextStep reset step: not a transition, but it starts the episode
                    refs[i] = streams[i].start(next_obs[i])
                    continue
                reward = float(rewards[i])
                trackers[i].observe(int(episode_length[i]), reward, int(lives[i]))
                shaped = transform_reward(reward, config.reward_mode, config.reward_scale)
                next_ref = streams[i].step(next_obs[i])
                accumulators[i].push(
                    refs[i], actions[i], shaped, next_ref, bool(terminated[i]), bool(truncated[i])
                )
                refs[i] = next_ref
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

            if now >= learning_from:
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
                save_atomic(
                    _bundle(agent, config, rng, now, episode), checkpoint_dir / f"step_{step}.pt"
                )

            if _crossed(previous, now, config.save_interval):
                bundle = _bundle(agent, config, rng, now, episode)
                save_atomic(bundle, checkpoint_dir / f"step_{now}.pt")
                save_atomic(bundle, latest_path)
                print(f"[checkpoint] step={now} saved step_{now}.pt and {LATEST_NAME}", flush=True)

            if _crossed(previous, now, PROGRESS_EVERY):
                print(
                    f"step={now}/{config.total_steps} episodes={episode} "
                    f"eps={eps:.3f} loss={last_loss:.4f}",
                    flush=True,
                )
    except KeyboardInterrupt:
        save_atomic(_bundle(agent, config, rng, env_steps, episode), latest_path)
        print(
            f"[interrupted] step={env_steps}: saved {latest_path}; continue with --resume",
            flush=True,
        )
        episodes_log.close()
        eval_log.close()
        env.close()
        eval_env.close()
        raise

    if last_eval_step != env_steps:
        t0 = time.perf_counter()
        _evaluate(agent, eval_env, config, env_steps, eval_log)
        breakdown["eval"] += time.perf_counter() - t0
    elapsed = time.perf_counter() - started

    episodes_log.close()
    eval_log.close()
    save_atomic(_bundle(agent, config, rng, env_steps, episode), latest_path)
    checkpoint = out / CHECKPOINT_NAME
    agent.save(checkpoint)
    session_steps = env_steps - start_step
    _write_timing(
        out / TIMING_JSON,
        config,
        elapsed,
        breakdown,
        session_steps=session_steps,
        resumed_from_step=start_step if resumed else None,
    )
    env.close()
    eval_env.close()

    return {
        "total_steps": config.total_steps,
        "episodes": episode,
        "checkpoint": str(checkpoint),
        "resumed_from_step": start_step if resumed else None,
        "seconds": elapsed,
        "steps_per_second": session_steps / elapsed if elapsed > 0 else None,
    }

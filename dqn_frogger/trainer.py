"""Training loop: environment interaction, replay, TD learning, logging and checkpointing."""

import csv
import random
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch

from dqn_frogger.agent import DQNAgent
from dqn_frogger.config import DQNConfig
from dqn_frogger.env import make_env
from dqn_frogger.evaluation import run_episodes
from dqn_frogger.replay_buffer import ReplayBuffer
from dqn_frogger.schedule import epsilon_at

CHECKPOINT_NAME = "dqn_frogger.pt"
EPISODES_CSV = "episodes.csv"
EVAL_CSV = "eval.csv"
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


def train(config: DQNConfig, output_dir: Path | str) -> dict[str, Any]:
    """Train a DQN agent on Frogger RAM and return summary metrics."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
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
    episodes_log = _CsvLog(
        out / EPISODES_CSV, ["episode", "global_step", "return", "length", "epsilon"]
    )
    eval_log = _CsvLog(out / EVAL_CSV, ["global_step", "mean_return", "std_return", "episodes"])

    obs, _ = env.reset(seed=config.seed)
    episode, episode_return, episode_length = 0, 0.0, 0
    last_eval_mean = float("nan")
    last_loss = float("nan")
    eps = config.eps_start

    for step in range(1, config.total_steps + 1):
        eps = epsilon_at(step - 1, config.eps_start, config.eps_end, config.eps_decay_steps)
        action = agent.act(obs, eps, rng)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        buffer.add(obs, action, _clip(reward, config.reward_clip), next_obs, terminated)
        episode_return += float(reward)
        episode_length += 1
        obs = next_obs

        if terminated or truncated:
            episode += 1
            episodes_log.write([episode, step, episode_return, episode_length, eps])
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

        if step % PROGRESS_EVERY == 0:
            print(
                f"step={step}/{config.total_steps} episodes={episode} "
                f"eps={eps:.3f} loss={last_loss:.4f}",
                flush=True,
            )

    episodes_log.close()
    eval_log.close()
    checkpoint = out / CHECKPOINT_NAME
    agent.save(checkpoint)
    env.close()
    eval_env.close()

    return {
        "total_steps": config.total_steps,
        "episodes": episode,
        "last_eval_mean_return": last_eval_mean,
        "checkpoint": str(checkpoint),
    }

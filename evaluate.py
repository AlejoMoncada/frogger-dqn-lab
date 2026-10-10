"""Evaluate a trained DQN checkpoint against a random baseline."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from dqn_frogger.agent import DQNAgent
from dqn_frogger.env import make_env
from dqn_frogger.evaluation import run_episodes

EVAL_SEED = 123_456


def _summary(returns: list[float]) -> dict[str, Any]:
    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "episodes": returns,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DQN and a random policy on Frogger.")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("results/dqn_frogger.pt"), help="Model weights."
    )
    parser.add_argument("--episodes", type=int, default=10, help="Greedy episodes to run.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Epsilon for the DQN policy.")
    parser.add_argument(
        "--output", type=Path, default=Path("results/evaluation.json"), help="JSON output path."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_env(EVAL_SEED)
    n_actions = int(env.action_space.n)
    obs_dim = int(env.observation_space.shape[0])

    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions)
    agent.load(args.checkpoint)
    rng = np.random.default_rng(EVAL_SEED)

    dqn_returns = run_episodes(
        env, lambda o: agent.act(o, args.epsilon, rng), args.episodes, seed=EVAL_SEED
    )
    random_returns = run_episodes(
        env, lambda _: int(env.action_space.sample()), args.episodes, seed=EVAL_SEED
    )
    env.close()

    result = {
        "checkpoint": str(args.checkpoint),
        "epsilon": args.epsilon,
        "dqn": _summary(dqn_returns),
        "random": _summary(random_returns),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"DQN    mean={result['dqn']['mean']:.1f} std={result['dqn']['std']:.1f}")
    print(f"Random mean={result['random']['mean']:.1f} std={result['random']['std']:.1f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

"""Plot the learning curve from the training CSVs."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dqn_frogger.trainer import EPISODES_CSV, EVAL_CSV  # noqa: E402

MOVING_AVG_WINDOW = 50
OUTPUT_NAME = "learning_curve.png"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.full(len(values), np.nan)
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), smoothed])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DQN learning curve.")
    parser.add_argument(
        "--input-dir", type=Path, default=Path("results"), help="Directory with the CSV logs."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = _read_csv(args.input_dir / EPISODES_CSV)
    evals = _read_csv(args.input_dir / EVAL_CSV)

    fig, ax = plt.subplots(figsize=(9, 5))
    if episodes:
        steps = np.array([float(r["global_step"]) for r in episodes])
        returns = np.array([float(r["return"]) for r in episodes])
        ax.plot(steps, returns, color="tab:blue", alpha=0.3, linewidth=0.8, label="episode return")
        ax.plot(
            steps,
            _moving_average(returns, MOVING_AVG_WINDOW),
            color="tab:blue",
            linewidth=2,
            label=f"moving average ({MOVING_AVG_WINDOW} episodes)",
        )
    if evals:
        ax.errorbar(
            [float(r["global_step"]) for r in evals],
            [float(r["mean_return"]) for r in evals],
            yerr=[float(r["std_return"]) for r in evals],
            fmt="o-",
            color="tab:orange",
            capsize=3,
            label="greedy evaluation (mean ± std)",
        )

    ax.set_xlabel("global step (agent steps)")
    ax.set_ylabel("score (raw, unclipped)")
    ax.set_title("DQN on ALE/Frogger-v5 (RAM)")
    ax.grid(alpha=0.3)
    if episodes or evals:
        ax.legend(loc="upper left")
    fig.tight_layout()

    output = args.input_dir / OUTPUT_NAME
    fig.savefig(output, dpi=120)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

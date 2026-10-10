"""Milestone leaderboard: medals per checkpoint, first-achievement screenshots and GIFs.

Usage: uv run python milestones.py --run-dir results/run_X --episodes 20
"""

import argparse
import csv
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dqn_frogger.agent import DQNAgent  # noqa: E402
from dqn_frogger.env import ENV_ID, make_env  # noqa: E402
from dqn_frogger.leaderboard import (  # noqa: E402
    MEDAL_EMOJI,
    PROXY_MARK,
    EpisodeOutcome,
    LeaderboardEntry,
    LeaderboardRow,
    leaderboard_document,
    render_markdown,
    summarize,
)
from dqn_frogger.milestones import (  # noqa: E402
    HOME_MEDALS,
    HOME_ROW,
    MEDAL_ORDER,
    PROXY_MEDALS,
    ROW_MEDALS,
    MilestoneTracker,
)
from dqn_frogger.seeds import EVAL_SEED_BASE  # noqa: E402

EVAL_EPSILON = 0.01
GIF_FRAME_EVERY = 2  # keep every 2nd environment step
GIF_MAX_FRAMES = 900
GIF_DOWNSCALE = 2
GIF_FPS = 30
CHECKPOINT_RE = re.compile(r"step_(\d+)\.pt$")
Policy = Callable[[np.ndarray, np.random.Generator], int]


@dataclass
class Episode:
    """Outcome of one evaluation episode plus the frames the report needs."""

    outcome: EpisodeOutcome
    medal_frames: dict[str, np.ndarray] = field(default_factory=dict)
    gif_frames: list[np.ndarray] = field(default_factory=list)


def play_episode(
    env, policy: Policy, seed: int, keep_gif: bool
) -> Episode:  # env is a gymnasium env with rgb_array rendering
    """Play one episode, keeping the first frame of every medal and optionally a GIF.

    The policy gets its own generator seeded from the episode seed, so the episode does not
    depend on how many episodes were played before it.
    """
    obs, _ = env.reset(seed=seed)
    policy_rng = np.random.default_rng(seed)
    tracker = MilestoneTracker()
    episode = Episode(outcome=EpisodeOutcome(score=0.0, milestones=tracker.result()))
    score, step, done = 0.0, 0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(obs, policy_rng))
        tracker.observe(step, float(reward), int(info["lives"]))
        score += float(reward)
        frame = env.render()
        for name in tracker.result().medals:
            episode.medal_frames.setdefault(name, frame)
        if keep_gif and step % GIF_FRAME_EVERY == 0 and len(episode.gif_frames) < GIF_MAX_FRAMES:
            episode.gif_frames.append(
                np.ascontiguousarray(frame[::GIF_DOWNSCALE, ::GIF_DOWNSCALE])
            )
        step += 1
        done = terminated or truncated
    episode.outcome = EpisodeOutcome(score=score, milestones=tracker.result())
    return episode


def find_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    """Checkpoints named step_<N>.pt, sorted by step."""
    found = []
    for path in checkpoint_dir.glob("step_*.pt"):
        match = CHECKPOINT_RE.search(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def find_run_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Checkpoints of one run: the list recorded in its config.json, else a glob fallback.

    The manifest keeps stale step_<N>.pt files from other runs out of the leaderboard.
    """
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return find_checkpoints(run_dir / "checkpoints")
    manifest = json.loads(config_path.read_text(encoding="utf-8")).get("checkpoints")
    if manifest is None:
        return find_checkpoints(run_dir / "checkpoints")
    found = []
    for relative in manifest:
        path = run_dir / relative
        if not path.exists():
            raise FileNotFoundError(f"config.json lists {relative} but the file is missing")
        found.append((int(CHECKPOINT_RE.search(path.name).group(1)), path))
    return sorted(found)


def evaluate(env, policy: Policy, episodes: int, keep_gif_first: bool) -> list[Episode]:
    return [
        play_episode(env, policy, EVAL_SEED_BASE + i, keep_gif=keep_gif_first and i == 0)
        for i in range(episodes)
    ]


def write_gif(path: Path, frames: list[np.ndarray]) -> None:
    if frames:
        imageio.mimsave(path, frames, duration=1 / GIF_FPS, loop=0)


def write_screenshots(run_dir: Path, step: int, episodes: list[Episode]) -> dict[str, str]:
    """Save the first frame of each medal from the first episode that achieved it.

    Returns medal -> path relative to run_dir.
    """
    saved: dict[str, str] = {}
    for name in MEDAL_ORDER:
        for episode in episodes:
            if name in episode.medal_frames:
                relative = f"screenshots/step_{step}_{name}.png"
                plt.imsave(run_dir / relative, episode.medal_frames[name])
                saved[name] = relative
                break
    return saved


def write_csv(path: Path, rows: list[LeaderboardRow]) -> None:
    fields = ["checkpoint", "episodes", "score_mean", "score_std", "max_rows_mean",
              "max_rows_max", "homes_mean", "homes_max"]
    fields += [f"pct_{name}" for name in MEDAL_ORDER]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for row in rows:
            writer.writerow(
                [row.label, row.episodes, row.score_mean, row.score_std, row.max_rows_mean,
                 row.max_rows_max, row.homes_mean, row.homes_max]
                + [row.medal_pct[name] for name in MEDAL_ORDER]
            )


def legend_markdown() -> str:
    lines = ["", "Medal thresholds (medal emoji shown when at least 50% of episodes achieved it):", ""]
    for name in MEDAL_ORDER:
        if name in ROW_MEDALS:
            rule = f"reach row {ROW_MEDALS[name]} in one life (home row is {HOME_ROW})"
        else:
            rule = f"{HOME_MEDALS[name]} frog(s) home in one episode"
        mark = PROXY_MARK if name in PROXY_MEDALS else ""
        lines.append(f"- {MEDAL_EMOJI[name]} `{name}`{mark}: {rule}")
    lines += [
        "",
        f"{PROXY_MARK} Unverified proxy: home detection is not validated against RAM traces, "
        "so this medal is an estimate.",
    ]
    return "\n".join(lines) + "\n"


def plot_medals(path: Path, steps: list[int], rows: list[LeaderboardRow]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in MEDAL_ORDER:
        ax.plot(steps, [row.medal_pct[name] for row in rows], marker="o", label=name)
    ax.set_xlabel("training step")
    ax.set_ylabel("% of evaluation episodes")
    ax.set_ylim(-2, 102)
    ax.set_title("Milestones achieved by checkpoint (greedy, eps=0.01)")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the milestone leaderboard for a run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run with checkpoints/.")
    parser.add_argument("--episodes", type=int, default=20, help="Greedy episodes per checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir: Path = args.run_dir
    checkpoints = find_run_checkpoints(run_dir)
    if not checkpoints:
        raise SystemExit(f"no step_<N>.pt checkpoints in {run_dir / 'checkpoints'}")
    for sub in ("screenshots", "gifs"):
        (run_dir / sub).mkdir(exist_ok=True)

    env = make_env(EVAL_SEED_BASE, render_mode="rgb_array")
    n_actions = int(env.action_space.n)
    obs_dim = int(env.observation_space.shape[0])

    random_eps = evaluate(env, lambda _obs, rng: int(rng.integers(n_actions)), args.episodes, True)
    write_gif(run_dir / "gifs" / "random.gif", random_eps[0].gif_frames)
    rows = [summarize("random", [e.outcome for e in random_eps])]
    entries = [LeaderboardEntry(row=rows[0], step=None, screenshots={}, gif="gifs/random.gif")]

    steps: list[int] = []
    for step, path in checkpoints:
        agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions)
        agent.load(path)
        episodes = evaluate(
            env, lambda obs, rng, a=agent: a.act(obs, EVAL_EPSILON, rng), args.episodes, True
        )
        shots = write_screenshots(run_dir, step, episodes)
        write_gif(run_dir / "gifs" / f"step_{step}.gif", episodes[0].gif_frames)
        rows.append(summarize(f"{step:,}", [e.outcome for e in episodes]))
        entries.append(
            LeaderboardEntry(
                row=rows[-1], step=step, screenshots=shots, gif=f"gifs/step_{step}.gif"
            )
        )
        steps.append(step)
        print(f"step={step} score={rows[-1].score_mean:.1f} rows_max={rows[-1].max_rows_max}")
    env.close()

    write_csv(run_dir / "leaderboard.csv", rows)
    report = (
        f"# Leaderboard: {run_dir.name}\n\n"
        f"{args.episodes} greedy episodes per checkpoint (epsilon={EVAL_EPSILON}, "
        f"seeds {EVAL_SEED_BASE}+i), random policy as baseline.\n\n"
        + render_markdown(rows)
        + legend_markdown()
    )
    (run_dir / "leaderboard.md").write_text(report, encoding="utf-8")
    document = leaderboard_document(
        algorithm="DQN", env_id=ENV_ID, run_id=run_dir.name, entries=entries
    )
    (run_dir / "leaderboard.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    plot_medals(run_dir / "milestones.png", steps, rows[1:])
    print(f"wrote {run_dir / 'leaderboard.md'}")


if __name__ == "__main__":
    main()

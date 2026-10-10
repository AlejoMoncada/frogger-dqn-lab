"""Calibrate milestone rows on ALE/Frogger-v5 with the 300k DQN checkpoint.

Run from the project root: uv run python results/calibration/calibrate.py
Writes row_*.png and rows_montage.png next to this file and prints the measurements
used in docs/notas-de-desarrollo.md, section 4.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dqn_frogger.agent import DQNAgent  # noqa: E402
from dqn_frogger.env import make_env  # noqa: E402

HERE = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "results" / "dqn_frogger.pt"
EPISODES = 5
EPISODE_SEED = 1_000
RAM_ROWS = 90  # hops reached in the current life; resets when a life is lost
RAM_SCORE_BCD = 78  # cumulative score, binary-coded decimal
FROG_RGB = np.array([110, 156, 66])
ROW0_CENTER_Y = 173.8  # frog centroid on the start strip
ROW_PIXELS = 12.97  # vertical distance between consecutive rows
ROW_NAMES = {
    0: "start strip",
    **{r: "road lane" for r in range(1, 6)},
    6: "median (road crossed)",
    **{r: "river" for r in range(7, 12)},
    7: "first river row",
    9: "middle river row",
    12: "home row",
}
TRACKED_ROWS = (0, 1, 6, 7, 9, 12)


def frog_row(frame: np.ndarray) -> tuple[int, int] | None:
    """Return (row, pixel count) of the frog sprite, or None if it is not visible."""
    mask = np.all(frame == FROG_RGB, axis=-1)
    ys, _ = np.nonzero(mask)
    if len(ys) == 0:
        return None
    row = round((ROW0_CENTER_Y - ys.mean()) / ROW_PIXELS)
    return int(row), len(ys)


def bcd_to_int(value: int) -> int:
    return (value >> 4) * 10 + (value & 15)


def play(agent: DQNAgent, env, seed: int) -> dict[str, list]:
    """Greedy episode recording RAM (before each step), reward, lives, frame and frog row."""
    rec: dict[str, list] = {"ram": [], "reward": [], "lives": [], "frame": []}
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    done = False
    while not done:
        rec["ram"].append(env.unwrapped.ale.getRAM().copy())
        action = agent.act(obs, 0.01, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        rec["reward"].append(float(reward))
        rec["lives"].append(int(info["lives"]))
        rec["frame"].append(env.render())
        done = terminated or truncated
    return rec


def verify_bytes(rec: dict[str, list]) -> None:
    """Check that RAM byte 90 counts hops in the life and byte 78 holds the BCD score."""
    ram = np.array(rec["ram"], dtype=int)
    reward = np.array(rec["reward"])
    lives = np.array(rec["lives"])
    rows, score, prev_lives = 0, 0.0, 4
    row_match = bcd_match = total = 0
    for t in range(len(reward) - 1):
        if reward[t] > 0:
            rows += 1
        score += reward[t]
        if lives[t] < prev_lives:
            rows = 0
        prev_lives = lives[t]
        total += 1
        row_match += int(ram[t + 1, RAM_ROWS] == rows)
        bcd_match += int(bcd_to_int(ram[t + 1, RAM_SCORE_BCD]) % 100 == int(score) % 100)
    print(f"byte {RAM_ROWS} equals hops-in-life: {row_match}/{total} steps")
    print(f"byte {RAM_SCORE_BCD} equals BCD score: {bcd_match}/{total} steps")


def first_frames_by_row(rec: dict[str, list]) -> dict[int, tuple[int, np.ndarray]]:
    """First frame (step, image) in which the frog is detected on each row."""
    seen: dict[int, tuple[int, np.ndarray]] = {}
    for t, frame in enumerate(rec["frame"]):
        found = frog_row(frame)
        if found and found[1] >= 20:
            seen.setdefault(found[0], (t, frame))
    return seen


def save_row_images(frames: dict[int, tuple[int, np.ndarray]]) -> None:
    rows = [r for r in TRACKED_ROWS if r in frames]
    for r in rows:
        step, frame = frames[r]
        plt.imsave(HERE / f"row_{r:02d}.png", frame)
    fig, axes = plt.subplots(1, len(rows), figsize=(3 * len(rows), 4.6))
    for ax, r in zip(np.atleast_1d(axes), rows):
        step, frame = frames[r]
        ax.imshow(frame)
        ax.set_title(f"row {r}: {ROW_NAMES[r]}\nstep {step}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(HERE / "rows_montage.png", dpi=110)
    plt.close(fig)


def main() -> None:
    env = make_env(EPISODE_SEED, render_mode="rgb_array")
    agent = DQNAgent(obs_dim=int(env.observation_space.shape[0]), n_actions=int(env.action_space.n))
    agent.load(CHECKPOINT)

    all_frames: dict[int, tuple[int, np.ndarray]] = {}
    for i in range(EPISODES):
        rec = play(agent, env, EPISODE_SEED + i)
        print(f"episode {i}: steps={len(rec['reward'])} score={sum(rec['reward']):.0f}")
        verify_bytes(rec)
        for row, value in first_frames_by_row(rec).items():
            all_frames.setdefault(row, value)
    env.close()

    print("first frame per row (row: step):", {r: all_frames[r][0] for r in sorted(all_frames)})
    save_row_images(all_frames)


if __name__ == "__main__":
    main()

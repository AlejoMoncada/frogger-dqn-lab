# Feature: dqn-frogger

## Objective
Deliver Taller 2 (U3, due Monday 2026-10-12 23:59): a public GitHub repo with a from-scratch
DQN (Gymnasium + PyTorch, no Stable-Baselines3) that learns `ALE/Frogger-v5`, plus a README.

## Problem / Why
The existing `frogger-rl-lab` repo is the PPO (SB3) deliverable. Mixing both assignments in one
repo would confuse graders, so DQN lives in this separate, independent project.

## Scope / Constraints
- Environment: `ALE/Frogger-v5` with `obs_type="ram"` (128-byte vector) -> MLP Q-network.
  Pixel/CNN DQN is out of scope: millions of frames do not fit the CPU deadline.
- Own DQN: replay buffer, epsilon-greedy, target network, Bellman/TD update, Huber loss.
- Dependencies: gymnasium, ale-py==0.12.1, torch (CPU index), numpy, matplotlib, pytest. No SB3.
- Code/comments/identifiers in English; README in neutral Spanish (course language).
- Do not touch the PPO repo.

## Acceptance criteria
- `uv run pytest` green (buffer, network shapes, epsilon schedule, Bellman target, smoke train).
- Training script produces metrics CSV, checkpoint and learning-curve PNG.
- README covers: observation/action spaces, rewards and termination, training flow with a
  diagram, network architecture, numeric results + curve, reflection on the hardest part.

## Tasks
- [x] T1 Scaffold + DQN implementation + tests (route: delegated writer, Haiku; 2+ non-trivial files)
- [ ] T2 Full training run on CPU, collect metrics/plot (route: inline background command)
- [ ] T3 README with results and diagram (route: delegated writer, Haiku)
- [ ] T4 Create public GitHub repo and push (user decision; not authorized yet)

## Delivery strategy
single-pr not applicable (new repo); work-unit commits on `main` after the initial scaffold.

## Progress / Evidence
- T1 done. Commits on `main`: `a3351d0` chore: scaffold uv project; `b66344e` feat: implement dqn agent for frogger ram; `37441bb` feat: add train, evaluate and plot entry points.
- RED: tests for buffer, schedule and agent written first; `pytest` failed at collection (`dqn_frogger` missing).
- `uv sync`: OK (torch 2.14.1+cpu, gymnasium 1.4.0, ale-py 0.12.1, Python 3.11.9). `uv.lock` committed.
- `uv run pytest -q`: `14 passed in 5.66s`.
- `uv run python train.py --smoke --output-dir runs/smoke`: exit 0, 7.3 s. Tail: `[eval] step=2000 mean_return=6.0`, `done: {'total_steps': 2000, 'episodes': 7, 'last_eval_mean_return': 6.0, ...}`. `episodes.csv` has 7 rows, `eval.csv` 2 rows.
- `uv run python plot.py --input-dir runs/smoke`: `wrote runs\smoke\learning_curve.png`.
- Extra sanity: `evaluate.py --checkpoint runs/smoke/dqn_frogger.pt --episodes 2` ran: DQN mean 1.0, random mean 10.5 (smoke model is untrained, so this is expected).
- Deviation: pytest needs `pythonpath = ["."]` because the uv project has no build-system; the package is not installed.

## Next step
T2 (full CPU training run; not started).

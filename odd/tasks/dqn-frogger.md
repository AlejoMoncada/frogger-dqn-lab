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
- [x] T2 Baseline 300k-step CPU run (route: inline background). 6m21s, ~790 steps/s; greedy eval 25.2±4.4 vs random 9.4±2.1.
- [x] T2b Milestone/medal system: periodic checkpoints, per-life row progress logging, milestone
      leaderboard per checkpoint, screenshots on first medal, GIF per checkpoint (route: delegated writer, Haiku).
      Home detection is NOT verified (fallback: row 12 counts as a home); see the evidence below.
- [ ] T2c Long run with checkpoints (2M-5M steps) + leaderboard + CPU vs GPU timing (route: inline background)
- [ ] T3 README with results, medals table, screenshots and diagram (route: delegated writer, Haiku)
- [ ] T4 Create public GitHub repo and push (user decision; not authorized yet)
- [x] T5 `milestones.py` also exports `leaderboard.json` with an algorithm-agnostic schema
      (algorithm, run_id, checkpoint step, score mean/std, max_rows, medal percentages, screenshot/gif paths)

### Follow-ups after the DQN deliverable (not this feature's scope)
- PPO repo (`frogger-rl-lab`, formative, already delivered): new branch adding SB3 CheckpointCallback,
  a copied milestone tracker, the same `leaderboard.json` export, and a timing run (CPU vs GTX 1660S).
- Dashboard reading both `leaderboard.json` files to compare DQN vs PPO medals per step.
- Thesis seed: checkpoints as a difficulty ladder for an RL-based tutor (cf. Maia Chess, curriculum learning).

### Milestone findings (probe of 300k agent, 3 episodes)
- Reward is always +1 per forward hop that reaches a new row; no other reward values observed.
- Agent makes ~6-7 hops per life then dies (likely crosses the road, drowns entering the river).
- Lane counts per milestone must be calibrated with rendered frames before being fixed.

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

### T2b evidence (milestone system)
- Commits on `feat/milestone-leaderboard`: `01f4a48` feat: add milestone tracker with medals; `5c73e33` feat: save periodic checkpoints and timing; `e614c04` feat: add milestone leaderboard with screenshots and gifs; `d22da6f` feat: overlay max rows per life on learning curve; `2c63e71` chore: move baseline results; `e7caa21` docs: record milestone calibration; `fbfead4` feat: export algorithm-agnostic leaderboard json.
- Calibration (`docs/notas-de-desarrollo.md` section 4, `results/calibration/`): byte 90 = rows in current life and byte 78 = BCD score, both verified on every step of 5 episodes. Road = rows 1-5, median = row 6, river = rows 7-11, home row = 12. Greedy 300k agent reached at most row 8.
- Home detection: unverified. A beam search over emulator states (beam width 200, 450 depth levels) reached row 11 but not home. Fallback in use: reaching row 12 counts as one home and resets the row count. `frog_home`, `three_homes` and `level_cleared` are therefore not verified.
- Tests: `uv run pytest -q` -> `33 passed in 7.02s`.
- `uv run python train.py --smoke --output-dir runs/smoke_ms`: exit OK, `done: {'total_steps': 2000, 'episodes': 7, ...}`, `timing.json` written.
- `uv run python milestones.py --run-dir runs/smoke_ms --episodes 2`: wrote `leaderboard.md`, `leaderboard.csv`, `leaderboard.json`, `milestones.png`, `screenshots/` and 3 GIFs (`random.gif`, `step_1000.gif`, `step_2000.gif`).
- Route: delegated direct (single writer, Haiku). Verification was run by the writer; no separate verifier.

## Next step
T2c (long run with checkpoints, 2M-5M steps) and the home-detection follow-up.

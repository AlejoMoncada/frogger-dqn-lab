# Feature: home-challenge

## Objective
Personal challenge (NOT part of the graded deliverable on `main`): break the row-10 ceiling of the
RAM+MLP DQN and get frogs home in `ALE/Frogger-v5`, using a mixed CPU/GPU design where each device
does what it is best at.

## Problem / Why
- RAM+MLP DQN plateaued at max row 10 from 250k to 2M steps (`results/run_2m/leaderboard.md`).
- Hypotheses (unverified): single RAM snapshot lacks velocity/timing info; ε floor 0.05 and sticky
  actions punish the river; home reward unknown.
- Measured: for the tiny MLP the GPU is 23% slower than CPU (`results/benchmark_cpu_gpu.json`), so
  the GPU only pays off with a heavier model and bigger batches.

## Design
- CPU: `ale_py.vector_env.AtariVectorEnv` (C++ threads) runs N Frogger envs in parallel and does
  preprocessing (84x84 grayscale, maxpool, 4-frame stack). Replay buffer in host RAM as uint8.
- GPU: Nature-DQN CNN; batched action selection for all envs; learner with Double DQN, n-step returns.
- Hardware: Ryzen 5 3600 (6c/12t), 31.9 GB RAM, GTX 1660 SUPER 6 GB, CUDA torch cu128.

## Constraints
- Branch `exp/home-challenge` only (worktree `frogger-dqn-lab-home`); never touch `main`.
- Keep the existing RAM+MLP pipeline working; new code lives beside it.
- Results must feed the same medal leaderboard / `leaderboard.json` schema (algorithm label distinct).
- Code/comments in English; notes in neutral Spanish.

## Tasks
- [x] H1 Pixel CNN DQN with AtariVectorEnv, Double DQN, n-step, CUDA support + tests (route: delegated writer, Haiku)
- [ ] H2 Pilot measurement: 100k steps mixed (CPU envs + GPU learner) vs all-CPU; steps/s, GPU util (route: inline background)
- [ ] H3 Decide budget from pilot; long run with checkpoints; leaderboard + GIFs (route: inline background)
- [ ] H4 Verify home detection with a real trajectory if any frog reaches row 12 (route: delegated)
- [ ] H5 Compare RAM+MLP vs pixels+CNN in one table; notes section (route: delegated writer)

## Acceptance criteria
- Pilot reports measured steps/s for both configurations.
- Any home claim is backed by a frame/GIF and the reward/lives trace.

## Progress / Evidence
- H1 done (branch `exp/home-challenge`):
  - `d33eb3b` build: torch cu128 index. `torch 2.11.0+cu128`, `cuda.is_available()=True`,
    device `NVIDIA GeForce GTX 1660 SUPER`.
  - `88c8ff2` feat(pixels): CNN, uint8 n-step replay, Double DQN agent. Tests written first (RED =
    collection errors, modules missing), then GREEN.
  - `ba87ff6` feat(pixels): AtariVectorEnv trainer and `train_pixels.py` CLI.
  - `bd358e2` feat(milestones): pixel checkpoints (GIFs, screenshots, `DQN-pixels (Double, n-step)`).
    Adds `opencv-python-headless` (required by gymnasium `AtariPreprocessing`).
  - `uv run pytest -q`: 70 passed (54 RAM + 16 new).
  - Smoke CUDA (`runs/px_smoke_cuda`): 2 000 env steps, 202.3 steps/s; time breakdown env 0.56 s,
    action 0.79 s, learner 3.22 s, eval 5.21 s.
  - Smoke CPU (`runs/px_smoke_cpu`): 2 000 env steps, 104.1 steps/s; env 0.50 s, action 0.88 s,
    learner 12.37 s, eval 5.35 s.
  - `milestones.py --run-dir runs/px_smoke_cuda --episodes 1` wrote leaderboard.json/md/csv,
    milestones.png and GIFs for random, 1 000 and 2 000.
  - AtariVectorEnv findings: `docs/notas-de-desarrollo.md`, section 7.
- Deviations: the replay stores obs and next_obs explicitly, so 200 000 transitions need ~11.3 GB
  of RAM, not ~5.6 GB. Playback (GIFs) uses a gym env with `AtariPreprocessing`, not the vector
  env, because the vector env has no RGB render and a gym env does not reproduce its trajectories.

## Next step
H2: pilot 100k steps, mixed (CPU envs + GPU learner) vs all-CPU, to measure steps/s and GPU use.

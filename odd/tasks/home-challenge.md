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
- [x] H2 Pilot measurement: 100k steps mixed (CPU envs + GPU learner) vs all-CPU; steps/s, GPU util (route: inline background)
- [x] H2b Harden runs: memory guard, periodic checkpoints, resume (route: delegated writer, Haiku)
- [ ] H3 Decide budget from pilot; long run with checkpoints; leaderboard + GIFs (route: inline background)
- [x] H4 Verify home detection with a real trajectory if any frog reaches row 12 (route: delegated; evidence: H6 trace, see Progress)
- [ ] H5 Compare RAM+MLP vs pixels+CNN in one table; notes section (route: delegated writer)
- [x] H6 Reward scaling option (home = +55 verified) (route: delegated writer, Haiku)
- [ ] H7 Run 3: buffer 200k, reward_mode=scale 0.1 (home +5.5, row +0.1), 5M steps (route: inline background)
- [x] H9 Frame-sharing replay, 8x less memory: one stored frame per env step, about 7.1 KB instead of 56.5 KB per transition (route: delegated writer)
- [x] H10 Colab overkill notebook: `colab/frogger_overkill.ipynb` + `colab/README.md`, CLI flags for batch, threads, target interval, warmup, epsilon decay and lr (route: delegated writer)
- [x] H11 Apple MPS support + Mac guide: `--device {cpu,cuda,mps,auto}`, device-aware sync and pinning, platform-conditional torch source, `docs/mac.md`. Real MPS execution is unverified until the user's 100k pilot (route: delegated writer, Haiku)
- [x] H12 Life loss as terminal (episodic life for TD only): `--life-loss-terminal` / `life_loss_terminal`; a lost life ends the bootstrap, the game goes on (route: delegated direct, writer)
- [x] H13 Run 5: 200k, sign, life_loss_terminal, 5M (route: inline background)
- [x] H14 Interactive story page (local, docs/historia): `tools/build_story.py` writes `docs/historia/data.js` and copies 220 referenced GIFs/screenshots (all present); `uv run pytest -q` 152 passed (4 new); headless Chrome: no console errors, no horizontal overflow at 400px, light and dark screenshots checked (route: delegated writer)

- [x] H8 Run 4: buffer 200k, reward_mode=sign, 5M steps (isolates buffer vs reward with run 3) — chained after run 3

### Ablation grid (pixels + CNN + Double DQN + n-step 3, 5M steps, 1 seed each)
| Run | Buffer | Reward | Status |
|---|---|---|---|
| 1 | 200k | sign | killed by OOM at 4.6M; home in 28% of train episodes at 4-4.6M |
| 2 | 100k | sign | done: max row 10, 0 homes in eval, final 38.1 |
| 3 | 200k | scale 0.1 | done (resumed at 2.5M with frame replay): 5M eval 78.2, frog_home 60% |
| 4 | 200k | sign | done: 3M frog_home 85%, 5M eval 101.2, frog_home 90%; 550.9 steps/s |
| 3b | 200k | scale 0.1, seed 43 | done on MacBook Pro M4 (MPS, 379.5 steps/s): 5M eval 36.4, frog_home 0%, max row 10 |
| 5 | 200k | sign + life_loss_terminal | done: 5M eval 122.2, frog_home 90%, homes mean/max 1.35/3, three_homes 5% (first ever) |

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

- H2 pilot (100k env steps, 8 envs, same config/seed):
  - Mixed (CPU envs + GPU CNN): 164.7 s, 607.0 steps/s; env 21.4 s, action 14.9 s, learner 119.7 s.
    GPU util mean 45.6% (max 56%) over 86 samples; GPU memory max 5431 MiB (whole card).
  - All-CPU: 814.6 s, 122.8 steps/s; env 20.7 s, action 30.4 s, learner 755.5 s.
  - Mixed is 4.9x faster; the learner is 73% of mixed time, so the loop is learner-bound and
    sequential (GPU ~46% busy). Eval at 100k: 10.0 in both (5 episodes; too early to compare learning).

- Incident (killed 5M run `results/run_px_5m`): stopped at 4.6M env steps by an out-of-memory
  condition. Replay at 200 000 transitions needs ~11.3 GB, plus 2-3 GB of torch/CUDA memory, while
  other applications already used ~18 GB of 32 GB. Checkpoints existed only at 100k, 250k, 500k,
  1M, 2M and 3M, so the weights from 3M to 4.6M were lost. The partial run stays untouched.
- H2b hardening (`exp/home-challenge`):
  - `6ddbc66` build(deps): `psutil`.
  - `4eeb352` feat(pixels): memory guard (estimate + 3 GB must fit in `psutil` available RAM, else
    refuse unless `--force-memory`); default replay 100 000; `--buffer-size`; periodic
    `checkpoints/step_N.pt` plus atomic `latest.pt` every `--save-interval` (default 250 000) with
    networks, optimizer, counters, epsilon position, RNG states and config; `--resume`; save on
    KeyboardInterrupt. Named checkpoints use the same bundle, and `PixelDQNAgent.load` still reads
    the online-only files.
  - `5028343` fix(milestones): missing checkpoints listed in `config.json` are skipped with a
    warning; `latest.pt` is never a leaderboard row.
  - `uv run pytest -q`: 88 passed (70 before, 18 new). RED observed first for every new test.
  - Smoke CUDA (`runs/px_safe_smoke`, `--save-interval 1000`): 2 000 steps, 170.8 steps/s;
    checkpoints `step_1000.pt`, `step_2000.pt` and `latest.pt` written.
  - Resume `--resume --total-steps 3000` then `6000`: log `resumed_from_step=2000` (then 3000);
    episodes 5-8 appended after the restored counter; eval rows appended; `timing.json` has
    `resumed_from_step` and `session_steps`.
  - Memory printout: `memory: replay estimate 0.06 GB + margin 3.0 GB = 3.06 GB required;
    available now 13.78 GB`. `--buffer-size 400000` is refused with 22.59 GB + 3.0 GB > 13.86 GB,
    and no directory is created.
- Deviations: a bundle is 27 MB on disk (online, target and Adam state), not ~7 MB. The replay buffer is
  not checkpointed, so after a resume the learner waits `learning_starts` steps again. Partial
  episodes at the resume point are dropped and env seeds are offset by the resume step.

- H6 reward scaling (`exp/home-challenge`):
  - Verified trace (ALE Frogger, step 231): reward 55 on the hop from row 11 to home, lives 3 to 3,
    RAM row byte 11 to 0, BCD score 20 to 75. Rows give +1.
  - `afa8613` feat(pixels): `reward_mode` (`sign` default, `scale`, `raw`) and `reward_scale` (0.1)
    replace `reward_clip`; `transform_reward` is applied before n-step accumulation; episode
    returns stay raw; CLI `--reward-mode` and `--reward-scale` written to config.json. Legacy
    `reward_clip` maps True to `sign` and False to `raw` (`upgrade_saved_config`), so resuming
    `results/run_px_5m_safe` keeps `sign` with no spurious change warning.
  - Tests: `tests/test_pixels_reward.py`, 8 new. RED observed first (collection ImportError for
    the missing symbols), then GREEN. `uv run pytest -q`: 96 passed (88 before + 8).
  - Smoke CPU (`runs/reward_smoke`, `--reward-mode scale --reward-scale 0.1`): config.json has
    `"reward_mode": "scale"`, `"reward_scale": 0.1`; 2 000 steps, 37.8 steps/s.

- H9 frame-sharing replay (`exp/home-challenge`):
  - `replay_impl` "frame" (default) or "stacked" (`--replay-impl`). `FrameReplay` in
    `dqn_frogger/pixels/frame_replay.py`. Stack semantics measured on AtariVectorEnv: the reset stack
    is `[0, 0, 0, f]`, and each step's first three channels equal the previous step's last three
    (4 488 of 4 488 checks). Frames are stored once, with zero fill before the episode start.
  - Tests (`tests/test_pixels_frame_replay.py`, 14 new): same scripted stream into both layouts
    (4 parameter sets, one with wraparound); real AtariVectorEnv stream (3 envs, 400 steps, with
    episode ends); eviction never samples overwritten transitions; zero fill never reads the previous
    episode; trainer stores one row per real step (autoreset skipped); config records `replay_impl`.
    RED: collection ImportError for the missing symbols. Mutation check: disabling zero fill or frame
    eviction fails the tests. `uv run pytest -q`: 110 passed (96 before + 14).
  - Bytes per transition: 7 105 (one 7 056-byte frame plus 49 index bytes) vs 56 468 stacked, 8x less.
    200 000 transitions: 1.42 GB of arrays; guard `replay estimate 1.42 GB + margin 3.0 GB`.
  - Smoke CUDA `runs/frame_smoke` (`--buffer-size 200000`): 209.9 steps/s (earlier 170-200).
  - RSS (psutil): 0.51 GB before allocation, 1.94 GB after filling 199 058 transitions with 8 envs.
    Deviation: the smoke fills about 2k transitions only, so RSS comes from a synthetic fill.
  - Frame slots per env = capacity / envs + n_step + 8. Each episode start uses a frame slot without a
    transition, so with short episodes some oldest transitions are evicted early (942 of 200k in the
    synthetic run). Evicted transitions are never sampled.
  - Resume: `replay_impl` is a resume-free field; saved configs without it resume unchanged.
- H10 Colab overkill notebook (`exp/home-challenge`):
  - `dba9ad9` feat(pixels): flags `--batch-size`, `--num-threads`, `--target-update-interval`,
    `--learning-starts`, `--eps-decay-steps`, `--lr`. They were already dataclass fields, so
    `trainer.py` and `config.py` are unchanged and existing checkpoints and resume are unaffected.
    Defaults are unchanged. Test `tests/test_pixels_cli.py` (3 tests). No RED run was observed: the tests were written after the flags.
  - `uv run pytest -q`: 113 passed.
  - CPU smoke `runs/cli_smoke` with `--batch-size 64 --num-threads 2`: config.json has
    `batch_size: 64`, `num_threads: 2`, `lr: 0.0001` (default, flag omitted), `learning_starts: 200`.
  - `36136c2` feat(colab): notebook with 14 cells (title and warnings, hardware report and CONFIG,
    Drive, clone and pip install with CUDA check, smoke, training with auto `--resume`, leaderboard
    and plots, results). `nbformat.validate` passes (nbformat 4.5). The notebook was not executed.
  - Deviation: `total_steps` is rounded down to a multiple of `num_envs`, since the trainer requires it
    (20 M is not divisible by 12, 24 or 28).
- H11 Apple MPS support + Mac guide (`exp/home-challenge`, route: delegated writer):
  - Lock: `pyproject.toml` torch source is `marker = "sys_platform != 'darwin'"` on `pytorch-cu128`.
    `uv lock` splits torch: Windows/Linux `2.11.0+cu128` (cu128 index); darwin `2.14.1` from PyPI with
    `macosx_14_0_arm64` wheels for cp311, cp312 and cp313. `ale-py==0.12.1` has `macosx_13_0_arm64`
    wheels for cp311, cp312 and cp313. Deviation: the Mac torch version (2.14.1) differs from Windows
    (2.11.0); uv picked the latest PyPI release for darwin.
  - Windows check (read-only, no sync run): `uv sync --dry-run` gives "Would make no changes";
    `.venv` python reports `torch 2.11.0+cu128`, `cuda.is_available() = True`.
  - Device: `dqn_frogger/pixels/device.py` (`resolve_device`, `default_device`, `pin_memory_enabled`,
    `sync_device`, `host_info`). `auto` = cuda, then mps, then cpu. Unavailable devices raise a clear
    error (`SystemExit` in the CLI). `train_pixels.py` and root `milestones.py` use it.
  - Pinning: both replays pin only when CUDA is available; the trainer passes pin only for CUDA.
    `torch.cuda.synchronize` is replaced by `sync_device` (`torch.mps.synchronize` on MPS, no-op on CPU).
  - Checkpoints: loads already use `map_location` (`agent.load` to its device, trainer restore to CPU
    then moves to the device). Covered by a test.
  - `config.json` and `timing.json` record `device` and `host` (system, machine, processor, macOS version).
  - Tests: `tests/test_pixels_device.py`, 25 new (resolve per name with monkeypatched availability,
    auto order, errors; pin gating in both replays; sync dispatch; checkpoint `map_location` cpu;
    host and device recorded). Not RED-first: the code was written before the tests.
  - `uv run --no-sync pytest -q`: 138 passed. `uv run --no-sync python train_pixels.py --smoke --device cpu
    --output-dir runs/mps_cpu_smoke`: done, `timing.json` device `cpu`, host `AMD64`.
  - Unverified: real MPS execution, MPS speed and MPS numeric behavior. These wait for the user's 100k pilot.
  - Guide: `docs/mac.md`.
- H12 Life loss as terminal (`exp/home-challenge`, route: delegated direct, writer):
  - Motivation: run 4 at 5M loses 4 of 5 frogs in the river before its single home per game; death has
    no direct cost in the TD target. Trace (6 seeds): 4 deaths in rows 7-10 at steps about 137, 180, 225
    and 265; then 1 home at step about 376-383 with reward 51-53; no lives left for a second home.
  - Behavior: `PixelDQNConfig.life_loss_terminal` (default False, recorded in `config.json`, missing key
    reads as False). Per env, `lives_prev` comes from the reset info and is refreshed on every real step
    and on the NextStep reset step. A life lost on a step (not terminated, not truncated) is stored with
    `done=True` and flushes that env's n-step window as terminal. The game keeps going: episode return,
    length, milestones and `episodes.csv` are unchanged. The next transition starts a new window from the
    post-death observation. `--resume` keeps a saved true; the flag cannot turn a saved true off.
  - Finding (real `AtariVectorEnv`, ale-py 0.12.1, `episodic_life=False`): the frame stack is NOT reset
    or zero-filled on a life loss. After each of 3 losses (steps 125, 162, 210) the new stack's first
    three channels equal the previous stack's last three. So `FrameReplay` bookkeeping needs no change:
    only the n-step done flag changes.
  - Tests: `tests/test_pixels_life_loss.py`, 10 new. RED: 9 failed before the change (the new field did not
    exist). GREEN: all 10 pass. Covered: done and window flush per life, flag-off unchanged, one episode row
    per full game, episodes.csv identical with and without the flag, lives tracking across the autoreset,
    both replay layouts, config round-trip and missing key, resume keeps the saved flag, CLI flag.
  - `uv run pytest -q`: 148 passed (baseline 138).
  - Smoke: `uv run python train_pixels.py --smoke --device cuda --life-loss-terminal --output-dir
    runs/life_smoke`: `config.json` has `"life_loss_terminal": true`; `episodes.csv` has 4 rows, episode ids
    1-4, one per game.
  - Hypothesis (unverified until H13): treating a lost life as terminal makes each frog worth its own TD
    target, which should reduce risky river crossings and allow more than one home per game.
  - Evaluation, MilestoneTracker and leaderboard are untouched (full 5-life games).
  - Commits: `dcbdee6` (feat pixels, code and tests), `5874dc5` (feat colab preset key).

- H4 home detection verified (`exp/home-challenge`, route: delegated writer):
  - Evidence: the H6 step-by-step trace of the 1M checkpoint (seed 900008): hop from row 11 to home at
    step 231, reward +55, lives 3 to 3, RAM row byte 11 to 0, BCD score 20 to 75.
  - Tracker check (`dqn_frogger/milestones.py`, `MilestoneTracker.observe`): a positive reward increments
    the row count; at `HOME_ROW` (12) with no life lost on that step it counts one home and resets the
    count. The trace matches this rule.
  - Page: `tools/build_story.py` no longer propagates `proxy_medals` and writes
    `home_detection: {status: "verified"}` to `data.js`; the page drops the proxy tags and the H4 note.

## Next step
H3: choose the long-run budget (user decision). Long runs now use `--save-interval`; resume a
killed run with `--resume` on the same `--output-dir`.

### Ablation reading (2026-10-11, 1 seed each, preliminary)
- Buffer size is the main factor: 100k + sign (run 2) never reached home; 200k + sign (run 4) reached home in 90% of greedy eval episodes at 5M.
- Reward scaling did not help on top of the 200k buffer: run 3 (scale) 60% vs run 4 (sign) 90% at 5M; within single-seed noise, not a confirmed effect.
- No run reached three homes; the max is 1 home per eval episode.
- Pending: seed-43 replica of run 3 on the Mac to estimate run-to-run variance.

### Updated reading (after runs 3b and 5)
- Seed fragility: run 3 (seed 42) 60% home vs run 3b (seed 43) 0% with the same config. 200k buffer enables but does not guarantee reaching home.
- Life loss as terminal (run 5): episodes 696 vs 422 steps (run 4, 4-5M window); 535/1434 games with 2+ homes vs 0/2363; first three-home game in eval (5%). Supports the reflection hypothesis; single seed.
- Open question: run 3's jump came right after a resume that emptied the replay.
- README extension on main updated in commit 49ef13c.

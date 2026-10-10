**No severe issues found in the supplied source.** The DQN target, termination-only bootstrap mask, target synchronization, epsilon schedule, replay sampling with replacement, clipped training rewards versus raw logged returns, and tensor device transfers appear correct.

### Medium severity

1. **Reusing an output directory mixes checkpoints from different runs.**  
   **Locations:** `dqn_frogger/trainer.py:79-82,103-104,162-163`; `milestones.py:85-92`  
   **Failure:** Run normal training, then smoke training into the same directory. CSVs and final weights are overwritten, but old periodic checkpoints remain. The leaderboard evaluates both runs’ checkpoints as one training progression.  
   **Fix:** Reject nonempty run directories unless explicitly resuming, or create a unique directory per run. Discover checkpoints from a run manifest rather than every matching filename.

2. **Home medals present an unverified proxy as a confirmed achievement.**  
   **Locations:** `dqn_frogger/milestones.py:67-75`; `milestones.py:140-145`  
   **Failure:** Eleven rewarding steps followed by a twelfth reward with decreased lives still awards `frog_home` before processing the death. This demonstrates that the tracker cannot distinguish a successful home from a row-12 arrival accompanied by death; whether that exact sequence occurs in ALE needs calibration. Consequently, `three_homes` and `level_cleared` are also estimates—not verified game events.  
   **Fix:** Validate home/level detection against RAM and recorded trajectories. Until then, explicitly label these metrics as proxies in Markdown and JSON. Do not assume that merely checking for unchanged lives fully validates home detection.

3. **Saved artifacts do not contain enough information to reproduce a run.**  
   **Locations:** `dqn_frogger/trainer.py:64-72,175-186`; `dqn_frogger/agent.py:79-81`; `pyproject.toml:8-13`  
   **Failure:** Training with different seeds or epsilon horizons produces weights and metrics without saving the actual configuration. Dependency specifications also permit version changes. Additionally, `--seed 900000` overlaps the leaderboard’s first evaluation seed (`train.py:43-44`; `milestones.py:41,95-98`), so seed separation holds for the defaults, not every supported configuration.  
   **Fix:** Save the full configuration, code revision, environment options, dependency versions, and evaluation seed manifest. Use a frozen dependency lock and check evaluation seeds against the recorded training/validation seeds.

4. **Smoke tests do not verify critical trainer behavior or generated artifacts.**  
   **Locations:** `tests/test_trainer.py:11-31,34-58`; `tests/test_milestones.py:58-74`; `tests/test_leaderboard.py:69-100`  
   **Failure:** Removing periodic target synchronization or changing the trainer to mask truncations could still satisfy the existing assertions. Synthetic home tests encode the fallback assumption rather than validate it. JSON tests check path strings but never exercise screenshot/GIF creation.  
   **Fix:** Add a scripted environment test verifying:
   - termination masks versus truncation bootstrapping, including the pre-reset next observation;
   - actual updates and target synchronization boundaries;
   - clipped replay rewards versus raw CSV returns;
   - first-achievement screenshots and readable GIFs with expected frames/timing.

   Separately, validate milestone semantics using real RAM/reward/lives traces.

### Low severity

5. **Trainer evaluation scores depend on previous evaluation history.**  
   **Locations:** `dqn_frogger/trainer.py:91,151-156`  
   **Failure:** Environment seeds restart each evaluation, but `eval_rng` continues advancing. With nonzero evaluation epsilon, an identical checkpoint can receive different exploration decisions depending on earlier evaluation lengths. Changing `eval_interval` can therefore change the final reported score without changing the training trajectory. This does **not** mean identical complete runs are necessarily nondeterministic.  
   **Fix:** Use reproducible policy RNG seeds per evaluation episode, independent of previous evaluations, and record them.

6. **JSON silently omits `homes_max`.**  
   **Locations:** `dqn_frogger/leaderboard.py:40-41,123-128`  
   **Failure:** CSV and Markdown expose mean/max homes, but JSON consumers cannot reconstruct that same statistic because the maximum is dropped.  
   **Fix:** Serialize `homes_max` and test the complete intended schema. The existing percentage-to-fraction conversion and generated relative artifact paths are consistent.

Static inspection only; tests were not executed. Required memory persistence was unavailable.

## Key Learnings:
1. Truncations correctly reset episodes without suppressing bootstrapping in this implementation.
2. Reward-derived home counts require independent validation before being reported as confirmed achievements.

## Resolution

| # | Finding | Status | What changed | Commit |
|---|---------|--------|--------------|--------|
| 1 | Output-dir reuse mixes checkpoints | Accepted | `train()` refuses a directory with `episodes.csv` or a non-empty `checkpoints/`; `--overwrite` clears only the known run artifacts. `milestones.py` reads checkpoints from the `config.json` manifest, with a glob fallback when it is absent. | `ead6fd8` |
| 2 | Home medals are an unverified proxy | Partially accepted | A life lost on the same step as the twelfth row no longer counts as a home. `frog_home`, `three_homes` and `level_cleared` are marked as proxies (`PROXY_MEDALS`, `proxy_medals` in `leaderboard.json`, `*` and a footnote in `leaderboard.md`). Validation against RAM traces is not done; it remains open in the ODD task. | `524639a`, `5723030` |
| 3 | Artifacts do not allow reproduction | Partially accepted | `config.json` is written at training start (full config, env id and options, library versions, `git rev-parse HEAD` or null, checkpoint manifest, eval seed range). Training seeds inside `[900000, 901000)` are rejected by `dqn_frogger/seeds.py`, shared with the leaderboard. Dependency pins were not changed: `uv.lock` was already tracked and pins exact versions. | `cfe2e08` |
| 4 | Smoke tests miss critical behavior | Accepted | Scripted-env tests cover truncation versus termination, pre-reset next observation, target sync at interval boundaries, clipped replay rewards versus raw returns, and PNG/GIF readback on a fake episode. Validation of milestone semantics against real RAM traces is not covered by these tests. | `28e6637` |
| 5 | Evaluation depends on history | Accepted | Each evaluation episode gets `default_rng(episode seed)` for the policy, in trainer, `evaluate.py` and `milestones.py`. `eval.csv` records `first_seed` and `last_seed`. | `7b4d375` |
| 6 | JSON omits `homes_max` | Accepted | `homes_max` is serialized per row, and a schema test checks the full set of top-level and row keys. | `5723030` |

Verification: `uv run pytest -q` passes (50 tests); the smoke reuse refusal and `--overwrite` path were run on `runs/smoke_fix`.
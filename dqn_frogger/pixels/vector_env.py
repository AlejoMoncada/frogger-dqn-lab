"""Factories for the pixel environments, plus the AtariVectorEnv behaviour the trainer relies on.

Findings from probing ale-py 0.12.1 (ale_py.vector_env.AtariVectorEnv), measured on Windows:

- Construction: `AtariVectorEnv(game="frogger", num_envs=N, num_threads=T, img_height=84,
  img_width=84, grayscale=True, stack_num=4, frameskip=4, maxpool=True,
  repeat_action_probability=0.25, episodic_life=False, reward_clipping=False)`. The ROM name is
  "frogger" (the gym id is ALE/Frogger-v5).
- Observations: `uint8` array of shape (N, 4, 84, 84), stacked grayscale frames.
- Actions: `Discrete(5)` per env. The gym RAM env `ALE/Frogger-v5` also reports `Discrete(5)`.
- Reset: `reset(seed=np.array([...]))` needs an int or one seed per env (a list raises TypeError).
  Reset info holds `lives`, `frame_number`, `episode_frame_number` and `env_id`. Lives start at 4.
- Termination: the terminal step returns `terminated=True` with `lives == 0`. This is the true
  terminal observation.
- Truncation: `max_num_frames_per_episode` (default 108000) gives `truncated=True`. Checked with
  a 40-frame limit: truncation arrived on the expected step.
- Autoreset `NextStep` (default, used here): the step AFTER a done returns the reset observation
  with reward 0 and terminated/truncated False. The action passed on that step is ignored, so
  the step is not a real transition and the trainer skips it.
- Autoreset `SameStep`: the reset observation is returned on the terminal step itself. The
  terminal observation is only exposed as `info["final_obs"]` (seen on truncation). Not used.
- `autoreset_mode="Disabled"` is rejected by ale-py 0.12.1.
- `reset(options={"reset_mask": ...})` supports partial resets. Not used here.
- No RGB rendering: the vector env has no `render()`. Screenshots and GIFs therefore come from a
  single gym env (see `make_playback_env`).
- The gym env and the vector env are not trajectory-identical for the same seed (their
  no-op and fire reset differ, and the sticky-action RNG is consumed differently). Measured:
  rewards, terminal steps and lives diverged within 400 steps on seeds 5 and 6. Playback
  therefore cannot run in lockstep with the vector env; see `make_playback_env`.
"""

import gymnasium as gym
import ale_py
from ale_py.vector_env import AtariVectorEnv
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

from dqn_frogger.env import ENV_ID

ALE_GAME = "frogger"
FRAME_SKIP = 4
REPEAT_ACTION_PROBABILITY = 0.25
NOOP_MAX = 30


def make_vector_env(num_envs: int, num_threads: int = 8) -> AtariVectorEnv:
    """Frogger vector env with the Nature-DQN preprocessing and NextStep autoreset."""
    return AtariVectorEnv(
        game=ALE_GAME,
        num_envs=num_envs,
        num_threads=num_threads,
        img_height=84,
        img_width=84,
        grayscale=True,
        stack_num=4,
        frameskip=FRAME_SKIP,
        maxpool=True,
        repeat_action_probability=REPEAT_ACTION_PROBABILITY,
        noop_max=NOOP_MAX,
        episodic_life=False,
        reward_clipping=False,
        autoreset_mode="NextStep",
    )


def make_playback_env(seed: int) -> gym.Env:
    """Single gym env with RGB rendering and the same frame pipeline shape (4, 84, 84) uint8.

    Used by milestones.py for screenshots and GIFs. Its preprocessing is gymnasium's
    AtariPreprocessing, not the C++ vector pipeline, so the stacked observations are close to
    but not bit-identical with training. It is the only env in which the played frames and the
    policy's inputs come from the same emulator run.
    """
    gym.register_envs(ale_py)
    env = gym.make(
        ENV_ID,
        render_mode="rgb_array",
        frameskip=1,
        repeat_action_probability=REPEAT_ACTION_PROBABILITY,
    )
    env = AtariPreprocessing(
        env,
        noop_max=NOOP_MAX,
        frame_skip=FRAME_SKIP,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    env.action_space.seed(seed)
    return env

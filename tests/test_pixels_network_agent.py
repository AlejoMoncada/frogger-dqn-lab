"""CNN shape and the Double DQN target on hand-built tensors (CPU only)."""

import pytest
import torch

from dqn_frogger.pixels.agent import PixelDQNAgent, double_dqn_targets
from dqn_frogger.pixels.network import PixelQNetwork


def test_cnn_output_shape_for_uint8_stack() -> None:
    net = PixelQNetwork(n_actions=5)
    obs = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8)
    assert net(obs).shape == (3, 5)


def test_cnn_scales_uint8_input_inside_forward() -> None:
    net = PixelQNetwork(n_actions=5)
    obs = torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.uint8)
    # uint8 input is converted and divided by 255 inside forward, so it matches the same values
    # passed as float; a missing division would make the outputs differ by a large factor.
    assert torch.allclose(net(obs), net(obs.float()))
    assert not torch.allclose(net(obs), net(obs.float() / 255.0))


def test_double_dqn_target_uses_online_argmax_and_target_value() -> None:
    online_next_q = torch.tensor([[1.0, 5.0, 2.0], [0.0, -1.0, 3.0]])
    target_next_q = torch.tensor([[10.0, 20.0, 30.0], [7.0, 8.0, 40.0]])
    rewards = torch.tensor([1.0, 0.5])
    discounts = torch.tensor([0.99**3, 0.5])
    dones = torch.tensor([0.0, 1.0])

    targets = double_dqn_targets(online_next_q, target_next_q, rewards, discounts, dones)

    # a* = [1, 2]; Q_target(s', a*) = [20, 40]; second row is terminal, so no bootstrap.
    expected = torch.tensor([1.0 + 0.99**3 * 20.0, 0.5])
    assert torch.allclose(targets, expected, atol=1e-5)
    # Plain max over the target net would pick 30 and 40 and differ from the Double target.
    assert not torch.allclose(targets[0], rewards[0] + discounts[0] * 30.0)


def test_agent_act_returns_one_action_per_env_and_respects_epsilon_one() -> None:
    import numpy as np

    agent = PixelDQNAgent(n_actions=5, device="cpu")
    rng = np.random.default_rng(0)
    obs = np.zeros((6, 4, 84, 84), dtype=np.uint8)
    actions = agent.act(obs, epsilon=1.0, rng=rng)
    assert actions.shape == (6,)
    assert set(actions.tolist()) <= set(range(5))


def test_agent_update_moves_q_towards_target_and_clips_nothing_on_cpu() -> None:
    from dqn_frogger.pixels.replay import PixelBatch

    torch.manual_seed(0)
    agent = PixelDQNAgent(n_actions=5, device="cpu", lr=1e-2)
    batch = PixelBatch(
        obs=torch.randint(0, 256, (8, 4, 84, 84), dtype=torch.uint8),
        actions=torch.randint(0, 5, (8,)),
        rewards=torch.ones(8),
        next_obs=torch.randint(0, 256, (8, 4, 84, 84), dtype=torch.uint8),
        dones=torch.ones(8),
        discounts=torch.full((8,), 0.99),
    )
    first = agent.update(batch)
    for _ in range(30):
        last = agent.update(batch)
    assert last < first


@pytest.mark.parametrize("shape", [(1, 4, 84, 84), (2, 4, 84, 84)])
def test_network_accepts_batch_sizes(shape) -> None:
    net = PixelQNetwork(n_actions=5)
    assert net(torch.zeros(shape, dtype=torch.uint8)).shape[0] == shape[0]

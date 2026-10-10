"""Tests for the QNetwork and the DQN agent (TD target, update, save/load)."""

import numpy as np
import torch

from dqn_frogger.agent import DQNAgent
from dqn_frogger.network import QNetwork
from dqn_frogger.replay_buffer import ReplayBuffer


def test_qnetwork_output_shape() -> None:
    net = QNetwork(obs_dim=128, n_actions=18)
    out = net(torch.zeros(5, 128))
    assert out.shape == (5, 18)


def _constant_target_agent(n_actions: int = 3) -> DQNAgent:
    """Agent whose target net outputs the constant Q-values [1, 3, 2] for every state."""
    agent = DQNAgent(obs_dim=4, n_actions=n_actions, hidden=(8,), lr=1e-3, gamma=0.5)
    with torch.no_grad():
        for p in agent.target_net.parameters():
            p.zero_()
        agent.target_net.net[-1].bias.copy_(torch.tensor([1.0, 3.0, 2.0]))
        # Online net deliberately different: it must NOT influence the target.
        for p in agent.online_net.parameters():
            p.zero_()
        agent.online_net.net[-1].bias.copy_(torch.tensor([0.0, 0.0, 100.0]))
    return agent


def test_td_target_uses_target_net_and_masks_done() -> None:
    agent = _constant_target_agent()
    rewards = torch.tensor([1.0, 2.0])
    next_obs = torch.zeros(2, 4)
    dones = torch.tensor([0.0, 1.0])

    targets = agent.compute_td_target(rewards, next_obs, dones, gamma=0.5)

    # Not done: 1 + 0.5 * max(1, 3, 2) = 2.5 ; done: bootstrap masked -> 2.0
    assert torch.allclose(targets, torch.tensor([2.5, 2.0]))


def test_update_reduces_loss_on_fixed_batch() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    agent = DQNAgent(obs_dim=4, n_actions=2, hidden=(16,), lr=1e-2, gamma=0.9)
    buffer = ReplayBuffer(capacity=32, obs_dim=4)
    for _ in range(32):
        buffer.add(
            rng.standard_normal(4).astype(np.float32),
            int(rng.integers(2)),
            float(rng.standard_normal()),
            rng.standard_normal(4).astype(np.float32),
            done=bool(rng.integers(2)),
        )
    batch = buffer.sample(batch_size=32, rng=rng)

    first = agent.update(batch)
    for _ in range(30):
        last = agent.update(batch)
    assert last < first


def test_save_and_load_round_trip(tmp_path) -> None:
    torch.manual_seed(0)
    path = tmp_path / "model.pt"
    original = DQNAgent(obs_dim=4, n_actions=2, hidden=(8,))
    original.save(path)

    restored = DQNAgent(obs_dim=4, n_actions=2, hidden=(8,))
    restored.load(path)
    x = torch.ones(1, 4)
    assert torch.allclose(original.online_net(x), restored.online_net(x))
    assert torch.allclose(restored.online_net(x), restored.target_net(x))

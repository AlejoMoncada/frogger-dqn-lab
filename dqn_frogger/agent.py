"""DQN agent: online and target Q-networks, epsilon-greedy acting, and the TD update."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dqn_frogger.network import QNetwork
from dqn_frogger.replay_buffer import Batch

GRAD_NORM_CLIP = 10.0


class DQNAgent:
    """Holds the online and target networks and performs the Bellman update."""

    def __init__(
        self,
        obs_dim: int = 128,
        n_actions: int = 18,
        hidden: Sequence[int] = (256, 256),
        lr: float = 1e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = torch.device(device)
        self.online_net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.sync_target()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

    def act(self, obs: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        """Epsilon-greedy action selection over the online network."""
        if rng.random() < epsilon:
            return int(rng.integers(self.n_actions))
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online_net(obs_t).argmax(dim=1).item())

    def compute_td_target(
        self,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Bellman target r + gamma * (1 - done) * max_a' Q_target(s', a'). No gradient."""
        with torch.no_grad():
            next_q_max = self.target_net(next_obs).max(dim=1).values
            return rewards + gamma * (1.0 - dones) * next_q_max

    def update(self, batch: Batch) -> float:
        """One gradient step on the Huber loss between Q(s, a) and the TD target."""
        obs = batch.obs.to(self.device)
        actions = batch.actions.to(self.device)
        targets = self.compute_td_target(
            batch.rewards.to(self.device),
            batch.next_obs.to(self.device),
            batch.dones.to(self.device),
            self.gamma,
        )
        q_taken = self.online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), GRAD_NORM_CLIP)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        """Hard-copy the online weights into the target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: Path | str) -> None:
        """Save the online network weights."""
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: Path | str) -> None:
        """Load weights into both the online and the target network."""
        state = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(state)
        self.sync_target()

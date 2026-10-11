"""Double DQN learner for pixel observations, with batched epsilon-greedy acting on any device."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from dqn_frogger.pixels.device import pin_memory_enabled
from dqn_frogger.pixels.network import PixelQNetwork
from dqn_frogger.pixels.replay import PixelBatch

GRAD_NORM_CLIP = 10.0
ADAM_EPS = 1.5e-4


def double_dqn_targets(
    online_next_q: torch.Tensor,
    target_next_q: torch.Tensor,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    dones: torch.Tensor,
) -> torch.Tensor:
    """y = R + discount * (1 - done) * Q_target(s', a*), with a* = argmax_a Q_online(s', a).

    `discounts` is gamma^n for full windows (gamma^m for flushed ones). No gradient.
    """
    with torch.no_grad():
        best_actions = online_next_q.argmax(dim=1, keepdim=True)
        next_value = target_next_q.gather(1, best_actions).squeeze(1)
        return rewards + discounts * (1.0 - dones) * next_value


class PixelDQNAgent:
    """Online and target CNNs, Adam, Huber loss, hard target sync."""

    def __init__(
        self,
        n_actions: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = torch.device(device)
        self.online_net = PixelQNetwork(n_actions).to(self.device)
        self.target_net = PixelQNetwork(n_actions).to(self.device)
        self.sync_target()
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr, eps=ADAM_EPS)

    @torch.no_grad()
    def act(self, obs_batch: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
        """Epsilon-greedy actions for a batch of stacked observations (N, 4, 84, 84)."""
        n = len(obs_batch)
        obs_t = torch.as_tensor(obs_batch, device=self.device)
        greedy = self.online_net(obs_t).argmax(dim=1).cpu().numpy()
        explore = rng.random(n) < epsilon
        random_actions = rng.integers(self.n_actions, size=n)
        return np.where(explore, random_actions, greedy).astype(np.int64)

    def update(self, batch: PixelBatch) -> float:
        """One gradient step; returns the Huber loss as a Python float."""
        pin = pin_memory_enabled(self.device)
        obs = batch.obs.to(self.device, non_blocking=pin)
        next_obs = batch.next_obs.to(self.device, non_blocking=pin)
        actions = batch.actions.to(self.device, non_blocking=pin)
        with torch.no_grad():
            online_next_q = self.online_net(next_obs)
            target_next_q = self.target_net(next_obs)
        targets = double_dqn_targets(
            online_next_q,
            target_next_q,
            batch.rewards.to(self.device, non_blocking=pin),
            batch.discounts.to(self.device, non_blocking=pin),
            batch.dones.to(self.device, non_blocking=pin),
        )
        q_taken = self.online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), GRAD_NORM_CLIP)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: Path | str) -> None:
        """Online weights only (the final model file, read by milestones)."""
        torch.save(self.online_net.state_dict(), path)

    def resumable_state(self) -> dict:
        """Online and target networks plus the optimizer: everything a resume needs."""
        return {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def restore_resumable_state(self, state: dict) -> None:
        self.online_net.load_state_dict(state["online_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])

    def load(self, path: Path | str) -> None:
        """Accepts both the online-only file and a resumable checkpoint bundle."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "online_net" in state:
            state = state["online_net"]
        self.online_net.load_state_dict(state)
        self.sync_target()

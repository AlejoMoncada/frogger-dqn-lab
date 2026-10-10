"""Q-network: an MLP mapping a RAM observation to one Q-value per action."""

from collections.abc import Sequence

import torch
from torch import nn


class QNetwork(nn.Module):
    """Fully connected network with ReLU activations and a linear output layer."""

    def __init__(
        self,
        obs_dim: int = 128,
        n_actions: int = 18,
        hidden: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for width in hidden:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return Q-values of shape (batch, n_actions)."""
        return self.net(obs)

"""Nature-DQN convolutional Q-network over a stack of 4 grayscale 84x84 frames."""

import torch
from torch import nn

FRAME_STACK = 4
FRAME_SIZE = 84


class PixelQNetwork(nn.Module):
    """Conv 32x8x8/s4, 64x4x4/s2, 64x3x3/s1, fc 512, linear output (one Q-value per action)."""

    def __init__(self, n_actions: int, in_channels: int = FRAME_STACK) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Accepts uint8 (N, 4, 84, 84) frames, scales them to [0, 1] and returns (N, n_actions)."""
        return self.net(obs.float().div(255.0))

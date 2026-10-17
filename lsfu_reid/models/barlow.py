"""Barlow Twins warmup head (Zbontar et al.)."""

from __future__ import annotations

import torch
import torch.nn as nn


class BarlowTwinsHead(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden: int = 2048, out_dim: int = 256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


def barlow_loss(z1: torch.Tensor, z2: torch.Tensor, lambd: float = 5e-3) -> torch.Tensor:
    # normalize along batch
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
    c = (z1.T @ z2) / z1.size(0)
    on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
    off_diag = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()
    return on_diag + lambd * off_diag

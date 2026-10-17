"""Backbones for LSFU-ReID (paper: ResNet50 / DenseNet121 / OSNet)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):
    """CNN backbone + BNNeck + L2-normalized embedding (GlobalAvgPool)."""

    def __init__(self, name: str = "resnet50", feat_dim: int = 2048, pretrained: bool = True):
        super().__init__()
        self.name = name.lower()
        weights = "DEFAULT" if pretrained else None
        if self.name == "resnet50":
            net = models.resnet50(weights=weights)
            self.backbone = nn.Sequential(*list(net.children())[:-1])
            in_dim = 2048
        elif self.name == "densenet121":
            net = models.densenet121(weights=weights)
            self.backbone = net.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            in_dim = 1024
        else:
            raise ValueError(f"Unsupported backbone: {name} (OSNet omitted; see DEVIATIONS.md)")
        self.feat_dim = int(feat_dim)
        if in_dim != self.feat_dim:
            self.fc = nn.Linear(in_dim, self.feat_dim, bias=False)
            nn.init.normal_(self.fc.weight, std=0.001)
        else:
            self.fc = nn.Identity()
            self.feat_dim = in_dim
        # BNNeck (Luo et al.): stabilizes ID / proxy CE for ReID fine-tuning
        self.bnneck = nn.BatchNorm1d(self.feat_dim)
        self.bnneck.bias.requires_grad_(False)
        nn.init.constant_(self.bnneck.weight, 1.0)
        nn.init.constant_(self.bnneck.bias, 0.0)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if self.name == "densenet121":
            x = self.backbone(x)
            x = F.relu(x, inplace=True)
            x = self.pool(x)
        else:
            x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bnneck(x)
        if normalize:
            x = F.normalize(x, p=2, dim=1)
        return x


class BarlowTwins(nn.Module):
    """Barlow Twins wrapper (paper Supp.)."""

    def __init__(self, backbone: EmbeddingNet, projector_dim: int = 2048):
        super().__init__()
        self.backbone = backbone
        dim = backbone.feat_dim
        self.projector = nn.Sequential(
            nn.Linear(dim, projector_dim, bias=False),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_dim, projector_dim, bias=False),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_dim, projector_dim, bias=False),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1 = self.projector(self.backbone(x1, normalize=False))
        z2 = self.projector(self.backbone(x2, normalize=False))
        return z1, z2


def build_backbone(cfg: dict, name: Optional[str] = None) -> EmbeddingNet:
    mcfg = cfg.get("model", {})
    return EmbeddingNet(
        name=name or mcfg.get("backbone", "resnet50"),
        feat_dim=int(mcfg.get("feat_dim", 2048)),
        pretrained=bool(mcfg.get("pretrained", True)),
    )

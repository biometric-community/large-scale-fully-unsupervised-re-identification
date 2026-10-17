"""Losses: Barlow Twins + L_proxy + L_hard (paper Eqs. 9–11 / Supp.)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambd: float = 5e-3) -> torch.Tensor:
    """Barlow Twins redundancy-reduction loss (Supp. Eq. 1)."""
    n, d = z1.shape
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
    c = (z1.T @ z2) / n
    on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
    off_diag = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()
    return on_diag + lambd * off_diag


def proxy_loss(
    feats: torch.Tensor,
    labels: torch.Tensor,
    proxies: torch.Tensor,
    proxy_labels: torch.Tensor,
    tau: float = 0.04,
) -> torch.Tensor:
    """Eq. 9: InfoNCE vs class proxies (L2-normalized)."""
    # feats: (B, D), proxies: (C, D)
    logits = (feats @ proxies.t()) / tau  # (B, C)
    # map label -> proxy column
    # assume proxy_labels are 0..C-1 matching rows
    target = labels.clone()
    return F.cross_entropy(logits, target)


def hard_contrastive_loss(
    feats: torch.Tensor,
    labels: torch.Tensor,
    tau: float = 0.04,
) -> torch.Tensor:
    """Eq. 10: hard-positive (furthest same ID) + hard-negative (closest other)."""
    B = feats.size(0)
    if B < 2:
        return feats.new_zeros(())
    sim = feats @ feats.t()  # (B, B)
    labels = labels.view(-1, 1)
    same = labels.eq(labels.t())
    diff = ~same
    eye = torch.eye(B, dtype=torch.bool, device=feats.device)
    same = same & ~eye
    losses = []
    for i in range(B):
        pos_mask = same[i]
        neg_mask = diff[i]
        if not pos_mask.any() or not neg_mask.any():
            continue
        # furthest positive
        pos_sims = sim[i].masked_fill(~pos_mask, 2.0)  # high = close; want min sim among pos
        v_pos_idx = pos_sims.argmin()
        # closest negative
        neg_sims = sim[i].masked_fill(~neg_mask, -2.0)
        v_neg_idx = neg_sims.argmax()
        s_pos = sim[i, v_pos_idx] / tau
        s_neg = sim[i, v_neg_idx] / tau
        losses.append(-torch.log(torch.exp(s_pos) / (torch.exp(s_pos) + torch.exp(s_neg))))
    if not losses:
        return feats.new_zeros(())
    return torch.stack(losses).mean()


class FinalLoss(nn.Module):
    """L_final = L_proxy + λ L_hard (Eq. 11)."""

    def __init__(self, tau: float = 0.04, lambd: float = 0.5):
        super().__init__()
        self.tau = float(tau)
        self.lambd = float(lambd)

    def forward(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        proxies: torch.Tensor,
        proxy_labels: torch.Tensor,
    ):
        # filter noise
        mask = labels >= 0
        if mask.sum() < 2:
            z = feats.new_zeros(())
            return {"loss": z, "loss_proxy": z, "loss_hard": z}
        feats = feats[mask]
        labels = labels[mask]
        # remap labels to proxy indices if needed
        # proxies rows correspond to proxy_labels values
        label_to_row = {int(l): i for i, l in enumerate(proxy_labels.tolist())}
        mapped = torch.tensor([label_to_row[int(l)] for l in labels.tolist()], device=feats.device)
        lp = proxy_loss(feats, mapped, proxies, proxy_labels, self.tau)
        lh = hard_contrastive_loss(feats, labels, self.tau)
        loss = lp + self.lambd * lh
        return {"loss": loss, "loss_proxy": lp.detach(), "loss_hard": lh.detach()}


def select_proxies(
    features: torch.Tensor,
    labels: torch.Tensor,
    mode: str = "centroid",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-cluster proxies (paper Sec. III-F).

    `centroid` (default): L2-normalized mean feature — more stable than a random
    exemplar when cluster purity is moderate. `random` matches the paper text.
    """
    uniq = sorted({int(x) for x in labels.tolist() if int(x) >= 0})
    proxies = []
    plabels = []
    for c in uniq:
        idxs = (labels == c).nonzero(as_tuple=False).view(-1)
        if mode == "random":
            pick = idxs[torch.randint(0, idxs.numel(), (1,))]
            proxies.append(features[pick])
        else:
            cen = features[idxs].mean(dim=0, keepdim=True)
            proxies.append(F.normalize(cen, p=2, dim=1))
        plabels.append(c)
    if not proxies:
        return features.new_zeros((0, features.size(1))), labels.new_zeros((0,), dtype=torch.long)
    return torch.cat(proxies, 0), torch.tensor(plabels, device=features.device, dtype=torch.long)

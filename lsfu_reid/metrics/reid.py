"""ReID metrics: CMC + mAP (Market-1501 protocol). Copied pattern from CTU/CapReID."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_rank(
    query_feats: torch.Tensor,
    query_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_feats: torch.Tensor,
    gallery_pids: np.ndarray,
    gallery_camids: np.ndarray,
    ranks: List[int] = (1, 5, 10),
) -> Dict[str, float]:
    q = F.normalize(query_feats, dim=1)
    g = F.normalize(gallery_feats, dim=1)
    dist = 1.0 - torch.mm(q, g.t())
    dist_np = dist.cpu().numpy()
    num_q = dist_np.shape[0]
    all_cmc = []
    all_ap = []
    g_pids = np.asarray(gallery_pids)
    g_cams = np.asarray(gallery_camids)
    q_pids = np.asarray(query_pids)
    q_cams = np.asarray(query_camids)
    for i in range(num_q):
        order = np.argsort(dist_np[i], kind="mergesort")
        pid = q_pids[i]
        cam = q_cams[i]
        keep = ~((g_pids[order] == pid) & (g_cams[order] == cam))
        matches = g_pids[order][keep] == pid
        if not np.any(matches):
            continue
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc)
        tmp_cmc = matches.cumsum().astype(np.float64)
        tmp_cmc = tmp_cmc / np.arange(1, len(tmp_cmc) + 1)
        ap = (tmp_cmc * matches).sum() / matches.sum()
        all_ap.append(ap)
    if not all_cmc:
        return {"mAP": 0.0, **{f"Rank-{r}": 0.0 for r in ranks}, "cmc": []}
    max_len = max(len(c) for c in all_cmc)
    cmc_mat = np.zeros((len(all_cmc), max_len))
    for i, c in enumerate(all_cmc):
        cmc_mat[i, : len(c)] = c
        if len(c) < max_len:
            cmc_mat[i, len(c) :] = c[-1]
    cmc_mean = cmc_mat.mean(axis=0)
    out = {"mAP": float(np.mean(all_ap)) * 100.0, "cmc": (cmc_mean * 100.0).tolist()}
    for r in ranks:
        out[f"Rank-{r}"] = float(cmc_mean[r - 1]) * 100.0
    return out

"""Noise-robust ε scheduling + DBSCAN (paper Sec. III-D / Fig. 2)."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN


def nr_epsilon(
    epoch: int,
    total_epochs: int,
    eps_min: float = 0.4,
    eps_max: float = 0.6,
) -> float:
    """Cosine warmup to half epochs → anneal to 75% → plateau.

    Paper Fig. 2 / Sec. III-D: start low, increase to mid-training peak,
    decrease until 75% of epochs, then hold constant.
    """
    epoch = int(epoch)
    T = max(int(total_epochs), 1)
    half = T // 2
    three_q = int(0.75 * T)
    lo, hi = float(eps_min), float(eps_max)
    if epoch < half:
        t = epoch / max(half, 1)
        # cosine rise lo → hi
        return lo + 0.5 * (hi - lo) * (1.0 - math.cos(math.pi * t))
    if epoch < three_q:
        t = (epoch - half) / max(three_q - half, 1)
        # cosine fall hi → lo
        return hi - 0.5 * (hi - lo) * (1.0 - math.cos(math.pi * t))
    return lo


def cluster_dbscan(
    distance_matrix: np.ndarray,
    eps: float,
    min_samples: int = 4,
) -> np.ndarray:
    """DBSCAN on precomputed distances; noise labeled -1 (paper)."""
    # sklearn requires zeros on diagonal for precomputed
    D = np.asarray(distance_matrix, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    labels = DBSCAN(
        eps=float(eps),
        min_samples=int(min_samples),
        metric="precomputed",
        n_jobs=-1,
    ).fit_predict(D)
    return labels.astype(np.int64)


def remapped_cluster_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Map cluster ids to contiguous [0, C); keep -1 as noise."""
    out = labels.copy()
    valid = sorted({int(x) for x in labels.tolist() if int(x) >= 0})
    mapping = {old: i for i, old in enumerate(valid)}
    for i, lab in enumerate(labels):
        out[i] = mapping[int(lab)] if int(lab) >= 0 else -1
    return out, len(valid)


def filter_monocamera_clusters(
    labels: np.ndarray,
    camids: np.ndarray,
    min_cams: int = 2,
) -> np.ndarray:
    """Mark clusters that span fewer than `min_cams` cameras as noise.

    Market eval removes same-camera matches; mono-camera pseudo-labels teach
    camera style and collapse cross-camera mAP (empirically cam_dom≈0.78).
    """
    out = np.asarray(labels, dtype=np.int64).copy()
    cams = np.asarray(camids, dtype=np.int64)
    if min_cams <= 1 or out.size == 0:
        return out
    for c in set(out.tolist()) - {-1}:
        m = out == int(c)
        if int(np.unique(cams[m]).size) < int(min_cams):
            out[m] = -1
    return out

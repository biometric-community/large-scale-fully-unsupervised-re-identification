"""NR-ε schedule + DBSCAN clustering + simple kNN Jaccard affinity."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def nr_epsilon(epoch: int, total_epochs: int, eps_max: float = 0.6, eps_min: float = 0.4) -> float:
    """Noise-robust ε: cosine warmup for first half, then plateau (paper Fig. 2)."""
    half = max(total_epochs // 2, 1)
    if epoch <= half:
        # cosine from eps_max → eps_min over warmup half
        t = (epoch - 1) / max(half - 1, 1)
        return eps_max + 0.5 * (eps_min - eps_max) * (1 - math.cos(math.pi * t))
    return eps_min


def knn_jaccard_distance(
    feats: np.ndarray, k: int = 20
) -> np.ndarray:
    """Approximate re-ranked distances via Jaccard over kNN sets (LRR-style O(kn))."""
    n = feats.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean", algorithm="brute")
    nn.fit(feats)
    inds = nn.kneighbors(return_distance=False)  # [n, k+1] includes self
    # drop self
    nbrs = [set(row[1:].tolist()) for row in inds]
    # build sparse-ish dense Jaccard distance for DBSCAN (n x n too big if n large —
    # for Market ~13k use only for clustering via precomputed on subsample OR
    # use euclidean on features which is standard for unsupervised ReID DBSCAN)
    # Here we return euclidean distances (caller may use metric=precomputed with this).
    # For memory, return None and let DBSCAN use euclidean on feats directly.
    return feats  # placeholder identity — clustering uses euclidean in cluster_features


def cluster_features(
    feats: torch.Tensor,
    eps: float,
    min_samples: int = 4,
) -> np.ndarray:
    """DBSCAN on L2 features; returns labels with -1 = noise."""
    x = feats.detach().cpu().numpy().astype(np.float64)
    # Paper uses density in re-ranked space; we cluster L2 features with cosine distance.
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    labels = clustering.fit_predict(x)
    return labels



def remapped_labels(labels: np.ndarray) -> Tuple[np.ndarray, int]:
    """Map cluster ids to [0, C), keep -1 as -1."""
    uniq = sorted(set(labels.tolist()) - {-1})
    mapping = {c: i for i, c in enumerate(uniq)}
    out = np.array([mapping.get(int(c), -1) for c in labels], dtype=np.int64)
    return out, len(uniq)

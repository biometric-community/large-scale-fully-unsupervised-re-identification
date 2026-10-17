"""Local Neighborhood Sampling (paper Sec. III-B)."""

from __future__ import annotations

import numpy as np


def local_neighborhood_sample(
    features: np.ndarray,
    p: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Return indices of top-p% neighbors of a random seed vector (cosine).

    For Market (~13k) paper reports at p=100% (full set). When p>=1.0, return all.
    """
    n = features.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    p = float(p)
    if p >= 1.0:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    v_idx = int(rng.integers(0, n))
    feats = features.astype(np.float32, copy=False)
    # L2-normalized cosine similarity → distance
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    feats = feats / norms
    sims = feats @ feats[v_idx]
    order = np.argsort(-sims)  # descending similarity
    keep = max(1, int(round(p * n)))
    return order[:keep].astype(np.int64)

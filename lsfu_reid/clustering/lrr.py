"""Local Re-Ranking (paper Sec. III-C, Eqs. 1–8). Memory O(kn) for R; dense for DBSCAN."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def local_re_ranking(
    features: np.ndarray,
    k: int = 20,
    return_full_distance: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute LRR refined distances (Eqs. 1–8).

    Uses sklearn NearestNeighbors (no FAISS). Builds dense n×n distances for
    sklearn DBSCAN on Market-scale sets (~13k); non-neighbors set to 1.0.
    """
    n = int(features.shape[0])
    if n <= 1:
        full = np.zeros((n, n), dtype=np.float64)
        return np.zeros((n, 0), dtype=np.int64), full

    k = int(min(max(k, 1), n - 1))
    feats = np.asarray(features, dtype=np.float32)
    # Cosine on L2-normalized embeddings (ReID default); distance in [0, 2].
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(feats)
    dist, idx = nn.kneighbors(feats)
    idx = idx[:, 1 : k + 1].astype(np.int64)
    dist = dist[:, 1 : k + 1].astype(np.float64)
    d_loc = np.exp(-dist)  # (n, k)

    # Membership matrix M[i, j]=1 if j in N(i,k) — sparse via sets for IoU
    # Precompute neighbor sets as sorted arrays for faster intersection
    neigh = [set(row.tolist()) for row in idx]

    # dloc lookup table as (n, n) sparse via dict of arrays: value only for neighbors
    # Vectorized path: for each i,j pair in knn, compute IoU using boolean masks
    R = np.ones((n, k), dtype=np.float64)

    # Build (n, n) dloc sparse matrix filled only at knn positions
    dloc_full = np.zeros((n, n), dtype=np.float64)
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    dloc_full[rows, cols] = d_loc.reshape(-1)

    for i in range(n):
        ni = neigh[i]
        for t in range(k):
            j = int(idx[i, t])
            nj = neigh[j]
            inter = ni & nj
            if inter:
                inter_l = list(inter)
                di = dloc_full[i, inter_l]
                dj = dloc_full[j, inter_l]
                s_min = float(np.minimum(di, dj).sum())
                s_max = float(np.maximum(di, dj).sum())
            else:
                s_min = 0.0
                s_max = 0.0
            e_i = ni - inter
            e_j = nj - inter
            si_j = float(dloc_full[i, list(e_i)].sum()) if e_i else 0.0
            sj_i = float(dloc_full[j, list(e_j)].sum()) if e_j else 0.0
            denom = s_max + si_j + sj_i
            d_iou = (s_min / denom) if denom > 0 else 0.0
            # Reciprocal: full Jaccard credit. Non-reciprocal: half credit so
            # DBSCAN can still chain early-training neighborhoods (eps~0.6–0.75).
            if i in nj:
                R[i, t] = 1.0 - d_iou
            else:
                R[i, t] = 1.0 - 0.5 * d_iou

    if not return_full_distance:
        return idx, R

    full = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(full, 0.0)
    for i in range(n):
        for t in range(k):
            j = int(idx[i, t])
            v = R[i, t]
            if v < full[i, j]:
                full[i, j] = v
            if v < full[j, i]:
                full[j, i] = v
    return idx, full


def jaccard_affinity_fallback(
    features: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Simplified shared-kNN Jaccard distances (document vs full LRR/FRR)."""
    n = int(features.shape[0])
    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)
    k = int(min(max(k, 1), n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(features)
    _, idx = nn.kneighbors(features)
    idx = idx[:, 1:]
    sets = [set(row.tolist()) for row in idx]
    full = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(full, 0.0)
    for i in range(n):
        for j in idx[i]:
            j = int(j)
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            d = 1.0 - (inter / union if union else 0.0)
            if d < full[i, j]:
                full[i, j] = d
            if d < full[j, i]:
                full[j, i] = d
    return full

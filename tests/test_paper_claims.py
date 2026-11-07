"""Clustering, reproduced with this repo's LRR and DBSCAN on synthetic identities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score

from lsfu_reid.clustering.dbscan_nr import cluster_dbscan
from lsfu_reid.clustering.lrr import local_re_ranking


def test_refined_distances_recover_identities():
    rng = np.random.RandomState(0)
    centers = rng.normal(size=(6, 32))
    labels = np.repeat(np.arange(6), 10)
    feats = centers[labels] + 0.3 * rng.normal(size=(60, 32))
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    _, distance = local_re_ranking(feats, k=8)
    found = cluster_dbscan(distance, eps=0.7, min_samples=4)
    assert adjusted_rand_score(labels, found) >= 0.9

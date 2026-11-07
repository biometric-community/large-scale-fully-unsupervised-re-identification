"""LRR, LNS, the epsilon schedule, losses and protocol, checked against the paper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from lsfu_reid.clustering.dbscan_nr import nr_epsilon
from lsfu_reid.clustering.lns import local_neighborhood_sample
from lsfu_reid.clustering.lrr import local_re_ranking
from lsfu_reid.data.market1501 import Market1501
from lsfu_reid.losses import barlow_twins_loss, hard_contrastive_loss, proxy_loss
from lsfu_reid.metrics import evaluate_rank

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def lrr():
    rng = np.random.RandomState(0)
    centers = rng.normal(size=(4, 8))
    feats = np.repeat(centers, 5, axis=0) + 0.6 * rng.normal(size=(20, 8))
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    idx, refined = local_re_ranking(feats, k=4, return_full_distance=False)
    return feats, idx, refined


def eq_7(feats: np.ndarray, idx: np.ndarray, i: int, j: int) -> float:
    d_loc = np.exp(-(1.0 - feats @ feats.T))
    n_i, n_j = set(idx[i].tolist()), set(idx[j].tolist())
    common = sorted(n_i & n_j)
    s_min = sum(min(d_loc[i, p], d_loc[j, p]) for p in common)
    s_max = sum(max(d_loc[i, p], d_loc[j, p]) for p in common)
    s_ij = sum(d_loc[i, p] for p in n_i - n_j)
    s_ji = sum(d_loc[j, p] for p in n_j - n_i)
    denominator = s_max + s_ij + s_ji
    return s_min / denominator if denominator > 0 else 0.0


def test_reciprocal_neighbours_follow_eq_7(lrr):
    feats, idx, refined = lrr
    pairs = [(i, t) for i in range(len(idx)) for t in range(idx.shape[1]) if i in idx[idx[i, t]]]
    assert pairs
    for i, t in pairs:
        assert refined[i, t] == pytest.approx(1.0 - eq_7(feats, idx, i, int(idx[i, t])), abs=1e-5)


def test_non_reciprocal_neighbours_get_distance_one(lrr):
    """Eq. (8): R is 1 unless xi is also a neighbour of xj."""
    _, idx, refined = lrr
    pairs = [(i, t) for i in range(len(idx)) for t in range(idx.shape[1]) if i not in idx[idx[i, t]]]
    assert pairs
    assert all(refined[i, t] == 1.0 for i, t in pairs)


def test_refined_distance_keeps_k_values_per_sample(lrr):
    """Sec. III-C: R has shape n x k, the source of the O(kn) memory claim."""
    _, idx, refined = lrr
    assert refined.shape == idx.shape == (20, 4)


def test_lns_keeps_the_neighbourhood_of_one_seed():
    feats = np.vstack([np.tile([1.0, 0.0], (10, 1)), np.tile([0.0, 1.0], (10, 1))])
    chosen = local_neighborhood_sample(feats, p=0.5, seed=0)
    assert len(chosen) == 10
    assert len({tuple(feats[i]) for i in chosen}) == 1


def test_lns_at_100_percent_keeps_everything():
    assert len(local_neighborhood_sample(np.ones((7, 2)), p=1.0)) == 7


def test_epsilon_rises_anneals_and_plateaus():
    """Sec. III-D and Fig. 2: up to half the epochs, down to 75%, then constant."""
    eps = [nr_epsilon(e, 40, eps_min=0.4, eps_max=0.6) for e in range(40)]
    assert eps[0] == pytest.approx(0.4)
    assert max(eps) == pytest.approx(eps[20]) == pytest.approx(0.6)
    assert eps[:20] == sorted(eps[:20])
    assert eps[20:30] == sorted(eps[20:30], reverse=True)
    assert all(e == pytest.approx(0.4) for e in eps[30:])


def test_proxy_loss_follows_eq_9():
    torch.manual_seed(0)
    feats, proxies = F.normalize(torch.randn(6, 8), dim=1), F.normalize(torch.randn(3, 8), dim=1)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    logits = feats @ proxies.t() / 0.04
    expected = -(logits[torch.arange(6), labels] - logits.logsumexp(dim=1)).mean()
    assert proxy_loss(feats, labels, proxies, torch.arange(3), 0.04).item() == pytest.approx(expected.item(), abs=1e-4)


def test_hard_loss_follows_eq_10():
    torch.manual_seed(1)
    feats = F.normalize(torch.randn(6, 8), dim=1)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    sim = feats @ feats.t() / 0.04
    terms = []
    for i in range(6):
        same = (labels == labels[i]) & (torch.arange(6) != i)
        hardest_positive, hardest_negative = sim[i][same].min(), sim[i][labels != labels[i]].max()
        terms.append(-(hardest_positive - torch.logaddexp(hardest_positive, hardest_negative)))
    assert hard_contrastive_loss(feats, labels, 0.04).item() == pytest.approx(torch.stack(terms).mean().item(), abs=1e-4)


def test_barlow_twins_prefers_correlated_views():
    torch.manual_seed(2)
    z = torch.randn(64, 8)
    assert barlow_twins_loss(z, z.clone()) < barlow_twins_loss(z, torch.randn(64, 8))


@pytest.mark.parametrize("key,config", [("tau", "tau"), ("lambda_hard", "lambda_hard"), ("num_pids", "num_pids"),
                                        ("num_instances", "num_instances"), ("num_repeats", "num_repeats")])
def test_training_settings_match_the_paper(default_cfg, targets, key, config):
    assert default_cfg["train"][config] == targets["training"][key]


def test_clustering_does_not_use_camera_labels(default_cfg):
    """Sec. II: the method is fully unsupervised and does not rely on camera labels."""
    assert default_cfg["clustering"].get("min_cams_per_cluster", 1) <= 1


def rank(q_pid, q_cam, g_pids, g_cams, g_feats) -> dict:
    return evaluate_rank(
        torch.tensor([[1.0, 0.0]]), np.array([q_pid]), np.array([q_cam]), torch.tensor(g_feats),
        np.array(g_pids), np.array(g_cams), ranks=[1],
    )


def test_junk_images_do_not_count_against_a_query(tmp_path):
    """Market-1501 protocol: id -1 images are junk and are ignored, unlike id 0 distractors."""
    (tmp_path / "bounding_box_test").mkdir()
    for name in ("0001_c2s1_000001_01.jpg", "-1_c2s1_000002_01.jpg", "0000_c2s1_000003_01.jpg"):
        Image.new("RGB", (8, 8)).save(tmp_path / "bounding_box_test" / name)
    gallery = Market1501(tmp_path, "gallery")
    feats = {-1: [1.0, 0.0], 1: [0.8, 0.6], 0: [0.0, 1.0]}
    assert rank(1, 1, gallery.pids, gallery.camids, [feats[p] for p in gallery.pids])["Rank-1"] == 100.0


def test_checkpoints_are_not_selected_on_query_and_gallery():
    """Keeping the best epoch by test mAP makes the reported test mAP optimistic."""
    source = (ROOT / "lsfu_reid" / "train.py").read_text(encoding="utf-8")
    assert 'metrics["mAP"] > best["mAP"]' not in source

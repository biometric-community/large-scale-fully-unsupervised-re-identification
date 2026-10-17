# FIDELITY_AUDIT — LSFU-ReID

Paper: Bertocco et al., *Large-scale Fully-Unsupervised Re-Identification*, TBIOM 2024  
DOI: 10.1109/TBIOM.2024.3446964 · Project: `projects/papers/large-scale-fully-unsupervised-re-identification/`

Scope of match: structural / algorithmic fidelity. Matching Table II numbers is out of scope by default; we still plot ours vs paper targets.

## Per-pass scorecard

| Pass | Focus | Confidence | Fixes applied | Residual gaps |
|------|-------|------------|---------------|---------------|
| 1 | Architecture | 88% | Barlow Twins; ResNet50+BNNeck; LNS; LRR; NR-ε DBSCAN; EMA teacher; co-training permute hook | DenseNet121/OSNet optional/missing (D2); FRR ablation not shipped |
| 2 | Losses / training | 90% | Barlow; `L_proxy + λ L_hard`; centroid proxies; PseudoPKSampler; Adam + LR warmup | Softmax-CE baselines not reimplemented (D8) |
| 3 | Data / eval | 88% | Market-1501 unlabeled train; query/gallery CMC+mAP; student eval (D17) | MSMT17 / Duke / vehicle suites blocked (D1) |
| 4 | Hyperparams / clustering | 86% | 40 ep; BT 15; ε 0.62–0.72 + adaptive; mono-cam filter; `ema_beta=0.9`; student eval | Absolute ε endpoints guessed (D4); FAISS→sklearn (D3); cam filter not in paper (D13) |
| 5 | Packaging / report | 92% | scripts, configs, DEVIATIONS, SOURCE_CODE, REPORT+dev-plot, LICENSE/README | Publish + submodule absorb |

**Weighted mean ≈ 89%.** Stop after Pass 5 (no Pass 6+ unless requested).

## Measured Market-1501 (this rebuild)

| Checkpoint | mAP | Rank-1 | Rank-5 |
|------------|----:|-------:|-------:|
| Epoch 10 | 9.47 | 22.92 | — |
| Epoch 20 | 21.02 | 41.48 | — |
| Epoch 30 | 31.55 | 54.51 | — |
| **Best (ep 40)** | **35.06** | **56.06** | **73.72** |
| Paper Table II (100%, 3 BB) | 85.8 | 94.0 | — |

Gap vs paper is expected under **D2** (single ResNet50 vs ResNet50+DenseNet121+OSNet co-training) plus ε/proxy engineering choices (D4, D13, D16, D17).

## Critical bug found & fixed (Pass 4–5)

| Issue | Evidence | Fix |
|-------|----------|-----|
| EMA `β=0.999` + EMA-only eval | Supervised smoke: student mAP **57%** vs EMA **1%** at ep2 | Eval **student**; `ema_beta=0.9` (D17) |

## Sign-off

| Item | Status |
|------|--------|
| 5-pass complete | Yes |
| Real-data metrics only | Yes (`outputs/logs/train_*.json`) |
| Deviations logged | `DEVIATIONS.md` D1–D17 |
| Upstream | None official (`SOURCE_CODE.md`) |

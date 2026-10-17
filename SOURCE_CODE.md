# Source code survey

## Official / author code for this paper

| Candidate | Result |
|-----------|--------|
| DOI / IEEE page | No linked code artifact found |
| arXiv 2307.14278 | No ancillary code |
| Author GitHub [Gabrielcb](https://github.com/Gabrielcb) | No repo named for this LNS/LRR/NR-ε TBIOM 2024 paper |

**Conclusion:** No public reference implementation for *Large-scale Fully-Unsupervised Re-Identification* (Bertocco et al., TBIOM 2024). This project is a **from-paper** PyTorch rebuild.

## Related author repositories (not vendored)

| Repo | Relation | Use here |
|------|----------|----------|
| [Gabrielcb/Leveraging_ensembles_and_self_supervised_fully_ReID](https://github.com/Gabrielcb/Leveraging_ensembles_and_self_supervised_fully_ReID) | Prior fully-unsupervised / ensemble SSL ReID line | Inspiration only; not copied |
| [Gabrielcb/Unsupervised_selfAdaptative_ReID](https://github.com/Gabrielcb/Unsupervised_selfAdaptative_ReID) | Earlier TIFS cross-domain ReID | Not used |

## In-monorepo reuse

| Component | Source |
|-----------|--------|
| Market-1501 loader / RE / transforms | `projects/papers/cascade-transformer-…/ctu/data/market1501.py`, CapReID, 2d-snet |
| mAP / CMC | `ctu/metrics/reid.py` pattern |
| Plot style | `paper-2-model/templates/plot_style.py` (dev-plot) |

## Stack mapping (paper → ours)

| Paper | Ours |
|-------|------|
| PyTorch + Torchreid + FAISS | PyTorch + torchvision + sklearn NearestNeighbors/DBSCAN |
| ResNet50 + DenseNet121 + OSNet | ResNet50 (optional DenseNet121); OSNet omitted |
| Barlow Twins warmup | `lsfu_reid.losses.barlow_twins_loss` + `BarlowTwins` |
| LRR / LNS / NR-ε | `lsfu_reid.clustering.*` |

# Deviations from the paper

Reproduction target: faithful **runnable** reimplementation on Market-1501, not bit-exact author code.
Every row is referenced from `FIDELITY_AUDIT.md`.

| ID | Paper detail | Our choice | Reason |
|----|--------------|------------|--------|
| D1 | Person + vehicle suites (Market, MSMT17, Duke discontinued, Veri, VehicleID, Veri-Wild) | **Market-1501 only** | Scope of this rebuild; other datasets not registered under `projects/datasets/` for this run |
| D2 | Co-training with ResNet50 + DenseNet121 + OSNet (M=3) | **Default single ResNet50**; DenseNet121 optional via config; OSNet not implemented | Time/GPU; Torchreid OSNet not vendored; multi-BB permutation hook retained for M≥2 |
| D3 | FAISS for k-NN / clustering support | **sklearn `NearestNeighbors` + `DBSCAN(metric=precomputed)`** | No FAISS dependency; equivalent clustering API |
| D4 | Exact ε_min / ε_max numeric endpoints (Fig. 2 qualitative) | `eps_min=0.65`, `eps_max=0.80` + adaptive raise if noise>55% (cap 0.92, step 0.03); `min_samples=2` | Supp. omits endpoints; 0.45–0.60 and later 0.58–0.75@85% still left ~80% noise / mAP~1.5; retuned to keep ≥45% labeled |
| D11 | Barlow Twins warmup length | **15 epochs** (was 3) on Market train | Short BT left near-isotropic embeddings; longer SSL before DBSCAN |
| D12 | LRR non-reciprocal distance | Soft `1−0.5·d_IoU` (vs hard 1.0) + cosine k-NN | Stabilizes DBSCAN chaining under early weak neighborhoods |
| D13 | No explicit camera constraint in clustering | Drop clusters with `<2` cameras (`min_cams_per_cluster=2`) | v3: purity~0.66 but cam_dom~0.78; Market same-cam removal → mAP≈ImageNet (~3%); multi-cam filter for cross-cam learning |
| D14 | Contaminated SSL init | Fresh Barlow from ImageNet each full run (do not reuse mid-train EMA as BT) | v3 `barlow.pt` was EMA after bad unsupervised steps |
| D15 | Backbone head | **BNNeck** (BatchNorm1d) before L2 | Standard ReID practice (Luo et al.); stabilizes proxy CE; not explicit in paper extract |
| D16 | Proxy selection | **Cluster centroid** (L2-normalized mean); `proxy_mode: random` available | Paper says random exemplar; centroid more stable under moderate purity |
| D17 | EMA eval / β | Eval **student**; `ema_beta=0.9` (was 0.999) | β=0.999 left EMA≈init for 40 ep; supervised smoke: student mAP 57% vs EMA 1%; clustering also uses EMA teacher |
| D5 | LRR as primary; FRR ablation | Full LRR Eqs. 1–8; optional Jaccard fallback (`use_lrr: false`) | Fallback documented for debugging; default is LRR |
| D6 | DenseNet121 + OSNet Barlow warmup | Barlow Twins on configured backbones only | Follows D2 |
| D7 | Five TITAN RTX training | Single GPU via `CUDA_VISIBLE_DEVICES=0` | Available hardware |
| D8 | Softmax CE mentioned in some unsupervised ReID baselines | Paper losses **L_proxy + λ L_hard** (Eqs. 9–11) implemented | Faithful to Bertocco et al., not Softmax-CE substitution |
| D9 | Image size not explicit in main Implementation Details | **256×128** (Torchreid / ReID default; user protocol) | Common person ReID resolution |
| D10 | k for LRR not numeric in main text | `k=20` | Standard k-reciprocal neighborhood scale; O(kn) retained |

## Upstream-related

| ID | Note |
|----|------|
| U1 | No official code release — see `SOURCE_CODE.md` |

## Hyperparameters guessed

- `eps_min`, `eps_max`, LRR `k`, DBSCAN `min_samples=4`
- Barlow projector width (2048 default; paper Supp. does not fix dim in extracted text)
- RandomResizedCrop / ColorJitter strengths for BT views (standard Barlow-style)

## Not implemented (out of scope unless requested)

- MSMT17 / vehicle datasets and LNS speedup tables
- OSNet backbone
- C++ LRR-P acceleration from paper runtime section
- Camera-label comparisons from Supplementary

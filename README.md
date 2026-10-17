# Large-scale Fully-Unsupervised Re-Identification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

PyTorch rebuild of Bertocco et al. (TBIOM 2024): Barlow Twins warmup + noise-robust DBSCAN ε scheduling for fully-unsupervised person ReID on **Market-1501**.

**Repository:** https://github.com/biometric-community/large-scale-fully-unsupervised-re-identification

**Paper:** [10.1109/TBIOM.2024.3446964](https://doi.org/10.1109/TBIOM.2024.3446964) · [arXiv:2307.14278](https://arxiv.org/abs/2307.14278)

```bash
bash scripts/setup_env.sh
bash scripts/train.sh configs/smoke.yaml
bash scripts/train_full.sh
bash scripts/eval.sh && bash scripts/report.sh
```

See `DEVIATIONS.md` (single backbone; Market-only).

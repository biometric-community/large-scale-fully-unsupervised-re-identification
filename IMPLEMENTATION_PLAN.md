# Implementation Plan — Large-scale Fully-Unsupervised ReID

1. Barlow Twins warmup (3 ep) on unlabeled Market train.
2. Iterative: extract features → NR-ε DBSCAN → CE on pseudo-IDs (ignore noise).
3. Eval Market query/gallery mAP / Rank-1 (labels unused in training).
4. Size gate Market &lt; 5 GiB → full 40-ep train.

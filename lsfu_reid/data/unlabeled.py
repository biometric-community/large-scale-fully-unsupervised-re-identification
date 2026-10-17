"""Unlabeled Market views for Barlow Twins / clustering."""

from __future__ import annotations

from typing import Callable, List, Optional

from torch.utils.data import Dataset

from .market1501 import Market1501


class TwoCropTransform:
    def __init__(self, base: Callable):
        self.base = base

    def __call__(self, x):
        return self.base(x), self.base(x)


class UnlabeledMarket(Dataset):
    """Wrap Market1501 train; optionally override labels with pseudo-ids."""

    def __init__(self, base: Market1501, pseudo_labels: Optional[List[int]] = None):
        self.base = base
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if self.pseudo_labels is not None:
            item = dict(item)
            item["label"] = int(self.pseudo_labels[idx])
        return item

    @property
    def labels(self):
        if self.pseudo_labels is not None:
            return list(self.pseudo_labels)
        return self.base.labels

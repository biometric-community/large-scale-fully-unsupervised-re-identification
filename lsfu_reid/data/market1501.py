"""Market-1501 loader (CTU / CapReID / 2d-snet pattern). Train IDs unused for loss."""

from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

_ID_RE = re.compile(r"^(-?\d+)_c(\d+)")


def parse_market_name(name: str) -> Tuple[int, int]:
    m = _ID_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized Market filename: {name}")
    return int(m.group(1)), int(m.group(2))


def resolve_market_root(cfg: dict) -> Path:
    root = Path(cfg["data"]["root"])
    if not root.is_absolute():
        proj = Path(__file__).resolve().parents[2]
        cand = (proj / root).resolve()
        if cand.exists():
            root = cand
        else:
            mono = Path(__file__).resolve().parents[4]
            root = (mono / "projects" / "datasets" / "market1501").resolve()
    extracted = root / "extracted" / "Market-1501-v15.09.15"
    if extracted.is_dir():
        return extracted
    if (root / "bounding_box_train").is_dir():
        return root
    raise FileNotFoundError(
        f"Market-1501 not found under {root}. "
        "Place the official release at projects/datasets/market1501/extracted/Market-1501-v15.09.15/"
    )


class Market1501(Dataset):
    def __init__(
        self,
        root: Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        indices: Optional[Sequence[int]] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        if split == "train":
            folder = self.root / "bounding_box_train"
            paths = sorted(folder.glob("*.jpg"))
            paths = [p for p in paths if parse_market_name(p.name)[0] != -1]
        elif split == "query":
            folder = self.root / "query"
            paths = sorted(folder.glob("*.jpg"))
        elif split in ("gallery", "test"):
            folder = self.root / "bounding_box_test"
            paths = sorted(folder.glob("*.jpg"))
        else:
            raise ValueError(split)
        if max_samples is not None:
            paths = paths[: int(max_samples)]
        self.paths = paths
        self.pids: List[int] = []
        self.camids: List[int] = []
        for p in self.paths:
            pid, cam = parse_market_name(p.name)
            self.pids.append(pid)
            self.camids.append(cam)
        if split == "train":
            uniq = sorted(set(self.pids))
            self.pid2label = {p: i for i, p in enumerate(uniq)}
            self.num_classes = len(uniq)
            self.labels = [self.pid2label[p] for p in self.pids]
        else:
            self.pid2label = {}
            self.num_classes = 0
            self.labels = list(self.pids)
        if indices is not None:
            idx = list(indices)
            self.paths = [self.paths[i] for i in idx]
            self.pids = [self.pids[i] for i in idx]
            self.camids = [self.camids[i] for i in idx]
            self.labels = [self.labels[i] for i in idx]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "label": self.labels[idx],
            "pid": self.pids[idx],
            "camid": self.camids[idx],
            "index": idx,
            "path": str(path),
        }


class RandomErasing:
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.485, 0.456, 0.406)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        for _ in range(100):
            area = img.size(1) * img.size(2)
            target = random.uniform(self.sl, self.sh) * area
            aspect = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target * aspect)))
            w = int(round(math.sqrt(target / aspect)))
            if w < img.size(2) and h < img.size(1):
                x1 = random.randint(0, img.size(1) - h)
                y1 = random.randint(0, img.size(2) - w)
                for c in range(img.size(0)):
                    img[c, x1 : x1 + h, y1 : y1 + w] = self.mean[c]
                return img
        return img


def build_transforms(cfg: dict, train: bool) -> Callable:
    h = int(cfg["data"].get("height", 256))
    w = int(cfg["data"].get("width", 128))
    mean = cfg["data"].get("mean", [0.485, 0.456, 0.406])
    std = cfg["data"].get("std", [0.229, 0.224, 0.225])
    if train:
        ops = [
            transforms.Resize((h, w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if cfg["data"].get("random_erasing", True):
            ops.append(RandomErasing(probability=0.5, mean=tuple(mean)))
        return transforms.Compose(ops)
    return transforms.Compose(
        [
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_ssl_transforms(cfg: dict) -> Callable:
    """Two-view Barlow Twins augmentations (paper Supp. + standard BT)."""
    h = int(cfg["data"].get("height", 256))
    w = int(cfg["data"].get("width", 128))
    mean = cfg["data"].get("mean", [0.485, 0.456, 0.406])
    std = cfg["data"].get("std", [0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize((h, w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop((h, w), scale=(0.6, 1.0)),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class PseudoPKSampler(Sampler[int]):
    """Sample P pseudo-IDs × K images (paper Sec. IV-A)."""

    def __init__(
        self,
        labels: Sequence[int],
        num_pids: int = 16,
        num_instances: int = 12,
        num_repeats: int = 5,
    ):
        self.labels = list(labels)
        self.num_pids = int(num_pids)
        self.num_instances = int(num_instances)
        self.num_repeats = int(num_repeats)
        self.index_dic: dict[int, List[int]] = {}
        for i, y in enumerate(self.labels):
            if y < 0:
                continue
            self.index_dic.setdefault(int(y), []).append(i)
        self.pids = sorted(self.index_dic.keys())

    def __iter__(self):
        if not self.pids:
            return iter([])
        for _ in range(self.num_repeats):
            pids = self.pids.copy()
            random.shuffle(pids)
            batch: List[int] = []
            for pid in pids:
                idxs = self.index_dic[pid]
                if len(idxs) >= self.num_instances:
                    chosen = random.sample(idxs, self.num_instances)
                else:
                    chosen = [random.choice(idxs) for _ in range(self.num_instances)]
                batch.extend(chosen)
                if len(batch) >= self.num_pids * self.num_instances:
                    yield from batch
                    batch = []
            if batch:
                # pad incomplete last batch by resampling
                while len(batch) < self.num_pids * self.num_instances and self.pids:
                    pid = random.choice(self.pids)
                    idxs = self.index_dic[pid]
                    batch.append(random.choice(idxs))
                yield from batch[: self.num_pids * self.num_instances]

    def __len__(self) -> int:
        if not self.pids:
            return 0
        per_pass = max(1, math.ceil(len(self.pids) / self.num_pids)) * self.num_pids * self.num_instances
        return per_pass * self.num_repeats


def build_dataloader(cfg: dict, split: str, shuffle: Optional[bool] = None) -> DataLoader:
    root = resolve_market_root(cfg)
    train = split == "train"
    ds = Market1501(
        root,
        split=split if split != "gallery" else "gallery",
        transform=build_transforms(cfg, train=train),
        max_samples=cfg["data"].get(f"max_{split}_samples")
        or (cfg["data"].get("max_train_samples") if train else cfg["data"].get("max_eval_samples")),
    )
    if shuffle is None:
        shuffle = train
    return DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]) if train else int(cfg["eval"].get("batch_size", 64)),
        shuffle=shuffle,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=train,
    )

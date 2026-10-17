"""Supervised smoke v2: eval student vs EMA; try Softmax CE head (no tau proxies)."""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.cuda import amp
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from lsfu_reid.data import Market1501, PseudoPKSampler, build_transforms, resolve_market_root
from lsfu_reid.metrics import evaluate_rank
from lsfu_reid.models import build_backbone
from lsfu_reid.train import extract_eval_pack, update_ema


class IDHead(nn.Module):
    def __init__(self, dim: int, n_class: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(dim, n_class, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x: torch.Tensor):
        # x already BNNeck+L2 from backbone when normalize=True; use unnormalized for CE
        return self.classifier(x)


def main():
    cfg = yaml.safe_load((_ROOT / "configs/default.yaml").read_text())
    device = torch.device("cuda")
    root = resolve_market_root(cfg)
    train_ds = Market1501(root, "train", transform=build_transforms(cfg, True))
    labels = list(train_ds.labels)
    n_class = train_ds.num_classes
    print("train", len(train_ds), "ids", n_class, flush=True)

    # backbone returns normalize=False for CE head
    model = build_backbone(cfg).to(device)
    head = IDHead(model.feat_dim, n_class).to(device)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=3.5e-4, weight_decay=5e-4
    )
    scaler = amp.GradScaler(enabled=True)

    q = Market1501(root, "query", transform=build_transforms(cfg, False))
    g = Market1501(root, "gallery", transform=build_transforms(cfg, False))
    ql = DataLoader(q, batch_size=128, shuffle=False, num_workers=4)
    gl = DataLoader(g, batch_size=128, shuffle=False, num_workers=4)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(train_ds)

        def __getitem__(self, i):
            item = train_ds[i]
            item["pseudo"] = int(labels[i])
            return item

    for ep in range(5):
        t0 = time.time()
        sampler = PseudoPKSampler(labels, num_pids=16, num_instances=4, num_repeats=5)
        loader = DataLoader(DS(), batch_size=64, sampler=sampler, num_workers=4, drop_last=True)
        model.train()
        head.train()
        run = 0.0
        n = 0
        for b in loader:
            y = b["pseudo"].to(device).long()
            x = b["image"].to(device)
            opt.zero_grad(set_to_none=True)
            with amp.autocast(enabled=True):
                # unnormalized BNNeck feat for classification
                feat = model(x, normalize=False)
                logits = head(feat)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            update_ema(ema, model, 0.999)
            run += float(loss.detach())
            n += 1
        # eval student + ema (normalized feats)
        model.eval()
        ema.eval()
        for tag, net in [("student", model), ("ema", ema)]:
            qq = extract_eval_pack(net, ql, device)
            gg = extract_eval_pack(net, gl, device)
            met = evaluate_rank(qq["feats"], qq["pids"], qq["cams"], gg["feats"], gg["pids"], gg["cams"])
            print(
                f"[ce] ep={ep+1} {tag} loss={run/max(n,1):.3f} "
                f"mAP={met['mAP']:.2f} R1={met['Rank-1']:.2f} sec={time.time()-t0:.1f}",
                flush=True,
            )
    print("[ce] done", flush=True)


if __name__ == "__main__":
    main()

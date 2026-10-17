"""Supervised smoke: true Market labels — if mAP rises, eval/model/loss OK."""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.cuda import amp
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from lsfu_reid.data import Market1501, PseudoPKSampler, build_transforms, resolve_market_root
from lsfu_reid.losses import FinalLoss, select_proxies
from lsfu_reid.metrics import evaluate_rank
from lsfu_reid.models import build_backbone
from lsfu_reid.train import extract_eval_pack, update_ema


def main():
    cfg = yaml.safe_load((_ROOT / "configs/default.yaml").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_market_root(cfg)
    train_ds = Market1501(root, "train", transform=build_transforms(cfg, True))
    labels = list(train_ds.labels)
    print("train", len(train_ds), "ids", train_ds.num_classes, flush=True)

    model = build_backbone(cfg).to(device)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
    loss_fn = FinalLoss(tau=0.04, lambd=0.5)
    scaler = amp.GradScaler(enabled=device.type == "cuda")

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
        sampler = PseudoPKSampler(labels, num_pids=16, num_instances=4, num_repeats=3)
        loader = DataLoader(DS(), batch_size=64, sampler=sampler, num_workers=4, drop_last=True)
        with torch.no_grad():
            model.eval()
            fs, ys = [], []
            for b in DataLoader(DS(), batch_size=128, shuffle=False, num_workers=4):
                x = b["image"].to(device)
                y = b["pseudo"]
                fs.append(model(x, normalize=True).cpu())
                ys.append(y)
            proxies, plabels = select_proxies(torch.cat(fs), torch.cat(ys), mode="centroid")
            proxies, plabels = proxies.to(device), plabels.to(device)
        model.train()
        run = 0.0
        n = 0
        for b in loader:
            y = b["pseudo"].to(device).long()
            x = b["image"].to(device)
            opt.zero_grad(set_to_none=True)
            with amp.autocast(enabled=device.type == "cuda"):
                loss = loss_fn(model(x, normalize=True), y, proxies, plabels)["loss"]
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            update_ema(ema, model, 0.999)
            run += float(loss)
            n += 1
        ema.eval()
        qq = extract_eval_pack(ema, ql, device)
        gg = extract_eval_pack(ema, gl, device)
        met = evaluate_rank(qq["feats"], qq["pids"], qq["cams"], gg["feats"], gg["pids"], gg["cams"])
        print(
            f"[sup] ep={ep+1} loss={run/max(n,1):.3f} "
            f"mAP={met['mAP']:.2f} R1={met['Rank-1']:.2f} sec={time.time()-t0:.1f}",
            flush=True,
        )
    print("[sup] done", flush=True)


if __name__ == "__main__":
    main()

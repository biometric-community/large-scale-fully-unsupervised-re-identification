"""Evaluate Market-1501 mAP / CMC with EMA checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lsfu_reid.data import Market1501, build_transforms, resolve_market_root
from lsfu_reid.metrics import evaluate_rank
from lsfu_reid.models import build_backbone
from lsfu_reid.train import extract_eval_pack


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Eval LSFU-ReID on Market-1501")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(
        args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    root = resolve_market_root(cfg)
    tf = build_transforms(cfg, train=False)
    q_ds = Market1501(root, "query", transform=tf, max_samples=cfg["data"].get("max_eval_samples"))
    g_ds = Market1501(root, "gallery", transform=tf, max_samples=cfg["data"].get("max_eval_samples"))
    nw = int(cfg["data"].get("num_workers", 4))
    bs = int(cfg["eval"].get("batch_size", 64))
    q_loader = DataLoader(q_ds, batch_size=bs, shuffle=False, num_workers=nw)
    g_loader = DataLoader(g_ds, batch_size=bs, shuffle=False, num_workers=nw)

    backbones = cfg["model"].get("backbones", [cfg["model"].get("backbone", "resnet50")])
    models = [build_backbone(cfg, name=n).to(device) for n in backbones]
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    emas = ckpt.get("emas", ckpt.get("models"))
    for i, m in enumerate(models):
        m.load_state_dict(emas[i])
        m.eval()

    # Single BB or average distances across BBs (paper Eq. 12)
    if len(models) == 1:
        q = extract_eval_pack(models[0], q_loader, device)
        g = extract_eval_pack(models[0], g_loader, device)
        metrics = evaluate_rank(
            q["feats"], q["pids"], q["cams"], g["feats"], g["pids"], g["cams"],
            ranks=list(cfg["eval"].get("ranks", [1, 5, 10])),
        )
    else:
        import numpy as np
        import torch.nn.functional as F

        dists = []
        q_meta = None
        g_meta = None
        for m in models:
            q = extract_eval_pack(m, q_loader, device)
            g = extract_eval_pack(m, g_loader, device)
            qf = F.normalize(q["feats"], dim=1)
            gf = F.normalize(g["feats"], dim=1)
            dists.append(1.0 - torch.mm(qf, gf.t()))
            q_meta, g_meta = q, g
        dist = torch.stack(dists, 0).mean(0).cpu().numpy()
        # reuse evaluate via manual CMC on averaged dist
        from lsfu_reid.metrics.reid import evaluate_rank as _er

        # wrap: build fake feats so cosine recovers averaged ranking poorly —
        # evaluate directly:
        query_pids, query_camids = q_meta["pids"], q_meta["cams"]
        gallery_pids, gallery_camids = g_meta["pids"], g_meta["cams"]
        ranks = list(cfg["eval"].get("ranks", [1, 5, 10]))
        all_cmc, all_ap = [], []
        for i in range(dist.shape[0]):
            order = np.argsort(dist[i])
            pid, cam = query_pids[i], query_camids[i]
            remove = (gallery_pids[order] == pid) & (gallery_camids[order] == cam)
            keep = ~remove
            ordered = gallery_pids[order][keep]
            matches = ordered == pid
            if not np.any(matches):
                continue
            cmc = matches.cumsum()
            cmc[cmc > 1] = 1
            all_cmc.append(cmc)
            num_rel = matches.sum()
            tmp = matches.cumsum()
            tmp = np.asarray([x / (j + 1.0) for j, x in enumerate(tmp)]) * matches
            all_ap.append(tmp.sum() / num_rel)
        max_len = max(len(c) for c in all_cmc)
        cmc_mat = np.zeros((len(all_cmc), max_len))
        for i, c in enumerate(all_cmc):
            cmc_mat[i, : len(c)] = c
            if len(c) < max_len:
                cmc_mat[i, len(c) :] = c[-1]
        cmc_mean = cmc_mat.mean(0)
        metrics = {"mAP": float(np.mean(all_ap)) * 100.0, "cmc": (cmc_mean * 100).tolist()}
        for r in ranks:
            metrics[f"Rank-{r}"] = float(cmc_mean[r - 1]) * 100.0

    out_dir = Path(cfg["paths"]["log_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "checkpoint": args.checkpoint,
        "metrics": {k: v for k, v in metrics.items() if k != "cmc"},
        "cmc": metrics.get("cmc", []),
        "paper_target": {"mAP": 85.8, "Rank-1": 94.0},
    }
    (out_dir / "eval_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["metrics"], indent=2))
    return out


if __name__ == "__main__":
    main()

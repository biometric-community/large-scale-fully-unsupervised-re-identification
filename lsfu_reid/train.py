"""Train LSFU-ReID: Barlow Twins warmup + iterative LNS/LRR/DBSCAN/proxy loss."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lsfu_reid.clustering import (
    cluster_dbscan,
    filter_monocamera_clusters,
    local_neighborhood_sample,
    local_re_ranking,
    nr_epsilon,
    remapped_cluster_labels,
)
from lsfu_reid.data import (
    Market1501,
    PseudoPKSampler,
    build_ssl_transforms,
    build_transforms,
    resolve_market_root,
)
from lsfu_reid.losses import FinalLoss, barlow_twins_loss, select_proxies
from lsfu_reid.metrics import evaluate_rank
from lsfu_reid.models import BarlowTwins, build_backbone


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _ensure_dirs(cfg: dict) -> None:
    for k in ("checkpoint_dir", "log_dir", "pred_dir", "figure_dir"):
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def extract_numpy_features(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    feats = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        f = model(x, normalize=True)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, 0) if feats else np.zeros((0, 1), dtype=np.float32)


@torch.no_grad()
def extract_eval_pack(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    feats, pids, cams = [], [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        feats.append(model(x, normalize=True).cpu())
        pids.append(batch["pid"].numpy())
        cams.append(batch["camid"].numpy())
    return {
        "feats": torch.cat(feats, 0),
        "pids": np.concatenate(pids),
        "cams": np.concatenate(cams),
    }


def update_ema(ema: nn.Module, model: nn.Module, beta: float = 0.999) -> None:
    with torch.no_grad():
        for e, m in zip(ema.parameters(), model.parameters()):
            e.data.mul_(beta).add_(m.data, alpha=1.0 - beta)
        for e, m in zip(ema.buffers(), model.buffers()):
            e.copy_(m)


def permute_pseudo_labels(label_sets: List[np.ndarray]) -> List[np.ndarray]:
    """Co-training: derangement of pseudo-label sets across backbones (Sec. III-E)."""
    m = len(label_sets)
    if m <= 1:
        return label_sets
    order = list(range(m))
    # random derangement
    for _ in range(20):
        np.random.shuffle(order)
        if all(order[i] != i for i in range(m)):
            break
    else:
        order = [(i + 1) % m for i in range(m)]
    return [label_sets[order[i]] for i in range(m)]


def run_barlow_warmup(cfg: dict, model: nn.Module, device: torch.device, log_dir: Path) -> List[dict]:
    bt_cfg = cfg.get("barlow", {})
    epochs = int(bt_cfg.get("epochs", 3))
    if epochs <= 0:
        return []
    root = resolve_market_root(cfg)
    ssl_tf = build_ssl_transforms(cfg)

    class TwoView(Market1501):
        def __getitem__(self, idx: int):
            path = self.paths[idx]
            from PIL import Image

            img = Image.open(path).convert("RGB")
            return {"v1": ssl_tf(img), "v2": ssl_tf(img)}

    ds = TwoView(
        root,
        "train",
        transform=None,
        max_samples=cfg["data"].get("max_train_samples"),
    )
    loader = DataLoader(
        ds,
        batch_size=int(bt_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )
    bt = BarlowTwins(model, projector_dim=int(bt_cfg.get("projector_dim", 2048))).to(device)
    opt = torch.optim.Adam(
        bt.parameters(),
        lr=float(bt_cfg.get("lr", 3.5e-5)),
        weight_decay=float(bt_cfg.get("weight_decay", 1.5e-6)),
    )
    lambd = float(bt_cfg.get("lambda_bt", 5e-3))
    max_steps = bt_cfg.get("max_steps")
    history = []
    scaler = amp.GradScaler(enabled=device.type == "cuda")
    step = 0
    for ep in range(epochs):
        bt.train()
        running = 0.0
        n = 0
        t0 = time.time()
        for batch in loader:
            v1 = batch["v1"].to(device, non_blocking=True)
            v2 = batch["v2"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp.autocast(enabled=device.type == "cuda"):
                z1, z2 = bt(v1, v2)
                loss = barlow_twins_loss(z1.float(), z2.float(), lambd)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach())
            n += 1
            step += 1
            if max_steps is not None and step >= int(max_steps):
                break
        row = {"stage": "barlow", "epoch": ep + 1, "loss": running / max(n, 1), "sec": time.time() - t0}
        history.append(row)
        print(f"[barlow] epoch={ep+1}/{epochs} loss={row['loss']:.4f} sec={row['sec']:.1f}", flush=True)
        if max_steps is not None and step >= int(max_steps):
            break
    # keep backbone weights in `model`
    model.load_state_dict(bt.backbone.state_dict())
    (log_dir / "barlow_history.json").write_text(json.dumps(history, indent=2))
    return history


def save_barlow_checkpoint(models: List[nn.Module], path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"models": [m.state_dict() for m in models], "cfg": cfg}, path)


def lr_at_epoch(epoch: int, cfg: dict) -> float:
    base = float(cfg["train"]["lr"])
    warm = int(cfg["train"].get("warmup_epochs", 10))
    lr0 = float(cfg["train"].get("lr_start", 3.5e-5))
    if epoch < warm:
        return lr0 + (base - lr0) * float(epoch + 1) / float(max(warm, 1))
    return base


def cluster_subset(
    feats: np.ndarray,
    cfg: dict,
    epoch: int,
    camids: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, int]:
    k = int(cfg["clustering"].get("k", 20))
    use_lrr = bool(cfg["clustering"].get("use_lrr", True))
    if use_lrr:
        _, dist = local_re_ranking(feats, k=k, return_full_distance=True)
    else:
        from lsfu_reid.clustering import jaccard_affinity_fallback

        dist = jaccard_affinity_fallback(feats, k=k)
    eps = nr_epsilon(
        epoch,
        int(cfg["train"]["epochs"]),
        float(cfg["clustering"].get("eps_min", 0.58)),
        float(cfg["clustering"].get("eps_max", 0.75)),
    )
    min_samples = int(cfg["clustering"].get("min_samples", 4))
    max_noise = float(cfg["clustering"].get("max_noise_frac", 0.85))
    eps_cap = float(cfg["clustering"].get("eps_adapt_cap", 0.90))
    eps_step = float(cfg["clustering"].get("eps_adapt_step", 0.05))
    labels = cluster_dbscan(dist, eps=eps, min_samples=min_samples)
    # Adaptive raise when NR-ε absolute scale undershoots LRR distances (early SSL).
    while float((labels < 0).mean()) > max_noise and eps + 1e-9 < eps_cap:
        eps = min(eps_cap, eps + eps_step)
        labels = cluster_dbscan(dist, eps=eps, min_samples=min_samples)
    min_cams = int(cfg["clustering"].get("min_cams_per_cluster", 2))
    if camids is not None and min_cams > 1:
        before = int((labels >= 0).sum())
        labels = filter_monocamera_clusters(labels, camids, min_cams=min_cams)
        dropped = before - int((labels >= 0).sum())
        if dropped > 0:
            print(f"[cluster] mono-cam filter dropped={dropped} (min_cams={min_cams})", flush=True)
    labels, n_clusters = remapped_cluster_labels(labels)
    return labels, eps, n_clusters


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train LSFU-ReID (Bertocco et al.)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-barlow", action="store_true")
    parser.add_argument("--eval-every", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    _ensure_dirs(cfg)
    device = torch.device(
        args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(int(cfg["train"].get("seed", 42)))
    np.random.seed(int(cfg["train"].get("seed", 42)))

    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])

    # size gate record if present
    size_json = log_dir / "dataset_size.json"
    if not size_json.exists():
        import subprocess

        data_root = Path(cfg["data"]["root"])
        if not data_root.is_absolute():
            data_root = (_ROOT / data_root).resolve()
        try:
            total = int(subprocess.check_output(["du", "-sb", str(data_root)], text=True).split()[0])
            payload = {
                "total_bytes": total,
                "total_gib": round(total / (1024**3), 4),
                "decision": "full_train" if total < 5 * 1024**3 else "skip",
                "threshold_bytes": 5 * 1024**3,
            }
            size_json.write_text(json.dumps(payload, indent=2))
            print(f"[size] {payload}", flush=True)
        except Exception as e:
            print(f"[size] warn: {e}", flush=True)

    backbones_cfg = cfg["model"].get("backbones", [cfg["model"].get("backbone", "resnet50")])
    models = [build_backbone(cfg, name=n).to(device) for n in backbones_cfg]
    emas = [copy.deepcopy(m) for m in models]
    for e in emas:
        for p in e.parameters():
            p.requires_grad_(False)

    barlow_ckpt = ckpt_dir / "barlow.pt"
    # Barlow on first backbone (and optionally others)
    if args.resume:
        pass
    elif args.skip_barlow and barlow_ckpt.exists():
        state = torch.load(barlow_ckpt, map_location=device, weights_only=False)
        for i, m in enumerate(models):
            m.load_state_dict(state["models"][i])
            emas[i].load_state_dict(m.state_dict())
        print(f"[barlow] loaded {barlow_ckpt}", flush=True)
    elif not args.skip_barlow:
        for i, m in enumerate(models):
            print(f"[barlow] backbone={backbones_cfg[i]}", flush=True)
            run_barlow_warmup(cfg, m, device, log_dir)
            emas[i].load_state_dict(m.state_dict())
        save_barlow_checkpoint(models, barlow_ckpt, cfg)
        print(f"[barlow] saved {barlow_ckpt}", flush=True)

    root = resolve_market_root(cfg)
    train_ds_full = Market1501(
        root,
        "train",
        transform=build_transforms(cfg, train=True),
        max_samples=cfg["data"].get("max_train_samples"),
    )
    eval_tf = build_transforms(cfg, train=False)
    train_eval_ds = Market1501(
        root,
        "train",
        transform=eval_tf,
        max_samples=cfg["data"].get("max_train_samples"),
    )
    query_ds = Market1501(root, "query", transform=eval_tf, max_samples=cfg["data"].get("max_eval_samples"))
    gallery_ds = Market1501(root, "gallery", transform=eval_tf, max_samples=cfg["data"].get("max_eval_samples"))

    opt = torch.optim.Adam(
        [p for m in models for p in m.parameters()],
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 5e-4)),
    )
    loss_fn = FinalLoss(
        tau=float(cfg["train"].get("tau", 0.04)),
        lambd=float(cfg["train"].get("lambda_hard", 0.5)),
    )
    scaler = amp.GradScaler(enabled=device.type == "cuda")
    epochs = int(cfg["train"]["epochs"])
    lns_every = int(cfg["clustering"].get("lns_every", 3))
    lns_p = float(cfg["clustering"].get("lns_p", 1.0))
    eval_every = args.eval_every or int(cfg["eval"].get("every", 10))
    max_steps = cfg["train"].get("max_steps")

    start_epoch = 0
    history: List[dict] = []
    best = {"mAP": -1.0}
    lns_indices = np.arange(len(train_eval_ds), dtype=np.int64)
    global_step = 0

    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        for i, m in enumerate(models):
            m.load_state_dict(state["models"][i])
            emas[i].load_state_dict(state["emas"][i])
        opt.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", 0))
        history = state.get("history", [])
        best = state.get("best", best)

    nw = int(cfg["data"].get("num_workers", 4))
    feat_loader = DataLoader(
        train_eval_ds,
        batch_size=int(cfg["eval"].get("batch_size", 64)),
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )
    q_loader = DataLoader(query_ds, batch_size=int(cfg["eval"].get("batch_size", 64)), shuffle=False, num_workers=nw)
    g_loader = DataLoader(gallery_ds, batch_size=int(cfg["eval"].get("batch_size", 64)), shuffle=False, num_workers=nw)

    stop = False
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        cur_lr = lr_at_epoch(epoch, cfg)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        # LNS refresh
        if epoch % lns_every == 0 or epoch == start_epoch:
            with torch.no_grad():
                # random backbone for LNS seed features
                bi = int(np.random.randint(0, len(models)))
                feats_all = extract_numpy_features(emas[bi], feat_loader, device)
            lns_indices = local_neighborhood_sample(feats_all, p=lns_p)
            print(f"[lns] epoch={epoch} p={lns_p} n={len(lns_indices)}/{len(train_eval_ds)}", flush=True)

        # Cluster each backbone on LNS subset
        subset_ds = Subset(train_eval_ds, lns_indices.tolist())
        subset_loader = DataLoader(
            subset_ds,
            batch_size=int(cfg["eval"].get("batch_size", 64)),
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
        )
        label_sets: List[np.ndarray] = []
        eps_used = 0.0
        n_clusters_list = []
        subset_cams = np.asarray(train_eval_ds.camids, dtype=np.int64)[lns_indices]
        for i, m in enumerate(emas):
            feats = extract_numpy_features(m, subset_loader, device)
            labels, eps_used, n_c = cluster_subset(feats, cfg, epoch, camids=subset_cams)
            label_sets.append(labels)
            n_clusters_list.append(n_c)
            print(
                f"[cluster] bb={backbones_cfg[i]} epoch={epoch} eps={eps_used:.4f} "
                f"clusters={n_c} noise={(labels < 0).sum()} labeled={(labels >= 0).sum()}",
                flush=True,
            )

        # Co-training permutation
        train_labels = permute_pseudo_labels(label_sets)

        # Map subset labels back to full-index arrays for PK sampling on subset only
        # Train only on LNS indices with pseudo labels
        epoch_losses = []
        for bi, model in enumerate(models):
            labels = train_labels[bi]
            # build train subset with pseudo labels
            class PseudoDS(torch.utils.data.Dataset):
                def __init__(self):
                    self.base = train_ds_full
                    self.indices = lns_indices
                    self.ylabels = labels

                def __len__(self):
                    return len(self.indices)

                def __getitem__(self, i):
                    real = int(self.indices[i])
                    item = self.base[real]
                    item["pseudo"] = int(self.ylabels[i])
                    return item

            pds = PseudoDS()
            # drop noise for sampler pid list: remap -1 excluded
            sampler = PseudoPKSampler(
                [int(labels[i]) for i in range(len(labels))],
                num_pids=int(cfg["train"].get("num_pids", 16)),
                num_instances=int(cfg["train"].get("num_instances", 12)),
                num_repeats=int(cfg["train"].get("num_repeats", 5)),
            )
            if len(sampler.pids) < 2:
                print(f"[train] skip bb={bi}: too few clusters", flush=True)
                continue
            loader = DataLoader(
                pds,
                batch_size=int(cfg["train"].get("num_pids", 16)) * int(cfg["train"].get("num_instances", 12)),
                sampler=sampler,
                num_workers=nw,
                pin_memory=True,
                drop_last=True,
            )
            # renew proxies from current features
            with torch.no_grad():
                model.eval()
                all_f, all_y = [], []
                for batch in DataLoader(pds, batch_size=64, shuffle=False, num_workers=nw):
                    y = torch.tensor([int(t) for t in batch["pseudo"]], dtype=torch.long)
                    keep = y >= 0
                    if keep.any():
                        x = batch["image"][keep].to(device)
                        all_f.append(model(x, normalize=True).cpu())
                        all_y.append(y[keep])
                if not all_f:
                    continue
                feat_cpu = torch.cat(all_f, 0)
                y_cpu = torch.cat(all_y, 0)
                proxies, plabels = select_proxies(
                    feat_cpu,
                    y_cpu,
                    mode=str(cfg["train"].get("proxy_mode", "centroid")),
                )
                proxies = proxies.to(device)
                plabels = plabels.to(device)

            model.train()
            running = 0.0
            nb = 0
            for batch in loader:
                y = batch["pseudo"].to(device) if torch.is_tensor(batch["pseudo"]) else torch.tensor(batch["pseudo"], device=device)
                if y.dtype != torch.long:
                    y = y.long()
                keep = y >= 0
                if keep.sum() < 2:
                    continue
                x = batch["image"].to(device, non_blocking=True)[keep]
                y = y[keep]
                opt.zero_grad(set_to_none=True)
                with amp.autocast(enabled=device.type == "cuda"):
                    feats = model(x, normalize=True)
                    out = loss_fn(feats, y, proxies, plabels)
                    loss = out["loss"]
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                update_ema(emas[bi], model, beta=float(cfg["train"].get("ema_beta", 0.999)))
                running += float(loss.detach())
                nb += 1
                global_step += 1
                if max_steps is not None and global_step >= int(max_steps):
                    stop = True
                    break
            epoch_losses.append(running / max(nb, 1))
            if stop:
                break

        row = {
            "epoch": epoch + 1,
            "lr": cur_lr,
            "eps": eps_used,
            "n_clusters": n_clusters_list,
            "loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "lns_n": int(len(lns_indices)),
            "sec": time.time() - t0,
        }

        # periodic eval — prefer student; EMA@0.999 lags near ImageNet (sup smoke)
        if ((epoch + 1) % eval_every == 0) or (epoch + 1 == epochs) or stop:
            eval_on = str(cfg["eval"].get("model", "student")).lower()
            eval_net = emas[0] if eval_on == "ema" else models[0]
            eval_net.eval()
            q = extract_eval_pack(eval_net, q_loader, device)
            g = extract_eval_pack(eval_net, g_loader, device)
            metrics = evaluate_rank(
                q["feats"], q["pids"], q["cams"], g["feats"], g["pids"], g["cams"],
                ranks=list(cfg["eval"].get("ranks", [1, 5, 10])),
            )
            row["mAP"] = metrics["mAP"]
            row["Rank-1"] = metrics["Rank-1"]
            row["Rank-5"] = metrics.get("Rank-5", 0.0)
            row["eval_model"] = eval_on if eval_on == "ema" else "student"
            print(
                f"[eval] epoch={epoch+1} ({row['eval_model']}) "
                f"mAP={metrics['mAP']:.2f} R1={metrics['Rank-1']:.2f}",
                flush=True,
            )
            if metrics["mAP"] > best["mAP"]:
                # Drop bulky CMC vector from summary JSON
                best = {
                    k: metrics[k]
                    for k in metrics
                    if k != "cmc"
                }
                best["epoch"] = epoch + 1
                best["eval_model"] = row["eval_model"]
                torch.save(
                    {
                        "models": [m.state_dict() for m in models],
                        "emas": [e.state_dict() for e in emas],
                        "epoch": epoch + 1,
                        "best": best,
                        "cfg": cfg,
                    },
                    ckpt_dir / "best.pt",
                )

        history.append(row)
        print(
            f"[epoch] {epoch+1}/{epochs} loss={row['loss']:.4f} eps={eps_used:.4f} sec={row['sec']:.1f}",
            flush=True,
        )
        (log_dir / "train_history.json").write_text(json.dumps(history, indent=2))
        torch.save(
            {
                "models": [m.state_dict() for m in models],
                "emas": [e.state_dict() for e in emas],
                "optimizer": opt.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "best": best,
            },
            ckpt_dir / "last.pt",
        )
        if stop:
            break

    summary = {
        "best": best,
        "paper_target": {"mAP": 85.8, "Rank-1": 94.0},
        "epochs_ran": len(history),
        "backbones": backbones_cfg,
        "config": args.config,
    }
    (log_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] best={best}", flush=True)
    return summary


if __name__ == "__main__":
    main()

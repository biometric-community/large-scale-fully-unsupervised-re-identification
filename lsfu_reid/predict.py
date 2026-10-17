"""Extract embeddings for query/gallery images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lsfu_reid.data import Market1501, build_transforms, resolve_market_root
from lsfu_reid.models import build_backbone
from lsfu_reid.train import extract_eval_pack


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict LSFU-ReID embeddings")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", type=str, default="query", choices=["query", "gallery", "train"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device(
        args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    root = resolve_market_root(cfg)
    ds = Market1501(root, args.split, transform=build_transforms(cfg, train=False))
    loader = DataLoader(ds, batch_size=int(cfg["eval"].get("batch_size", 64)), shuffle=False, num_workers=4)
    bb = cfg["model"].get("backbones", [cfg["model"].get("backbone", "resnet50")])[0]
    model = build_backbone(cfg, name=bb).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("emas", ckpt.get("models"))[0]
    model.load_state_dict(state)
    pack = extract_eval_pack(model, loader, device)
    out_dir = Path(cfg["paths"]["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{args.split}_feats.npz",
        feats=pack["feats"].numpy(),
        pids=pack["pids"],
        cams=pack["cams"],
    )
    meta = {"split": args.split, "n": int(pack["feats"].shape[0]), "dim": int(pack["feats"].shape[1])}
    (out_dir / f"{args.split}_meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()

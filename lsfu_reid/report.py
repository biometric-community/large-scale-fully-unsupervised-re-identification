"""Generate REPORT.md + SVG figures from real train/eval logs (dev-plot)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lsfu_reid.clustering.dbscan_nr import nr_epsilon
from lsfu_reid.plot_style import (
    BAR_COLOR,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    FIGSIZE,
    LINEWIDTH_MAIN,
    MULTI_SERIES_COLORS,
    apply_rcparams,
    apply_style,
)


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def plot_market_bars(eval_m: dict, figdir: Path, paper_map=85.8, paper_r1=94.0):
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels = ["mAP", "Rank-1"]
    ours = [float(eval_m.get("mAP", 0)), float(eval_m.get("Rank-1", 0))]
    paper = [paper_map, paper_r1]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, ours, w, label="Ours", color=BAR_COLOR, edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
    ax.bar(x + w / 2, paper, w, label="Paper (Table II)", color=MULTI_SERIES_COLORS[1], edgecolor="#800000", linewidth=BAR_EDGE_WIDTH)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("Market-1501 Fully-Unsupervised ReID")
    ax.legend()
    apply_style(ax, grid_axis="y")
    fig.tight_layout()
    out = figdir / "market_map_r1.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def plot_eps_schedule(cfg: dict, figdir: Path):
    apply_rcparams()
    epochs = int(cfg["train"]["epochs"])
    xs = list(range(epochs))
    ys = [
        nr_epsilon(
            e,
            epochs,
            float(cfg["clustering"].get("eps_min", 0.4)),
            float(cfg["clustering"].get("eps_max", 0.6)),
        )
        for e in xs
    ]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(xs, ys, color=BAR_COLOR, linewidth=LINEWIDTH_MAIN, label="NR-ε")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ε")
    ax.set_title("Noise-Robust Density Scheduling (Fig. 2 style)")
    ax.legend()
    apply_style(ax)
    fig.tight_layout()
    out = figdir / "eps_schedule.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def plot_train_curves(history: list, figdir: Path):
    apply_rcparams()
    ep = [h["epoch"] for h in history]
    loss = [h.get("loss", 0) for h in history]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(ep, loss, color=BAR_COLOR, linewidth=LINEWIDTH_MAIN, label="Train loss")
    maps = [(h["epoch"], h["mAP"]) for h in history if "mAP" in h]
    if maps:
        ax2 = ax.twinx()
        ax2.plot([e for e, _ in maps], [m for _, m in maps], color=MULTI_SERIES_COLORS[1], linewidth=LINEWIDTH_MAIN, label="mAP")
        ax2.set_ylabel("mAP (%)")
        lines, labels = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(lines + l2, labels + lb2, loc="best")
    else:
        ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training curves (Ours)")
    apply_style(ax)
    fig.tight_layout()
    out = figdir / "train_curves.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def plot_cmc(cmc: list, figdir: Path, max_rank: int = 20):
    if not cmc:
        return None
    apply_rcparams()
    ranks = list(range(1, min(len(cmc), max_rank) + 1))
    vals = cmc[: len(ranks)]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(ranks, vals, color=BAR_COLOR, linewidth=LINEWIDTH_MAIN, marker="o", markersize=4)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Recognition rate (%)")
    ax.set_title("CMC curve (Market-1501, Ours)")
    apply_style(ax)
    fig.tight_layout()
    out = figdir / "cmc.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def write_report(cfg: dict, logs: Path, figdir: Path, report_path: Path):
    hist = load_json(logs / "train_history.json") or []
    summary = load_json(logs / "train_summary.json") or {}
    eval_res = load_json(logs / "eval_results.json") or {}
    size = load_json(logs / "dataset_size.json") or {}
    metrics = eval_res.get("metrics") or summary.get("best") or {}
    figs = []
    if metrics:
        figs.append(plot_market_bars(metrics, figdir))
    figs.append(plot_eps_schedule(cfg, figdir))
    if hist:
        figs.append(plot_train_curves(hist, figdir))
    cmc = eval_res.get("cmc") or []
    if cmc:
        figs.append(plot_cmc(cmc, figdir))

    paper = {"mAP": 85.8, "Rank-1": 94.0}
    lines = [
        "# Report: Large-scale Fully-Unsupervised Re-Identification",
        "",
        "Faithful Market-1501 reconstruction of Bertocco et al., TBIOM 2024 "
        "(DOI [10.1109/TBIOM.2024.3446964](https://doi.org/10.1109/TBIOM.2024.3446964)).",
        "",
        "## Setup",
        "",
        f"- Backbones: `{cfg['model'].get('backbones', [cfg['model'].get('backbone')])}`",
        f"- Image: {cfg['data'].get('height')}×{cfg['data'].get('width')}, LNS p={cfg['clustering'].get('lns_p')}",
        f"- Epochs: Barlow {cfg.get('barlow', {}).get('epochs', 3)} + unsupervised {cfg['train']['epochs']}",
        f"- Dataset size gate: {size.get('decision', 'n/a')} ({size.get('total_gib', '?')} GiB)",
        "",
        "## Market-1501 results",
        "",
        "| | mAP | Rank-1 | Rank-5 |",
        "|--|--:|--:|--:|",
        f"| **Ours** | {metrics.get('mAP', float('nan')):.2f} | {metrics.get('Rank-1', float('nan')):.2f} | {metrics.get('Rank-5', float('nan')):.2f} |",
        f"| Paper (Table II, 100%) | {paper['mAP']:.1f} | {paper['Rank-1']:.1f} | — |",
        "",
        "Identity labels are used **only** for query/gallery evaluation, never for training.",
        "",
        "## Figures",
        "",
    ]
    for p in figs:
        if p is None:
            continue
        rel = p.relative_to(report_path.parent) if p.is_absolute() else p
        lines.append(f"### `{rel}`")
        lines.append("")
        lines.append(f"![]({rel})")
        lines.append("")
    lines += [
        "## Regenerate",
        "",
        "```bash",
        "bash scripts/report.sh",
        "```",
        "",
        "See `DEVIATIONS.md` and `FIDELITY_AUDIT.md` for known gaps vs the paper.",
        "",
    ]
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Report LSFU-ReID")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--logs", type=str, default=None)
    parser.add_argument("--figdir", type=str, default=None)
    args = parser.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    logs = Path(args.logs or cfg["paths"]["log_dir"])
    figdir = Path(args.figdir or cfg["paths"]["figure_dir"])
    figdir.mkdir(parents=True, exist_ok=True)
    write_report(cfg, logs, figdir, _ROOT / "REPORT.md")


if __name__ == "__main__":
    main()

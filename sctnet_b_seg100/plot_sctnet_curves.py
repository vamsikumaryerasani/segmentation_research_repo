#!/usr/bin/env python3
import os
import re
import sys
import glob
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt

COL_TRAIN_ACC  = (255/255.0, 128/255.0, 255/255.0)
COL_VAL_ACC    = (70/255.0,  204/255.0,  51/255.0)
COL_TRAIN_LOSS = (63/255.0,   72/255.0, 204/255.0)
COL_VAL_LOSS   = (255/255.0,   0/255.0,   0/255.0)
COL_MIOU       = (255/255.0, 165/255.0,   0/255.0)


def _plot_safe(ax, x, y, **kwargs):
    xx, yy = [], []
    for a, b in zip(x, y):
        if b is not None:
            xx.append(a)
            yy.append(b)
    if xx:
        return ax.plot(xx, yy, **kwargs)[0]
    return None


def _save_both(fig, outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{stem}.png"), dpi=200)
    fig.savefig(os.path.join(outdir, f"{stem}.svg"), format="svg")


def auto_find_log(work_dir):
    logs = sorted(glob.glob(os.path.join(work_dir, "*.log")))
    if not logs:
        raise FileNotFoundError(f"No .log files found in {work_dir}")
    return logs[-1]


def parse_sctnet_log(log_path):
    # Train iteration lines
    re_train = re.compile(
        r"Epoch\s+\[(\d+)\]\[(\d+)/(\d+)\].*?"
        r"decode\.acc_seg:\s*([0-9.]+).*?"
        r"loss:\s*([0-9.]+)"
    )

    # Val-loss hook lines
    re_val_loss = re.compile(
        r"\[ValLossEarlyStopHook\]\s+epoch=(\d+)\s+val_loss\(mean\)=([0-9.]+)"
    )

    # Validation summary line
    re_val_metrics = re.compile(
        r"Epoch\(val\)\s+\[(\d+)\]\[\d+\]\s+"
        r"aAcc:\s*([0-9.]+),\s*mIoU:\s*([0-9.]+),\s*mAcc:\s*([0-9.]+)"
    )

    train_loss_by_epoch = defaultdict(list)
    train_acc_by_epoch = defaultdict(list)
    val_loss_by_epoch = {}
    miou_by_epoch = {}
    macc_by_epoch = {}
    aacc_by_epoch = {}

    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = re_train.search(line)
            if m:
                ep = int(m.group(1))
                train_acc_by_epoch[ep].append(float(m.group(4)))
                train_loss_by_epoch[ep].append(float(m.group(5)))
                continue

            m = re_val_loss.search(line)
            if m:
                ep = int(m.group(1))
                val_loss_by_epoch[ep] = float(m.group(2))
                continue

            m = re_val_metrics.search(line)
            if m:
                ep = int(m.group(1))
                aacc_by_epoch[ep] = float(m.group(2)) * 100.0
                miou_by_epoch[ep] = float(m.group(3)) * 100.0
                macc_by_epoch[ep] = float(m.group(4)) * 100.0
                continue

    all_epochs = sorted(
        set(train_loss_by_epoch.keys())
        | set(val_loss_by_epoch.keys())
        | set(miou_by_epoch.keys())
    )

    if not all_epochs:
        raise RuntimeError(f"Could not parse SCTNet metrics from log: {log_path}")

    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []   # we do not have direct val_acc in SCTNet log, use aAcc as proxy
    miou = []
    macc = []
    aacc = []

    for ep in all_epochs:
        epochs.append(ep)

        if ep in train_loss_by_epoch and train_loss_by_epoch[ep]:
            train_loss.append(sum(train_loss_by_epoch[ep]) / len(train_loss_by_epoch[ep]))
            train_acc.append(sum(train_acc_by_epoch[ep]) / len(train_acc_by_epoch[ep]))
        else:
            train_loss.append(None)
            train_acc.append(None)

        val_loss.append(val_loss_by_epoch.get(ep, None))
        aacc_ep = aacc_by_epoch.get(ep, None)
        miou_ep = miou_by_epoch.get(ep, None)
        macc_ep = macc_by_epoch.get(ep, None)

        val_acc.append(aacc_ep)
        aacc.append(aacc_ep)
        miou.append(miou_ep)
        macc.append(macc_ep)

    valid_miou = [(i, v) for i, v in enumerate(miou) if v is not None]
    if not valid_miou:
        raise RuntimeError("No mIoU values found in log.")
    best_idx = max(valid_miou, key=lambda x: x[1])[0]

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "mIoU": miou,
        "mAcc": macc,
        "aAcc": aacc,
        "best_epoch": epochs[best_idx],
        "best_miou": miou[best_idx],
        "source": log_path,
    }


def plot_curves(data, model_label, outdir, stem_prefix="sctnet_b_seg100"):
    epochs = data["epochs"]
    train_loss = data["train_loss"]
    train_acc = data["train_acc"]
    val_loss = data["val_loss"]
    val_acc = data["val_acc"]
    miou = data["mIoU"]

    acc_vals = [v for v in (train_acc + val_acc + miou) if v is not None]
    loss_vals = [v for v in (train_loss + val_loss) if v is not None]

    # Combined
    fig, ax_acc = plt.subplots(figsize=(12, 8))
    ax_loss = ax_acc.twinx()

    l1 = _plot_safe(ax_acc, epochs, train_acc, linestyle=":", marker="o", color=COL_TRAIN_ACC, label="Training Acc")
    l2 = _plot_safe(ax_acc, epochs, val_acc, linestyle="-.", marker="x", color=COL_VAL_ACC, label="Validation aAcc")
    l4 = _plot_safe(ax_loss, epochs, train_loss, linestyle="--", marker="^", color=COL_TRAIN_LOSS, label="Training Loss")
    l5 = _plot_safe(ax_loss, epochs, val_loss, linestyle="-", marker="s", color=COL_VAL_LOSS, label="Validation Loss")

    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy [%]")
    ax_loss.set_ylabel("Loss")
    ax_acc.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_acc.set_title(f"{model_label} - Loss and Accuracy")

    if acc_vals:
        ax_acc.set_ylim(max(0, min(acc_vals) - 2.0), min(100, max(acc_vals) + 2.0))
    if loss_vals:
        ax_loss.set_ylim(max(0, min(loss_vals) - 0.02), max(loss_vals) + 0.02)

    lines = [l for l in [l1, l2, l4, l5] if l is not None]
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=4, frameon=True)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save_both(fig, outdir, f"training_curves_{stem_prefix}")
    plt.close(fig)

    # Accuracy-only
    fig, ax = plt.subplots(figsize=(10, 6))
    l1 = _plot_safe(ax, epochs, train_acc, linestyle=":", marker="o", color=COL_TRAIN_ACC, label="Training Acc")
    l2 = _plot_safe(ax, epochs, val_acc, linestyle="-.", marker="x", color=COL_VAL_ACC, label="Validation aAcc")
    l3 = _plot_safe(ax, epochs, miou, linestyle="-", marker="d", color=COL_MIOU, label="mIoU")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy / mIoU [%]")
    ax.set_title(f"{model_label} - Accuracy")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    if acc_vals:
        ax.set_ylim(max(0, min(acc_vals) - 2.0), min(100, max(acc_vals) + 2.0))

    lines = [l for l in [l1, l2, l3] if l is not None]
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=3, frameon=True)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save_both(fig, outdir, f"accuracy_{stem_prefix}")
    plt.close(fig)

    # Loss-only
    fig, ax = plt.subplots(figsize=(10, 6))
    l1 = _plot_safe(ax, epochs, train_loss, linestyle="--", marker="^", color=COL_TRAIN_LOSS, label="Training Loss")
    l2 = _plot_safe(ax, epochs, val_loss, linestyle="-", marker="s", color=COL_VAL_LOSS, label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_label} - Loss")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    if loss_vals:
        ax.set_ylim(max(0, min(loss_vals) - 0.02), max(loss_vals) + 0.02)

    lines = [l for l in [l1, l2] if l is not None]
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=2, frameon=True)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save_both(fig, outdir, f"loss_{stem_prefix}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True, help="SCTNet work dir containing .log")
    ap.add_argument("--logfile", default="", help="Optional explicit .log path")
    ap.add_argument("--label", default="SCTNet-B-Seg100", help="Plot title prefix")
    ap.add_argument("--outdir", default="", help="Output directory")
    ap.add_argument("--stem-prefix", default="sctnet_b_seg100", help="Output file stem prefix")
    args = ap.parse_args()

    log_path = args.logfile if args.logfile else auto_find_log(args.workdir)
    outdir = args.outdir if args.outdir else os.path.join(args.workdir, "plots_updated")

    try:
        data = parse_sctnet_log(log_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    plot_curves(data, model_label=args.label, outdir=outdir, stem_prefix=args.stem_prefix)

    print(f"[INFO] Source used: {data['source']}")
    print(f"[INFO] Epoch range plotted: {data['epochs'][0]} -> {data['epochs'][-1]}")
    print(f"[INFO] Best mIoU: {data['best_miou']:.2f}% at epoch {data['best_epoch']}")
    print("[INFO] Saved PNG + SVG for all 3 plots")


if __name__ == "__main__":
    main()

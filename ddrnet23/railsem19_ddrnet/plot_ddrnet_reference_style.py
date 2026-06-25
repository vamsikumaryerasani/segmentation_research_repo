#!/usr/bin/env python3
import os
import re
import csv
import sys
import argparse
from glob import glob

import matplotlib.pyplot as plt

COL_TRAIN_ACC  = (255/255.0, 128/255.0, 255/255.0)
COL_VAL_ACC    = (70/255.0,  204/255.0,  51/255.0)
COL_TRAIN_LOSS = (63/255.0,   72/255.0, 204/255.0)
COL_VAL_LOSS   = (255/255.0,   0/255.0,   0/255.0)
COL_MIOU       = (255/255.0, 165/255.0,   0/255.0)


def _to_percent_if_needed(x):
    if x is None:
        return None
    x = float(x)
    return x * 100.0 if x <= 1.5 else x


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


def parse_metrics_csv(csv_path):
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    miou = []
    macc = []
    aacc = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(_to_percent_if_needed(row["train_acc"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(_to_percent_if_needed(row["val_acc"]))
            miou.append(_to_percent_if_needed(row["mIoU"]))
            macc.append(_to_percent_if_needed(row["mAcc"]))
            aacc.append(_to_percent_if_needed(row["aAcc"]))

    if not epochs:
        raise RuntimeError(f"No rows found in CSV: {csv_path}")

    best_idx = max(range(len(miou)), key=lambda i: miou[i] if miou[i] is not None else -1)

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
        "source": csv_path,
    }


def parse_train_log(log_path):
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    miou = []
    macc = []
    aacc = []

    re_epoch = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+\|\s+"
        r"train_loss=([0-9.]+)\s+train_acc=([0-9.]+)\s+\|\s+"
        r"val_loss=([0-9.]+)\s+val_acc=([0-9.]+)\s+\|\s+"
        r"mIoU=([0-9.]+)\s+mAcc=([0-9.]+)\s+aAcc=([0-9.]+)"
    )

    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = re_epoch.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(3)))
                train_acc.append(float(m.group(4)))
                val_loss.append(float(m.group(5)))
                val_acc.append(float(m.group(6)))
                miou.append(float(m.group(7)))
                macc.append(float(m.group(8)))
                aacc.append(float(m.group(9)))

    if not epochs:
        raise RuntimeError(f"Could not parse any epoch rows from log: {log_path}")

    best_idx = max(range(len(miou)), key=lambda i: miou[i])

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


def load_data(csv_path=None, log_path=None):
    if csv_path:
        return parse_metrics_csv(csv_path)
    if log_path:
        return parse_train_log(log_path)

    csv_candidates = sorted(glob("*.csv"))
    if csv_candidates:
        return parse_metrics_csv(csv_candidates[0])

    log_candidates = sorted(glob("*.log"))
    if log_candidates:
        return parse_train_log(log_candidates[0])

    raise RuntimeError("No metrics.csv or train.log found.")


def plot_curves(data, model_label, outdir):
    epochs = data["epochs"]
    train_loss = data["train_loss"]
    train_acc = data["train_acc"]
    val_loss = data["val_loss"]
    val_acc = data["val_acc"]
    miou = data["mIoU"]

    acc_vals = [v for v in (train_acc + val_acc) if v is not None]
    loss_vals = [v for v in (train_loss + val_loss) if v is not None]

    # Combined plot without mIoU
    fig, ax_acc = plt.subplots(figsize=(12, 8))
    ax_loss = ax_acc.twinx()

    l1 = _plot_safe(ax_acc, epochs, train_acc, linestyle=":", marker="o", color=COL_TRAIN_ACC, label="Training Acc")
    l2 = _plot_safe(ax_acc, epochs, val_acc, linestyle="-.", marker="x", color=COL_VAL_ACC, label="Validation Acc")
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
    _save_both(fig, outdir, "training_curves_ddrnet23")
    plt.close(fig)

    # Accuracy-only plot, keep mIoU here
    fig, ax = plt.subplots(figsize=(10, 6))
    l1 = _plot_safe(ax, epochs, train_acc, linestyle=":", marker="o", color=COL_TRAIN_ACC, label="Training Acc")
    l2 = _plot_safe(ax, epochs, val_acc, linestyle="-.", marker="x", color=COL_VAL_ACC, label="Validation Acc")
    l3 = _plot_safe(ax, epochs, miou, linestyle="-", marker="d", color=COL_MIOU, label="mIoU")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy / mIoU [%]")
    ax.set_title(f"{model_label} - Accuracy")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    acc_vals_accplot = [v for v in (train_acc + val_acc + miou) if v is not None]
    if acc_vals_accplot:
        ax.set_ylim(max(0, min(acc_vals_accplot) - 2.0), min(100, max(acc_vals_accplot) + 2.0))

    lines = [l for l in [l1, l2, l3] if l is not None]
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=3, frameon=True)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save_both(fig, outdir, "accuracy_ddrnet23")
    plt.close(fig)

    # Loss-only plot
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
    _save_both(fig, outdir, "loss_ddrnet23")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Path to metrics.csv")
    ap.add_argument("--logfile", default=None, help="Path to train.log")
    ap.add_argument("--label", default="DDRNet-23 RailSem19", help="Plot title prefix")
    ap.add_argument("--outdir", default="plots_updated", help="Output directory")
    args = ap.parse_args()

    try:
        data = load_data(csv_path=args.csv, log_path=args.logfile)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    plot_curves(data, model_label=args.label, outdir=args.outdir)

    print(f"[INFO] Source used: {data['source']}")
    print(f"[INFO] Epoch range plotted: {data['epochs'][0]} -> {data['epochs'][-1]}")
    print(f"[INFO] Best mIoU: {data['best_miou']:.2f}% at epoch {data['best_epoch']}")
    print("[INFO] Saved PNG + SVG for all 3 plots")


if __name__ == "__main__":
    main()

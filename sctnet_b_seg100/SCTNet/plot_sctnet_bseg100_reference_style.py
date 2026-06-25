#!/usr/bin/env python3
import re
import sys
import os
import argparse
from glob import glob
import matplotlib.pyplot as plt

# ---- Colors ----
COL_TRAIN_ACC  = (255/255.0, 128/255.0, 255/255.0)  # pink
COL_VAL_ACC    = (70/255.0,  204/255.0,  51/255.0)  # green
COL_TRAIN_LOSS = (63/255.0,   72/255.0, 204/255.0)  # blue
COL_VAL_LOSS   = (255/255.0,   0/255.0,   0/255.0)  # red


def _to_percent_if_needed(x: float) -> float:
    return x * 100.0 if x <= 1.5 else x


def _fmt_miou(miou: float) -> str:
    if miou is None:
        return "n/a"
    return f"{miou*100:.2f}%" if miou <= 1.5 else f"{miou:.2f}%"


def parse_logs(log_paths):
    train_loss_by_ep = {}
    train_acc_by_ep = {}
    val_loss_by_ep = {}
    val_aAcc_by_ep = {}
    val_mIoU_by_ep = {}
    early_bad_by_ep = {}
    early_stop_epoch = None
    best_epoch_from_log = None
    best_miou_from_log = None

    re_train = re.compile(
        r"Epoch\s+\[(\d+)\]\[\d+/\d+\].*?decode\.loss_ce:\s*([0-9.]+).*?decode\.acc_seg:\s*([0-9.]+).*?loss:\s*([0-9.]+)"
    )
    re_val_loss_hook = re.compile(r"\[ValLossHook\]\s*epoch=(\d+).*?val\.loss[:=]\s*([0-9.]+)")
    re_val_loss_1 = re.compile(r"\[ValLossEarlyStopHook\].*?epoch=(\d+).*?val\.loss[:=]\s*([0-9.]+)")
    re_val_loss_2 = re.compile(r"\[ValLossEarlyStopHook\].*?epoch=(\d+).*?val_loss\(mean\)=([0-9.]+)")
    re_val_metrics = re.compile(
        r"Epoch\(val\)\s+\[(\d+)\]\[.*?\].*?aAcc:\s*([0-9.]+).*?mIoU:\s*([0-9.]+).*?mAcc:\s*([0-9.]+)"
    )
    re_early_noimp = re.compile(r"\[EarlyStoppingHook\].*?no improvement:\s*(\d+)/(\d+)")
    re_early_stop = re.compile(r"\[EarlyStoppingHook\].*?Early stopping triggered\.")
    re_best = re.compile(r"Best mIoU is\s*([0-9.]+)\s*at\s*(\d+)\s*epoch")

    current_epoch_for_early = None

    for path in sorted(log_paths):
        with open(path, "r", errors="ignore") as f:
            for line in f:
                m = re_train.search(line)
                if m:
                    ep = int(m.group(1))
                    tl = float(m.group(4))
                    ta = _to_percent_if_needed(float(m.group(3)))
                    train_loss_by_ep[ep] = tl
                    train_acc_by_ep[ep] = ta
                    continue

                m = re_val_metrics.search(line)
                if m:
                    ep = int(m.group(1))
                    aacc = _to_percent_if_needed(float(m.group(2)))
                    miou = _to_percent_if_needed(float(m.group(3)))
                    val_aAcc_by_ep[ep] = aacc
                    val_mIoU_by_ep[ep] = miou
                    continue

                m = re_val_loss_hook.search(line) or re_val_loss_1.search(line) or re_val_loss_2.search(line)
                if m:
                    ep = int(m.group(1))
                    vl = float(m.group(2))
                    val_loss_by_ep[ep] = vl
                    current_epoch_for_early = ep
                    continue

                m = re_early_noimp.search(line)
                if m and current_epoch_for_early is not None:
                    bad = int(m.group(1))
                    pat = int(m.group(2))
                    early_bad_by_ep[current_epoch_for_early] = (bad, pat)
                    continue

                if re_early_stop.search(line) and current_epoch_for_early is not None:
                    early_stop_epoch = current_epoch_for_early
                    continue

                m = re_best.search(line)
                if m:
                    best_miou_from_log = _to_percent_if_needed(float(m.group(1)))
                    best_epoch_from_log = int(m.group(2))
                    continue

    common = sorted(set(train_loss_by_ep.keys()) & set(val_loss_by_ep.keys()))
    if not common:
        common = sorted(set(train_loss_by_ep.keys()) | set(val_loss_by_ep.keys()))

    if not common:
        print("[ERROR] Could not parse any epochs from logs.")
        sys.exit(1)

    epochs = common
    train_loss = [train_loss_by_ep.get(e) for e in epochs]
    train_acc = [train_acc_by_ep.get(e) for e in epochs]
    val_loss = [val_loss_by_ep.get(e) for e in epochs]
    val_acc = [val_aAcc_by_ep.get(e) for e in epochs]

    best_epoch = best_epoch_from_log
    best_miou = best_miou_from_log
    if best_epoch is None and val_mIoU_by_ep:
        best_epoch = max(val_mIoU_by_ep.keys(), key=lambda k: val_mIoU_by_ep[k])
        best_miou = val_mIoU_by_ep[best_epoch]

    last_bad_info = None
    if early_bad_by_ep:
        last_ep = max(early_bad_by_ep.keys())
        last_bad_info = (last_ep, early_bad_by_ep[last_ep])

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "best_epoch": best_epoch,
        "best_miou": best_miou,
        "early_stop_epoch": early_stop_epoch,
        "last_bad_info": last_bad_info,
    }


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
    fig.savefig(os.path.join(outdir, f"{stem}.png"), dpi=200)
    fig.savefig(os.path.join(outdir, f"{stem}.svg"), format="svg")


def plot_curves(data, model_label, outdir):
    os.makedirs(outdir, exist_ok=True)

    epochs = data["epochs"]
    train_loss = data["train_loss"]
    train_acc = data["train_acc"]
    val_loss = data["val_loss"]
    val_acc = data["val_acc"]

    acc_vals = [v for v in (train_acc + val_acc) if v is not None]
    loss_vals = [v for v in (train_loss + val_loss) if v is not None]

    # --- 1) Combined ---
    fig, ax_acc = plt.subplots(figsize=(12, 8))
    ax_loss = ax_acc.twinx()

    l1 = _plot_safe(ax_acc, epochs, train_acc, linestyle=":", marker="o", color=COL_TRAIN_ACC, label="Training Acc")
    l2 = _plot_safe(ax_acc, epochs, val_acc, linestyle="-.", marker="x", color=COL_VAL_ACC, label="Validation Acc")
    l3 = _plot_safe(ax_loss, epochs, train_loss, linestyle="--", marker="^", color=COL_TRAIN_LOSS, label="Training Loss")
    l4 = _plot_safe(ax_loss, epochs, val_loss, linestyle="-", marker="s", color=COL_VAL_LOSS, label="Validation Loss")

    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy [%]")
    ax_loss.set_ylabel("Loss")
    ax_acc.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_acc.set_title(f"{model_label} - Loss and Accuracy")

    if acc_vals:
        ax_acc.set_ylim(max(0, min(acc_vals) - 1.0), min(100, max(acc_vals) + 1.0))
    if loss_vals:
        ax_loss.set_ylim(max(0, min(loss_vals) - 0.02), max(loss_vals) + 0.02)

    # no top-left text box

    lines = [l for l in [l1, l2, l3, l4] if l is not None]
    labels = [l.get_label() for l in lines]
    fig.legend(
        lines, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=4,
        frameon=True
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save_both(fig, outdir, "training_curves_sctnet_b_seg100")
    plt.close(fig)

    # --- 2) Accuracy only ---
    fig, ax = plt.subplots(figsize=(10, 6))
    l1 = _plot_safe(ax, epochs, train_acc, linestyle=":", marker="o", color=COL_TRAIN_ACC, label="Training Acc")
    l2 = _plot_safe(ax, epochs, val_acc, linestyle="-.", marker="x", color=COL_VAL_ACC, label="Validation Acc")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy [%]")
    ax.set_title(f"{model_label} - Accuracy")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    if acc_vals:
        ax.set_ylim(max(0, min(acc_vals) - 1.0), min(100, max(acc_vals) + 1.0))

    lines = [l for l in [l1, l2] if l is not None]
    labels = [l.get_label() for l in lines]
    fig.legend(
        lines, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=2,
        frameon=True
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save_both(fig, outdir, "accuracy_sctnet_b_seg100")
    plt.close(fig)

    # --- 3) Loss only ---
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
    fig.legend(
        lines, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=2,
        frameon=True
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save_both(fig, outdir, "loss_sctnet_b_seg100")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logfile", default=None, help="Optional single .log file. If omitted, uses all *.log in current dir")
    ap.add_argument("--label", default="SCTNet-B-Seg100 RailSem19 Augmented", help="Plot title prefix")
    ap.add_argument("--outdir", default=".", help="Output directory for image files")
    args = ap.parse_args()

    if args.logfile:
        if not os.path.isfile(args.logfile):
            print(f"[ERROR] Log file not found: {args.logfile}")
            sys.exit(1)
        log_paths = [args.logfile]
    else:
        log_paths = sorted(glob("*.log"))
        if not log_paths:
            print("[ERROR] No .log files found in current directory.")
            sys.exit(1)

    data = parse_logs(log_paths)
    plot_curves(data, model_label=args.label, outdir=args.outdir)

    print(f"[INFO] Epoch range plotted: {data['epochs'][0]} -> {data['epochs'][-1]}")
    print("[INFO] Saved PNG + SVG for all 3 plots")


if __name__ == "__main__":
    main()

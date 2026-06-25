#!/usr/bin/env python3
import os
import sys
import json
import argparse
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
    fig.savefig(os.path.join(outdir, f"{stem}.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{stem}.svg"), format="svg", bbox_inches="tight")


def parse_mmseg_jsonlog(jsonlog_path):
    per_epoch = {}

    with open(jsonlog_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            epoch = row.get("epoch", None)
            if epoch is None:
                continue

            if epoch not in per_epoch:
                per_epoch[epoch] = {
                    "train_loss": None,
                    "train_acc": None,
                    "val_loss": None,
                    "val_acc": None,
                    "mIoU": None,
                    "mAcc": None,
                    "aAcc": None,
                }

            mode = row.get("mode", None)

            # train rows
            if mode == "train":
                if "loss" in row:
                    per_epoch[epoch]["train_loss"] = row["loss"]

                # mmseg train accuracy can appear with different keys
                if "decode.acc_seg" in row:
                    per_epoch[epoch]["train_acc"] = row["decode.acc_seg"]
                elif "acc_seg" in row:
                    per_epoch[epoch]["train_acc"] = row["acc_seg"]
                elif "acc" in row:
                    per_epoch[epoch]["train_acc"] = row["acc"]

            # val rows
            if mode == "val" or any(k in row for k in ["mIoU", "mAcc", "aAcc", "val.loss"]):
                if "val.loss" in row:
                    per_epoch[epoch]["val_loss"] = row["val.loss"]
                elif mode == "val" and "loss" in row:
                    per_epoch[epoch]["val_loss"] = row["loss"]

                # for validation accuracy, prefer aAcc
                if "aAcc" in row:
                    per_epoch[epoch]["val_acc"] = row["aAcc"]
                    per_epoch[epoch]["aAcc"] = row["aAcc"]
                elif "acc_seg" in row:
                    per_epoch[epoch]["val_acc"] = row["acc_seg"]
                elif "acc" in row:
                    per_epoch[epoch]["val_acc"] = row["acc"]

                if "mIoU" in row:
                    per_epoch[epoch]["mIoU"] = row["mIoU"]
                if "mAcc" in row:
                    per_epoch[epoch]["mAcc"] = row["mAcc"]
                if "aAcc" in row:
                    per_epoch[epoch]["aAcc"] = row["aAcc"]

    epochs = sorted(per_epoch.keys())
    if not epochs:
        raise RuntimeError(f"No valid epoch entries found in {jsonlog_path}")

    train_loss = [per_epoch[e]["train_loss"] for e in epochs]
    train_acc  = [_to_percent_if_needed(per_epoch[e]["train_acc"]) for e in epochs]
    val_loss   = [per_epoch[e]["val_loss"] for e in epochs]
    val_acc    = [_to_percent_if_needed(per_epoch[e]["val_acc"]) for e in epochs]
    miou       = [_to_percent_if_needed(per_epoch[e]["mIoU"]) for e in epochs]
    macc       = [_to_percent_if_needed(per_epoch[e]["mAcc"]) for e in epochs]
    aacc       = [_to_percent_if_needed(per_epoch[e]["aAcc"]) for e in epochs]

    valid_miou = [(i, v) for i, v in enumerate(miou) if v is not None]
    if valid_miou:
        best_idx, best_miou = max(valid_miou, key=lambda t: t[1])
        best_epoch = epochs[best_idx]
    else:
        best_idx, best_miou, best_epoch = None, None, None

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "mIoU": miou,
        "mAcc": macc,
        "aAcc": aacc,
        "best_epoch": best_epoch,
        "best_miou": best_miou,
        "source": jsonlog_path,
    }


def plot_curves(data, model_label, outdir, stem_prefix):
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
    _save_both(fig, outdir, f"{stem_prefix}_training_curves")
    plt.close(fig)

    # Accuracy-only plot
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
    _save_both(fig, outdir, f"{stem_prefix}_accuracy")
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
    _save_both(fig, outdir, f"{stem_prefix}_loss")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonlog", required=True, help="Path to mmseg .log.json")
    ap.add_argument("--label", default="SCTNet", help="Plot title prefix")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--stem-prefix", default="sctnet", help="Filename stem prefix")
    args = ap.parse_args()

    try:
        data = parse_mmseg_jsonlog(args.jsonlog)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    plot_curves(data, model_label=args.label, outdir=args.outdir, stem_prefix=args.stem_prefix)

    print(f"[INFO] Source used: {data['source']}")
    print(f"[INFO] Epoch range plotted: {data['epochs'][0]} -> {data['epochs'][-1]}")
    if data["best_miou"] is not None:
        print(f"[INFO] Best mIoU: {data['best_miou']:.2f}% at epoch {data['best_epoch']}")
    else:
        print("[WARN] Best mIoU not available")
    print("[INFO] Saved PNG + SVG for all 3 plots")


if __name__ == "__main__":
    main()

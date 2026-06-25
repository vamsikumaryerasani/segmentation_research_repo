#!/usr/bin/env python3
import os
import csv
import json
import glob
import argparse

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


def parse_logjsons(workdir):
    files = sorted(glob.glob(os.path.join(workdir, "*.log.json")))
    train_by_epoch = {}
    val_by_epoch = {}

    for fp in files:
        with open(fp, "r", errors="ignore") as f:
            for line in f:
                try:
                    row = json.loads(line.strip())
                except Exception:
                    continue

                ep = row.get("epoch", None)
                if ep is None:
                    continue

                mode = row.get("mode", None)
                if mode == "train":
                    train_by_epoch.setdefault(ep, {"loss": [], "acc": []})
                    if row.get("loss", None) is not None:
                        train_by_epoch[ep]["loss"].append(float(row["loss"]))
                    elif row.get("decode.loss_ce", None) is not None:
                        train_by_epoch[ep]["loss"].append(float(row["decode.loss_ce"]))
                    if row.get("decode.acc_seg", None) is not None:
                        train_by_epoch[ep]["acc"].append(float(row["decode.acc_seg"]))

                elif mode == "val":
                    aacc = row.get("aAcc", None)
                    miou = row.get("mIoU", None)
                    macc = row.get("mAcc", None)
                    val_by_epoch[ep] = {
                        "aAcc": float(aacc) * 100.0 if aacc is not None and float(aacc) <= 1.5 else (float(aacc) if aacc is not None else None),
                        "mIoU": float(miou) * 100.0 if miou is not None and float(miou) <= 1.5 else (float(miou) if miou is not None else None),
                        "mAcc": float(macc) * 100.0 if macc is not None and float(macc) <= 1.5 else (float(macc) if macc is not None else None),
                    }

    return train_by_epoch, val_by_epoch


def parse_val_loss_csv(csv_path):
    out = {}
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[int(row["epoch"])] = float(row["val_loss"])
    return out


def build_data(workdir, val_loss_csv):
    train_by_epoch, val_by_epoch = parse_logjsons(workdir)
    val_loss_by_epoch = parse_val_loss_csv(val_loss_csv)

    epochs = sorted(set(train_by_epoch.keys()) | set(val_by_epoch.keys()) | set(val_loss_by_epoch.keys()))

    train_loss, train_acc, val_loss, val_acc, miou, macc, aacc = [], [], [], [], [], [], []

    for ep in epochs:
        trec = train_by_epoch.get(ep, {})
        tl = trec.get("loss", [])
        ta = trec.get("acc", [])

        train_loss.append(sum(tl) / len(tl) if tl else None)
        train_acc.append(sum(ta) / len(ta) if ta else None)

        vrec = val_by_epoch.get(ep, {})
        val_loss.append(val_loss_by_epoch.get(ep, None))
        val_acc.append(vrec.get("aAcc", None))
        miou.append(vrec.get("mIoU", None))
        macc.append(vrec.get("mAcc", None))
        aacc.append(vrec.get("aAcc", None))

    best_idx = max(
        [(i, v) for i, v in enumerate(miou) if v is not None],
        key=lambda x: x[1]
    )[0]

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
    }


def plot_curves(data, model_label, outdir, stem_prefix):
    epochs = data["epochs"]
    train_loss = data["train_loss"]
    train_acc = data["train_acc"]
    val_loss = data["val_loss"]
    val_acc = data["val_acc"]
    miou = data["mIoU"]

    acc_vals = [v for v in (train_acc + val_acc + miou) if v is not None]
    loss_vals = [v for v in (train_loss + val_loss) if v is not None]

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--val-loss-csv", required=True)
    ap.add_argument("--label", default="SCTNet-B-Seg100")
    ap.add_argument("--stem-prefix", default="sctnet_b_seg100")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    outdir = args.outdir if args.outdir else os.path.join(args.workdir, "plots_updated")
    data = build_data(args.workdir, args.val_loss_csv)
    plot_curves(data, args.label, outdir, args.stem_prefix)

    print(f"[INFO] Epoch range plotted: {data['epochs'][0]} -> {data['epochs'][-1]}")
    print(f"[INFO] Best mIoU: {data['best_miou']:.2f}% at epoch {data['best_epoch']}")


if __name__ == "__main__":
    main()

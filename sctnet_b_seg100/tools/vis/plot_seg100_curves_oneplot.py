#!/usr/bin/env python3
"""
Usage:
  python tools/vis/plot_seg100_curves_oneplot.py /path/to/train.log "MODEL TITLE"

Outputs in current folder:
  training_curves.png
  per_class_best_epoch.csv
  per_class_best_epoch.md

Notes:
- Train loss/acc from: Epoch [E][...], decode.loss_ce, decode.acc_seg
- Val loss from: [ValLossEarlyStopHook] epoch=E val_loss(mean)=X
- Val accuracy from: Epoch(val) ... aAcc: 0.xxx  (converted to %)
- Per-class table parsed from the "per class results" block after evaluation.
"""

import os
import re
import sys
import csv
import matplotlib.pyplot as plt


# ---- colors (same style you used) ----
COL_TRAIN_ACC  = (255/255.0, 128/255.0, 255/255.0)  # pink
COL_VAL_ACC    = (70/255.0,  204/255.0,  51/255.0)  # green
COL_TRAIN_LOSS = (63/255.0,   72/255.0, 204/255.0)  # blue
COL_VAL_LOSS   = (255/255.0,   0/255.0,   0/255.0)  # red


def parse_log(log_path):
    # last train metrics per epoch
    tr_loss = {}
    tr_acc = {}

    # val metrics per epoch
    va_loss = {}   # val_loss(mean)
    va_acc = {}    # aAcc (%)
    va_miou = {}   # mIoU (0..1)

    # per-class table per epoch: {epoch: [(cls, iou, acc), ...]}
    per_class = {}

    # ---- regex ----
    re_train = re.compile(
        r"Epoch\s+\[(\d+)\]\[\d+/\d+\].*decode\.loss_ce:\s*([0-9.]+).*decode\.acc_seg:\s*([0-9.]+)"
    )
    re_val_loss = re.compile(
        r"\[ValLossEarlyStopHook\]\s+epoch=(\d+)\s+val_loss\(mean\)=([0-9.]+)"
    )
    re_val_metrics = re.compile(
        r"Epoch\(val\)\s+\[(\d+)\]\[\d+\].*aAcc:\s*([0-9.]+).*mIoU:\s*([0-9.]+)"
    )

    # detect per-class table start
    # We'll look for:
    # "per class results:" then the table rows until the next "+-------+"
    re_perclass_header = re.compile(r"per class results:")
    re_epoch_val_line = re.compile(r"Epoch\(val\)\s+\[(\d+)\]\[")

    # row example: |   0   | 51.86 | 60.04 |
    re_row = re.compile(r"^\|\s*([0-9]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")

    current_epoch_for_table = None
    in_table = False
    table_rows = []

    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            # train
            m = re_train.search(line)
            if m:
                e = int(m.group(1))
                tr_loss[e] = float(m.group(2))
                tr_acc[e]  = float(m.group(3))  # already %
                continue

            # val loss mean
            m = re_val_loss.search(line)
            if m:
                e = int(m.group(1))
                va_loss[e] = float(m.group(2))
                continue

            # val metrics
            m = re_val_metrics.search(line)
            if m:
                e = int(m.group(1))
                aacc = float(m.group(2)) * 100.0
                miou = float(m.group(3))
                va_acc[e] = aacc
                va_miou[e] = miou
                continue

            # per-class table parsing
            if re_perclass_header.search(line):
                # epoch is usually nearby in log; we’ll set it when we see Epoch(val) line
                in_table = False
                table_rows = []
                continue

            m = re_epoch_val_line.search(line)
            if m:
                current_epoch_for_table = int(m.group(1))
                # do not continue; this line may also be something else later
                # continue
            if line.strip().startswith("+-------+") and current_epoch_for_table is not None:
                # table borders appear multiple times; toggle logic:
                if not in_table:
                    # next lines likely include header/rows
                    in_table = True
                    table_rows = []
                else:
                    # second border after rows; finalize if we have rows
                    if table_rows:
                        per_class[current_epoch_for_table] = table_rows[:]
                    in_table = False
                continue

            if in_table:
                r = re_row.match(line.strip())
                if r:
                    cls = int(r.group(1))
                    iou = float(r.group(2))
                    acc = float(r.group(3))
                    table_rows.append((cls, iou, acc))

    # build aligned epoch list: only epochs where we have train+val_loss
    epochs = sorted(set(tr_loss.keys()) & set(va_loss.keys()))
    if not epochs:
        raise RuntimeError("No common epochs found between train lines and val_loss(mean) lines.")

    train_loss = [tr_loss[e] for e in epochs]
    train_acc  = [tr_acc[e] for e in epochs]
    val_loss   = [va_loss[e] for e in epochs]

    # val acc might be missing in some epochs; keep None and skip those points
    val_acc = [va_acc.get(e, None) for e in epochs]

    # choose "best epoch" by mIoU if available, else last epoch
    if va_miou:
        best_epoch = max(va_miou.keys(), key=lambda k: va_miou[k])
    else:
        best_epoch = epochs[-1]

    return epochs, train_loss, train_acc, val_loss, val_acc, best_epoch, per_class


def save_per_class_tables(per_class, best_epoch):
    rows = per_class.get(best_epoch, [])
    if not rows:
        print(f"[WARN] No per-class table found for epoch {best_epoch}. Skipping table export.")
        return

    csv_path = "per_class_best_epoch.csv"
    md_path  = "per_class_best_epoch.md"

    # CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Class", "IoU", "Acc"])
        for cls, iou, acc in rows:
            w.writerow([cls, iou, acc])

    # Markdown
    with open(md_path, "w") as f:
        f.write(f"# Per-class results (best epoch = {best_epoch})\n\n")
        f.write("| Class | IoU | Acc |\n")
        f.write("|---:|---:|---:|\n")
        for cls, iou, acc in rows:
            f.write(f"| {cls} | {iou:.2f} | {acc:.2f} |\n")

    print(f"[OK] Saved: {csv_path}")
    print(f"[OK] Saved: {md_path}")


def plot_one_combined(epochs, train_loss, train_acc, val_loss, val_acc, model_label):
    # filter val acc points where not None
    va_x = [e for e, a in zip(epochs, val_acc) if a is not None]
    va_y = [a for a in val_acc if a is not None]

    fig, ax_loss = plt.subplots(figsize=(12, 8))
    ax_acc = ax_loss.twinx()

    # LOSS (left)
    l_train_loss, = ax_loss.plot(
        epochs, train_loss,
        linestyle="--", marker="^",
        color=COL_TRAIN_LOSS,
        label="Training Loss",
    )
    l_val_loss, = ax_loss.plot(
        epochs, val_loss,
        linestyle="-", marker="s",
        color=COL_VAL_LOSS,
        label="Validation Loss",
    )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, which="both", linestyle="--", alpha=0.3)

    # ACC (right)
    l_train_acc, = ax_acc.plot(
        epochs, train_acc,
        linestyle=":", marker="o",
        color=COL_TRAIN_ACC,
        label="Training Acc",
    )
    l_val_acc = None
    if va_x:
        l_val_acc, = ax_acc.plot(
            va_x, va_y,
            linestyle="-", marker="x",
            color=COL_VAL_ACC,
            label="Validation Acc",
        )
    ax_acc.set_ylabel("Accuracy [%]")
    ax_acc.set_ylim(0, 100)  # FIX: cap at 100

    ax_loss.set_title(f"{model_label} - Loss and Accuracy")

    # Legend BELOW (like your example)
    lines = [l_train_acc]
    if l_val_acc is not None:
        lines.append(l_val_acc)
    lines += [l_train_loss, l_val_loss]
    labels = [l.get_label() for l in lines]

    fig.legend(
        lines, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        fancybox=True
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    out = "training_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved: {out}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_seg100_curves_oneplot.py path/to/logfile.log [MODEL_LABEL]")
        sys.exit(1)

    log_path = sys.argv[1]
    model_label = sys.argv[2] if len(sys.argv) >= 3 else "SCTNet RailSem19"

    if not os.path.isfile(log_path):
        print(f"[ERROR] Log not found: {log_path}")
        sys.exit(1)

    epochs, tr_loss, tr_acc, va_loss, va_acc, best_epoch, per_class = parse_log(log_path)

    # one combined plot only
    plot_one_combined(epochs, tr_loss, tr_acc, va_loss, va_acc, model_label)

    # export per-class table for report
    save_per_class_tables(per_class, best_epoch)

    print(f"[INFO] best_epoch_for_table = {best_epoch} (table only; NOT printed on graph)")


if __name__ == "__main__":
    main()

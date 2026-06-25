#!/usr/bin/env python3
"""
Usage:
    python plot_sctnet_curves_newlog.py path/to/logfile.log [MODEL_LABEL]

Examples:
    python plot_sctnet_curves_newlog.py work_dirs/.../20260111_113956.log
    python plot_sctnet_curves_newlog.py work_dirs/.../20260111_113956.log "SCTNet-B-Seg100 RailSem19"

Outputs (in current directory):
  training_curves_sctnet_prof.png   (combined loss+accuracy)
  accuracy_sctnet_prof.png          (accuracy only)
  loss_sctnet_prof.png              (loss only)

This parser supports BOTH:
  - old style: "Iter [...] decode.loss_ce: ... decode.acc_seg: ..." and "[ValLossHook] epoch=..."
  - new style: "Epoch [e][i/...]" training lines, "Epoch(val) [e][...]" metrics lines,
               and "[ValLossEarlyStopHook] ... val_loss(mean)=..."
"""

import re
import sys
import matplotlib.pyplot as plt


def _to_percent(x: float) -> float:
    return x * 100.0 if x <= 1.0 else x


def parse_log(path):
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    # New style training:
    # Epoch [23][850/860] ... decode.loss_ce: 0.3449, decode.acc_seg: 87.7124, loss: 0.3449
    re_train_epoch = re.compile(
        r"Epoch\s+\[(\d+)\]\[\d+/\d+\].*decode\.loss_ce:\s*([0-9.]+).*decode\.acc_seg:\s*([0-9.]+)"
    )

    # Old style training:
    re_train_iter = re.compile(
        r"Iter\s+\[\d+/\d+\].*decode\.loss_ce:\s*([0-9.]+).*decode\.acc_seg:\s*([0-9.]+)"
    )

    # Val loss:
    # [ValLossEarlyStopHook] epoch=34 val_loss(mean)=0.380944
    # or val_loss=70.038620 or val.loss=...
    re_val_loss = re.compile(
        r"\[(?:ValLossEarlyStopHook|ValLossHook)\]\s*epoch=(\d+)\s+.*?"
        r"(?:val_loss\(mean\)|val_loss|val\.loss)=\s*([0-9.]+)"
    )

    # Val acc:
    # Epoch(val) [34][765]   aAcc: 0.8679, mIoU: 0.5844, ...
    re_val_aacc = re.compile(
        r"Epoch\(val\)\s*\[(\d+)\]\[\d+\]\s*aAcc:\s*([0-9.]+)"
    )

    train_by_epoch = {}     # epoch -> (loss, acc)
    val_loss_by_epoch = {}  # epoch -> loss
    val_acc_by_epoch = {}   # epoch -> aAcc(%)

    last_train_loss = None
    last_train_acc = None

    with open(path, "r") as f:
        for line in f:
            m = re_train_epoch.search(line)
            if m:
                e = int(m.group(1))
                last_train_loss = float(m.group(2))
                last_train_acc = float(m.group(3))
                train_by_epoch[e] = (last_train_loss, last_train_acc)
                continue

            m = re_train_iter.search(line)
            if m:
                last_train_loss = float(m.group(1))
                last_train_acc = float(m.group(2))
                continue

            m = re_val_loss.search(line)
            if m:
                e = int(m.group(1))
                vloss = float(m.group(2))
                val_loss_by_epoch[e] = vloss
                if e not in train_by_epoch and last_train_loss is not None and last_train_acc is not None:
                    train_by_epoch[e] = (last_train_loss, last_train_acc)
                continue

            m = re_val_aacc.search(line)
            if m:
                e = int(m.group(1))
                aacc = float(m.group(2))
                val_acc_by_epoch[e] = _to_percent(aacc)
                continue

    aligned_epochs = sorted(set(train_by_epoch.keys()) & set(val_loss_by_epoch.keys()) & set(val_acc_by_epoch.keys()))
    if not aligned_epochs:
        print("No aligned epochs found. Your log must include:")
        print("  - Training: decode.loss_ce and decode.acc_seg lines")
        print("  - Val loss: [ValLossHook]/[ValLossEarlyStopHook] val_loss(...) lines")
        print("  - Val acc:  Epoch(val) ... aAcc: ... lines")
        sys.exit(1)

    for e in aligned_epochs:
        epochs.append(e)
        train_loss.append(train_by_epoch[e][0])
        train_acc.append(train_by_epoch[e][1])
        val_loss.append(val_loss_by_epoch[e])
        val_acc.append(val_acc_by_epoch[e])

    return epochs, train_loss, train_acc, val_loss, val_acc


def plot_curves(epochs, train_loss, train_acc, val_loss, val_acc, model_label="SCTNet RailSem19"):
    # Colors (RGB 0–255 -> matplotlib 0–1), SAME as your example
    col_train_acc  = (255 / 255.0, 128 / 255.0, 255 / 255.0)  # Training Acc
    col_val_acc    = (70 / 255.0, 204 / 255.0, 51 / 255.0)    # Validation Acc
    col_train_loss = (63 / 255.0, 72 / 255.0, 204 / 255.0)    # Training Loss
    col_val_loss   = (255 / 255.0, 0 / 255.0, 0 / 255.0)      # Validation Loss

    # ---------- 1) Combined plot ----------
    fig, ax_left = plt.subplots(figsize=(12, 8))
    ax_right = ax_left.twinx()

    # Accuracy on LEFT
    l_train_acc, = ax_left.plot(
        epochs, train_acc, linestyle=":", marker="o",
        color=col_train_acc, label="Training Acc"
    )
    l_val_acc, = ax_left.plot(
        epochs, val_acc, linestyle="-.", marker="x",
        color=col_val_acc, label="Validation Acc"
    )
    ax_left.set_xlabel("Epoch")
    ax_left.set_ylabel("Accuracy [%]")
    ax_left.grid(True, which="both", linestyle="--", alpha=0.3)

    # Loss on RIGHT
    l_train_loss, = ax_right.plot(
        epochs, train_loss, linestyle="--", marker="^",
        color=col_train_loss, label="Training Loss"
    )
    l_val_loss, = ax_right.plot(
        epochs, val_loss, linestyle="-", marker="s",
        color=col_val_loss, label="Validation Loss"
    )
    ax_right.set_ylabel("Loss")

    ax_left.set_title(f"{model_label} - Loss and Accuracy")

    # Legend below plot (all 4)
    lines = [l_train_acc, l_val_acc, l_train_loss, l_val_loss]
    labels = [l.get_label() for l in lines]
    fig.legend(
        lines, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.subplots_adjust(bottom=0.22)
    out_combined = "training_curves_sctnet_prof.png"
    fig.savefig(out_combined, dpi=200, bbox_inches="tight", pad_inches=0.25)
    print(f"Saved combined plot to {out_combined}")

    # ---------- 2) Accuracy-only ----------
    fig_acc, axa = plt.subplots(figsize=(10, 6))
    axa.plot(epochs, train_acc, linestyle=":", marker="o", color=col_train_acc, label="Training Acc")
    axa.plot(epochs, val_acc,   linestyle="-.", marker="x", color=col_val_acc,   label="Validation Acc")
    axa.set_xlabel("Epoch")
    axa.set_ylabel("Accuracy [%]")
    axa.set_title(f"{model_label} - Accuracy")
    axa.grid(True, which="both", linestyle="--", alpha=0.3)
    axa.legend(loc="best", frameon=False)
    fig_acc.tight_layout()
    out_acc = "accuracy_sctnet_prof.png"
    fig_acc.savefig(out_acc, dpi=200)
    print(f"Saved accuracy plot to {out_acc}")

    # ---------- 3) Loss-only ----------
    fig_loss, axl = plt.subplots(figsize=(10, 6))
    axl.plot(epochs, train_loss, linestyle="--", marker="^", color=col_train_loss, label="Training Loss")
    axl.plot(epochs, val_loss,   linestyle="-",  marker="s", color=col_val_loss,   label="Validation Loss")
    axl.set_xlabel("Epoch")
    axl.set_ylabel("Loss")
    axl.set_title(f"{model_label} - Loss")
    axl.grid(True, which="both", linestyle="--", alpha=0.3)
    axl.legend(loc="best", frameon=False)
    fig_loss.tight_layout()
    out_loss = "loss_sctnet_prof.png"
    fig_loss.savefig(out_loss, dpi=200)
    print(f"Saved loss plot to {out_loss}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_sctnet_curves_newlog.py path/to/logfile.log [MODEL_LABEL]")
        sys.exit(1)

    log_path = sys.argv[1]
    model_label = sys.argv[2] if len(sys.argv) >= 3 else "SCTNet RailSem19"

    epochs, tr_loss, tr_acc, va_loss, va_acc = parse_log(log_path)
    plot_curves(epochs, tr_loss, tr_acc, va_loss, va_acc, model_label=model_label)


if __name__ == "__main__":
    main()

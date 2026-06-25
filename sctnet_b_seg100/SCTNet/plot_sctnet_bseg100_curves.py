#!/usr/bin/env python3
"""
Plot SCTNet-B-Seg100 MMseg training curves.

Works with:
- .log (text logger)
- .log.json (json logger)  <-- preferred if available

Outputs (to --outdir):
- training_curves.png   (combined: loss on left axis, accuracy on right axis)
"""

import os
import re
import json
import argparse

import matplotlib.pyplot as plt


def _to_percent_if_needed(x: float) -> float:
    return x * 100.0 if x <= 1.5 else x


def parse_json_log(json_path):
    train_last = {}
    val_metrics = {}
    val_loss = {}
    early_bad = {}

    with open(json_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue

            ep = d.get("epoch", None)
            mode = d.get("mode", None)

            if mode == "train" and ep is not None:
                tl = d.get("decode.loss_ce", None)
                ta = d.get("decode.acc_seg", None)
                if tl is not None and ta is not None:
                    train_last[int(ep)] = {
                        "loss": float(tl),
                        "acc": _to_percent_if_needed(float(ta)),
                    }

            if mode == "val" and ep is not None:
                aAcc = d.get("aAcc", None)
                mIoU = d.get("mIoU", None)
                mAcc = d.get("mAcc", None)

                if aAcc is not None or mIoU is not None or mAcc is not None:
                    val_metrics[int(ep)] = {
                        "aAcc": _to_percent_if_needed(float(aAcc)) if aAcc is not None else None,
                        "mIoU": _to_percent_if_needed(float(mIoU)) if mIoU is not None else None,
                        "mAcc": _to_percent_if_needed(float(mAcc)) if mAcc is not None else None,
                    }

                vloss = d.get("val.loss", None)
                if vloss is not None:
                    val_loss[int(ep)] = float(vloss)

                bad = d.get("bad", None)
                patience = d.get("patience", None)
                if bad is not None and patience is not None:
                    early_bad[int(ep)] = (int(bad), int(patience))

    return train_last, val_loss, val_metrics, early_bad


def parse_text_log(log_path):
    last_train_loss_by_epoch = {}
    last_train_acc_by_epoch = {}
    val_loss_by_epoch = {}
    val_aacc_by_epoch = {}
    val_miou_by_epoch = {}
    val_macc_by_epoch = {}
    early_bad_by_epoch = {}

    re_train = re.compile(
        r"Epoch\s+\[(\d+)\]\[\d+/\d+\].*decode\.loss_ce:\s*([0-9.]+).*decode\.acc_seg:\s*([0-9.]+).*loss:\s*([0-9.]+)"
    )

    re_val_loss = re.compile(
        r"\[ValLossEarlyStopHook\].*epoch=(\d+).*val_loss\(mean\)=([0-9.]+)"
    )

    re_early_noimp = re.compile(
        r"\[EarlyStoppingHook\].*no improvement:\s*(\d+)/(\d+)"
    )

    re_val_metrics = re.compile(
        r"Epoch\(val\)\s+\[(\d+)\]\[.*?\].*aAcc:\s*([0-9.]+).*mIoU:\s*([0-9.]+).*mAcc:\s*([0-9.]+)"
    )

    current_epoch_for_early = None

    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = re_train.search(line)
            if m:
                ep = int(m.group(1))
                tl = float(m.group(4))
                ta = _to_percent_if_needed(float(m.group(3)))
                last_train_loss_by_epoch[ep] = tl
                last_train_acc_by_epoch[ep] = ta
                continue

            m = re_val_loss.search(line)
            if m:
                ep = int(m.group(1))
                vl = float(m.group(2))
                val_loss_by_epoch[ep] = vl
                current_epoch_for_early = ep
                continue

            m = re_val_metrics.search(line)
            if m:
                ep = int(m.group(1))
                aacc = _to_percent_if_needed(float(m.group(2)))
                miou = _to_percent_if_needed(float(m.group(3)))
                macc = _to_percent_if_needed(float(m.group(4)))
                val_aacc_by_epoch[ep] = aacc
                val_miou_by_epoch[ep] = miou
                val_macc_by_epoch[ep] = macc
                continue

            m = re_early_noimp.search(line)
            if m and current_epoch_for_early is not None:
                bad = int(m.group(1))
                pat = int(m.group(2))
                early_bad_by_epoch[current_epoch_for_early] = (bad, pat)
                continue

    train_last = {
        ep: {"loss": last_train_loss_by_epoch[ep], "acc": last_train_acc_by_epoch[ep]}
        for ep in last_train_loss_by_epoch.keys()
    }

    val_loss = dict(val_loss_by_epoch)
    val_metrics = {
        ep: {
            "aAcc": val_aacc_by_epoch.get(ep),
            "mIoU": val_miou_by_epoch.get(ep),
            "mAcc": val_macc_by_epoch.get(ep),
        }
        for ep in set(val_aacc_by_epoch.keys()) | set(val_miou_by_epoch.keys()) | set(val_macc_by_epoch.keys())
    }

    return train_last, val_loss, val_metrics, early_bad_by_epoch


def pick_best_epoch_from_miou(val_metrics):
    best_ep = None
    best_miou = None
    for ep, d in val_metrics.items():
        miou = d.get("mIoU", None)
        if miou is None:
            continue
        if best_miou is None or miou > best_miou:
            best_miou = miou
            best_ep = ep
    return best_ep, best_miou


def plot_and_save(epochs, train_loss, train_acc, val_loss, val_acc, title, outdir):
    os.makedirs(outdir, exist_ok=True)

    fig, ax_loss = plt.subplots(figsize=(12, 8))
    ax_acc = ax_loss.twinx()

    l_train_acc, = ax_acc.plot(epochs, train_acc, linestyle=":", marker="o", label="Training Acc")
    l_val_acc, = ax_acc.plot(epochs, val_acc, linestyle="-.", marker="x", label="Validation Acc")

    l_train_loss, = ax_loss.plot(epochs, train_loss, linestyle="--", marker="^", label="Training Loss")
    l_val_loss, = ax_loss.plot(epochs, val_loss, linestyle="-", marker="s", label="Validation Loss")

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_acc.set_ylabel("Accuracy [%]")
    ax_acc.set_ylim(0, 100)
    ax_loss.set_title(title)
    ax_loss.grid(True, which="both", linestyle="--", alpha=0.3)

    lines = [l_train_acc, l_val_acc, l_train_loss, l_val_loss]
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    outpath = os.path.join(outdir, "training_curves.png")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    return outpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="Path to .log file")
    ap.add_argument("--json", default=None, help="Path to .log.json file (preferred)")
    ap.add_argument("--label", default="SCTNet-B-Seg100 RailSem19 Augmented - Loss and Accuracy")
    ap.add_argument("--outdir", default=".", help="Output directory")
    args = ap.parse_args()

    if args.json is None:
        cand = [f for f in os.listdir(".") if f.endswith(".log.json")]
        if cand:
            args.json = sorted(cand)[-1]
    if args.log is None:
        cand = [f for f in os.listdir(".") if f.endswith(".log")]
        if cand:
            args.log = sorted(cand)[-1]

    if args.json and os.path.isfile(args.json):
        train_last, val_loss, val_metrics, early_bad = parse_json_log(args.json)
    elif args.log and os.path.isfile(args.log):
        train_last, val_loss, val_metrics, early_bad = parse_text_log(args.log)
    else:
        raise FileNotFoundError("Could not find log files. Provide --json or --log.")

    epochs = sorted(set(train_last.keys()) & set(val_loss.keys()))
    if not epochs:
        epochs = sorted(train_last.keys())

    train_loss = []
    train_acc = []
    vloss = []
    vacc = []

    for ep in epochs:
        train_loss.append(train_last[ep]["loss"])
        train_acc.append(train_last[ep]["acc"])
        vloss.append(val_loss.get(ep, float("nan")))
        aAcc = val_metrics.get(ep, {}).get("aAcc", None)
        vacc.append(aAcc if aAcc is not None else float("nan"))

    best_ep, best_miou = pick_best_epoch_from_miou(val_metrics)
    outpath = plot_and_save(
        epochs, train_loss, train_acc, vloss, vacc,
        title=args.label, outdir=args.outdir
    )

    print("[OK] Saved:", os.path.abspath(outpath))
    if best_ep is not None and best_miou is not None:
        print(f"[INFO] Best epoch: {best_ep}")
        print(f"[INFO] Best mIoU : {best_miou:.2f}%")
    if early_bad:
        last_ep = max(early_bad.keys())
        bad, pat = early_bad[last_ep]
        print(f"[INFO] Last early-stop state seen: {bad}/{pat} at epoch {last_ep}")


if __name__ == "__main__":
    main()

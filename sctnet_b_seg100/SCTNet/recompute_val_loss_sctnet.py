#!/usr/bin/env python3
import os
import re
import csv
import glob
import argparse
from collections import OrderedDict

import torch
from mmcv import Config
from mmcv.parallel import scatter
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
from mmcv.cnn.utils import revert_sync_batchnorm


def checkpoint_epoch(path):
    m = re.search(r'epoch_(\d+)\.pth$', os.path.basename(path))
    return int(m.group(1)) if m else -1


@torch.no_grad()
def compute_val_loss(model, dataloader, device):
    model.eval()
    losses = []

    for data in dataloader:
        if device.type == "cuda":
            gpu_id = 0 if device.index is None else int(device.index)
            data = scatter(data, [gpu_id])[0]

        out = model(return_loss=True, **data)

        if isinstance(out, dict):
            total_loss = 0.0
            for value in out.values():
                if isinstance(value, (list, tuple)):
                    for v in value:
                        if hasattr(v, "mean"):
                            total_loss = total_loss + v.mean()
                else:
                    if hasattr(value, "mean"):
                        total_loss = total_loss + value.mean()
            total_loss = float(total_loss.item() if hasattr(total_loss, "item") else total_loss)
        else:
            total_loss = float(out.item() if hasattr(out, "item") else out)

        losses.append(total_loss)

    return sum(losses) / max(len(losses), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Config .py used for training")
    ap.add_argument("--workdir", required=True, help="Work dir containing epoch_*.pth")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--samples-per-gpu", type=int, default=1)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg.data.val)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=None,
        drop_last=False,
        persistent_workers=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpts = sorted(
        glob.glob(os.path.join(args.workdir, "epoch_*.pth")),
        key=checkpoint_epoch
    )
    if not ckpts:
        raise FileNotFoundError(f"No epoch_*.pth checkpoints found in {args.workdir}")

    rows = []
    for ckpt in ckpts:
        ep = checkpoint_epoch(ckpt)
        print(f"[INFO] Evaluating epoch {ep}: {ckpt}")

        model = build_segmentor(
            cfg.model,
            train_cfg=cfg.get("train_cfg"),
            test_cfg=cfg.get("test_cfg")
        )
        load_checkpoint(model, ckpt, map_location="cpu")

        if device.type == "cuda":
            model = revert_sync_batchnorm(model)
            model = model.to(device)
        else:
            model = model.cpu()

        val_loss = compute_val_loss(model, dataloader, device)
        rows.append((ep, val_loss))
        print(f"[INFO] epoch={ep} val_loss={val_loss:.6f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "val_loss"])
        for ep, vl in rows:
            w.writerow([ep, f"{vl:.6f}"])

    print(f"[OK] Wrote {args.out_csv}")


if __name__ == "__main__":
    main()

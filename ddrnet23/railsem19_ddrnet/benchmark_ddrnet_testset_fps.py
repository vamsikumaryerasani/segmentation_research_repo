#!/usr/bin/env python3
import time
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True

from models.ddrnet23 import get_ddrnet23
from mydatasets.new_seg_dataset import NewSegDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--num_classes", type=int, default=6)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = NewSegDataset(
        data_root=args.data_root,
        split_file=args.split_file,
        image_size=(args.width, args.height),
        augment=False,
        ignore_index=255,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = get_ddrnet23(num_classes=args.num_classes).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()

    total_images = 0

    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            bs = x.shape[0]
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                y = model(x)
                if isinstance(y, (list, tuple)):
                    y = y[0]
            total_images += bs

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    total_time = end - start
    fps = total_images / total_time
    ms_per_image = (total_time / total_images) * 1000

    print(f"Device: {device}")
    print(f"Total images: {total_images}")
    print(f"Batch size: {args.batch_size}")
    print(f"Input size: {args.height}x{args.width}")
    print(f"Total time: {total_time:.6f} s")
    print(f"Average time per image: {ms_per_image:.3f} ms")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    main()

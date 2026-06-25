#!/usr/bin/env python3
import os
import time
import argparse

import torch
from mmseg.apis import init_segmentor
from mmcv.parallel import scatter


def make_dummy_data(h, w, device):
    img = torch.randn(1, 3, h, w, device=device)
    img_metas = [[{
        'ori_shape': (h, w, 3),
        'img_shape': (h, w, 3),
        'pad_shape': (h, w, 3),
        'filename': 'dummy.jpg',
        'scale_factor': 1.0,
        'flip': False,
        'flip_direction': None,
        'img_norm_cfg': dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True
        )
    }]]
    return dict(img=[img], img_metas=img_metas)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    model.eval()

    data = make_dummy_data(args.height, args.width, device)

    # warmup
    for _ in range(args.warmup):
        if args.fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast():
                _ = model(return_loss=False, rescale=True, **data)
        else:
            _ = model(return_loss=False, rescale=True, **data)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(args.iters):
        if args.fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast():
                _ = model(return_loss=False, rescale=True, **data)
        else:
            _ = model(return_loss=False, rescale=True, **data)

    if device.type == "cuda":
        torch.cuda.synchronize()

    total = time.time() - start
    sec_per_img = total / args.iters
    fps = 1.0 / sec_per_img
    ms = sec_per_img * 1000.0

    print(f"Config      : {args.config}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Input size  : 1x3x{args.height}x{args.width}")
    print(f"FP16        : {args.fp16}")
    print(f"Iterations  : {args.iters}")
    print(f"Latency     : {ms:.3f} ms/image")
    print(f"FPS         : {fps:.3f}")


if __name__ == "__main__":
    main()

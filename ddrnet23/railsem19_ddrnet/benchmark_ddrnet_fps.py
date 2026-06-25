#!/usr/bin/env python3
import time
import argparse
import torch
from models.ddrnet23 import get_ddrnet23

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_classes", type=int, default=6)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = get_ddrnet23(num_classes=args.num_classes).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()

    x = torch.randn(1, 3, args.height, args.width, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            y = model(x)
            if isinstance(y, (list, tuple)):
                y = y[0]

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(args.iters):
            y = model(x)
            if isinstance(y, (list, tuple)):
                y = y[0]

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / args.iters
    fps = args.iters / total_time

    print(f"Device: {device}")
    print(f"Input size: 1x3x{args.height}x{args.width}")
    print(f"Total time: {total_time:.6f} s")
    print(f"Average time per image: {avg_time*1000:.3f} ms")
    print(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    main()

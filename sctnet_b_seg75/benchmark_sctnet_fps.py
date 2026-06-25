#!/usr/bin/env python3
import time
import argparse
import torch
torch.backends.cudnn.benchmark = True

from mmcv import Config
from mmseg.models import build_segmentor

def load_model_from_config(config_path, checkpoint_path, device):
    cfg = Config.fromfile(config_path)
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def run_forward(model, x, img_metas):
    if hasattr(model, "forward_dummy"):
        return model.forward_dummy(x)
    return model.encode_decode(x, img_metas=img_metas)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model_from_config(args.config, args.checkpoint, device)

    x = torch.randn(1, 3, args.height, args.width, device=device)
    img_metas = [{
        "img_shape": (args.height, args.width, 3),
        "ori_shape": (args.height, args.width, 3),
        "pad_shape": (args.height, args.width, 3),
        "scale_factor": 1.0,
        "flip": False
    }]

    with torch.inference_mode():
        for _ in range(args.warmup):
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                _ = run_forward(model, x, img_metas)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(args.iters):
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                _ = run_forward(model, x, img_metas)

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

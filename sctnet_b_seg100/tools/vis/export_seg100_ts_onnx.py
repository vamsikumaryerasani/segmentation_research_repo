#!/usr/bin/env python3
import os
import argparse
import torch

from mmseg.apis import init_segmentor
from mmcv import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--input-h", type=int, default=720)
    ap.add_argument("--input-w", type=int, default=1280)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) load model
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    model.eval()

    # 2) make dummy input (N,C,H,W)
    dummy = torch.randn(1, 3, args.input_h, args.input_w, device=args.device)

    # 3) wrap for ONNX: return logits tensor
    class Wrapper(torch.nn.Module):
        def __init__(self, segmentor):
            super().__init__()
            self.m = segmentor

        def forward(self, x):
            # In mmseg, model.encode_decode returns (N, num_classes, H, W)
            # Works for most encoder-decoder segmentors.
            return self.m.encode_decode(x, None)

    wrapped = Wrapper(model).to(args.device).eval()

    # -------- TorchScript (trace) --------
    ts_path = os.path.join(args.out_dir, "sctnet_b_seg100.ts.pt")
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, dummy)
        traced.save(ts_path)
    print("[OK] TorchScript saved:", ts_path)

    # -------- ONNX export --------
    onnx_path = os.path.join(args.out_dir, "sctnet_b_seg100.onnx")
    input_names = ["input"]
    output_names = ["logits"]

    dynamic_axes = {
        "input":  {0: "batch", 2: "height", 3: "width"},
        "logits": {0: "batch", 2: "height", 3: "width"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            onnx_path,
            opset_version=args.opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
    print("[OK] ONNX saved:", onnx_path)

    # -------- quick sanity print --------
    print("[INFO] Done. Files in out-dir:")
    for f in sorted(os.listdir(args.out_dir)):
        print(" -", f)


if __name__ == "__main__":
    main()

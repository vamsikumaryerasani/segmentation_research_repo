#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import mmcv
import torch

from mmseg.apis import init_segmentor, inference_segmentor


def colorize_mask(mask: np.ndarray, palette):
    """mask: (H,W) int64; palette: list of [R,G,B]"""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    pal = np.array(palette, dtype=np.uint8)

    # handle ignore index if any (commonly 255)
    valid = mask != 255
    color[valid] = pal[mask[valid]]
    # keep ignore as black
    return color


def make_side_by_side(img_bgr, mask_rgb, overlay_bgr):
    """Return a single BGR image: [img | mask | overlay]"""
    # convert mask_rgb -> mask_bgr for concatenation with BGR images
    mask_bgr = mask_rgb[..., ::-1]
    # match heights if needed
    h = img_bgr.shape[0]
    if mask_bgr.shape[0] != h:
        mask_bgr = mmcv.imresize(mask_bgr, (img_bgr.shape[1], h))
    if overlay_bgr.shape[0] != h:
        overlay_bgr = mmcv.imresize(overlay_bgr, (img_bgr.shape[1], h))
    return np.concatenate([img_bgr, mask_bgr, overlay_bgr], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to config .py")
    ap.add_argument("checkpoint", help="path to checkpoint .pth")
    ap.add_argument("--in-dir", required=True, help="folder with input images")
    ap.add_argument("--out-dir", required=True, help="output folder")
    ap.add_argument("--glob", default="*.jpg", help="glob pattern, e.g. *.png")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    ap.add_argument("--opacity", type=float, default=0.5, help="overlay opacity")
    ap.add_argument("--max-images", type=int, default=30, help="limit number of saved images")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_mask_dir = os.path.join(args.out_dir, "mask")
    out_overlay_dir = os.path.join(args.out_dir, "overlay")
    out_side_dir = os.path.join(args.out_dir, "side_by_side")
    os.makedirs(out_mask_dir, exist_ok=True)
    os.makedirs(out_overlay_dir, exist_ok=True)
    os.makedirs(out_side_dir, exist_ok=True)

    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    model.eval()

    # palette + classes from model (mmseg sets these if dataset config is correct)
    palette = getattr(model, "PALETTE", None)
    if palette is None:
        raise RuntimeError("Model has no PALETTE. Ensure dataset config defines a palette (classes/colors).")

    img_paths = sorted(glob.glob(os.path.join(args.in_dir, args.glob)))
    if not img_paths:
        raise RuntimeError(f"No images found in {args.in_dir} with pattern {args.glob}")

    img_paths = img_paths[: args.max_images]

    print(f"Found {len(img_paths)} images. Saving to: {args.out_dir}")
    for p in img_paths:
        name = os.path.splitext(os.path.basename(p))[0]

        img = mmcv.imread(p)  # BGR
        result = inference_segmentor(model, img)  # list with 1 mask
        mask = result[0].astype(np.int64)

        # 1) colored mask (RGB)
        mask_rgb = colorize_mask(mask, palette)
        mmcv.imwrite(mask_rgb[..., ::-1], os.path.join(out_mask_dir, f"{name}_mask.png"))  # save as BGR

        # 2) overlay
        overlay_path = os.path.join(out_overlay_dir, f"{name}_overlay.png")
        model.show_result(
            img,
            result,
            out_file=overlay_path,
            opacity=args.opacity,
            show=False
        )
        overlay = mmcv.imread(overlay_path)  # BGR

        # 3) side-by-side: original | mask | overlay
        side = make_side_by_side(img, mask_rgb, overlay)
        mmcv.imwrite(side, os.path.join(out_side_dir, f"{name}_side.png"))

        print("Saved:", name)

    print("\nDone.")
    print("Outputs:")
    print(" -", out_mask_dir)
    print(" -", out_overlay_dir)
    print(" -", out_side_dir)


if __name__ == "__main__":
    main()

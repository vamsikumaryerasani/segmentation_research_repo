#!/usr/bin/env python3
import os
import json
import shutil
import argparse
import numpy as np
import cv2

from mmseg.apis import init_segmentor, inference_segmentor


def load_rs19_palette(rs19_config_path):
    with open(rs19_config_path, "r") as f:
        cfg = json.load(f)

    labels = cfg["labels"][:19]
    class_names = [x["name"] for x in labels]
    palette_rgb = [x["color"] for x in labels]   # RGB from json
    return class_names, palette_rgb


def colorize_mask(mask, palette_rgb):
    """
    Input:
      mask: HxW uint8 with class ids 0..18 (ignore 255 optional)
      palette_rgb: list of 19 RGB colors
    Output:
      color image in BGR (for OpenCV writing)
    """
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for idx, rgb in enumerate(palette_rgb):
        bgr = (rgb[2], rgb[1], rgb[0])
        out[mask == idx] = bgr

    # ignore label 255 -> black
    out[mask == 255] = (0, 0, 0)
    return out


def find_gt_mask(img_path, gt_dir):
    base = os.path.splitext(os.path.basename(img_path))[0]
    gt_path = os.path.join(gt_dir, base + ".png")
    return gt_path if os.path.isfile(gt_path) else None


def make_triptych(img_bgr, gt_bgr, pred_bgr,
                  labels=("Original image", "RGB ground truth", "RGB predicted mask")):
    h, w = img_bgr.shape[:2]
    header_h = 50
    canvas = np.full((header_h + h, 3 * w, 3), 255, dtype=np.uint8)

    canvas[header_h:header_h + h, 0:w] = img_bgr
    canvas[header_h:header_h + h, w:2 * w] = gt_bgr
    canvas[header_h:header_h + h, 2 * w:3 * w] = pred_bgr

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thick = 2
    color = (0, 0, 0)

    centers = [w // 2, w + w // 2, 2 * w + w // 2]
    for c, txt in zip(centers, labels):
        (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
        x = c - tw // 2
        y = 32
        cv2.putText(canvas, txt, (x, y), font, scale, color, thick, cv2.LINE_AA)

    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="Model config .py")
    ap.add_argument("checkpoint", help="Model checkpoint .pth")
    ap.add_argument("--images", nargs="+", required=True, help="Input image paths")
    ap.add_argument("--gt-dir", required=True, help="Ground-truth mask folder")
    ap.add_argument("--rs19-config", required=True, help="Path to rs19-config.json")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    class_names, palette_rgb = load_rs19_palette(args.rs19_config)

    print("[INFO] Classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    print("[INFO] Loading model...")
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    # attach classes/palette for clarity
    model.CLASSES = class_names
    model.PALETTE = palette_rgb

    for idx, img_path in enumerate(args.images, 1):
        if not os.path.isfile(img_path):
            print(f"[WARN] Image not found: {img_path}")
            continue

        base = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = find_gt_mask(img_path, args.gt_dir)
        if gt_path is None:
            print(f"[WARN] GT not found for: {img_path}")
            continue

        print(f"[{idx}/{len(args.images)}] Processing {base}")

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt is None:
            print(f"[WARN] Could not read GT mask: {gt_path}")
            continue
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        gt = gt.astype(np.uint8)

        # prediction
        result = inference_segmentor(model, img_path)
        pred = result[0] if isinstance(result, (list, tuple)) else result
        pred = pred.astype(np.uint8)

        h, w = img_bgr.shape[:2]
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        if gt.shape != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

        gt_bgr = colorize_mask(gt, palette_rgb)
        pred_bgr = colorize_mask(pred, palette_rgb)

        sample_dir = os.path.join(args.out_dir, base)
        os.makedirs(sample_dir, exist_ok=True)

        # Save full-resolution outputs
        orig_out = os.path.join(sample_dir, f"{base}_original.png")
        gt_out = os.path.join(sample_dir, f"{base}_gt_rgb.png")
        pred_out = os.path.join(sample_dir, f"{base}_pred_rgb.png")
        tri_out = os.path.join(sample_dir, f"{base}_triptych_rgb.png")

        cv2.imwrite(orig_out, img_bgr)
        cv2.imwrite(gt_out, gt_bgr)
        cv2.imwrite(pred_out, pred_bgr)

        tri = make_triptych(img_bgr, gt_bgr, pred_bgr)
        cv2.imwrite(tri_out, tri)

        print(f"  saved: {orig_out}")
        print(f"  saved: {gt_out}")
        print(f"  saved: {pred_out}")
        print(f"  saved: {tri_out}")

    print(f"\n[OK] Done. Outputs saved in: {args.out_dir}")


if __name__ == "__main__":
    main()

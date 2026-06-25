#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import cv2

from mmseg.apis import init_segmentor, inference_segmentor

# FORCE RailSem19 palette (RGB) -> converted to BGR for OpenCV
RAILSEM19_PALETTE_RGB = [
    [128, 64,128],  # 0
    [244, 35,232],  # 1
    [70, 70, 70],   # 2
    [102,102,156],  # 3
    [190,153,153],  # 4
    [153,153,153],  # 5
    [250,170, 30],  # 6
    [220,220,  0],  # 7
    [107,142, 35],  # 8
    [152,251,152],  # 9
    [70,130,180],   # 10
    [220, 20, 60],  # 11
    [255,  0,  0],  # 12
    [  0,  0,142],  # 13
    [  0,  0, 70],  # 14
    [  0, 60,100],  # 15
    [  0, 80,100],  # 16
    [  0,  0,230],  # 17
    [119, 11, 32],  # 18
]

IGNORE_LABEL = 255


def list_images(in_dir):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    imgs = []
    for e in exts:
        imgs += sorted(glob.glob(os.path.join(in_dir, e)))
    return imgs


def find_gt_mask_by_basename(img_path, gt_dir):
    base = os.path.splitext(os.path.basename(img_path))[0]
    gt_path = os.path.join(gt_dir, base + ".png")
    return gt_path if os.path.isfile(gt_path) else None


def find_raw_image_by_basename(img_path, raw_dir):
    """
    Example:
      rs00029_bri0p5.jpg -> rs00029.jpg
      rs00029_con0p7.jpg -> rs00029.jpg
    """
    base = os.path.splitext(os.path.basename(img_path))[0]

    root = base
    for tag in ["_bri", "_con"]:
        if tag in root:
            root = root.split(tag, 1)[0]
            break

    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
        cand = os.path.join(raw_dir, root + ext)
        if os.path.isfile(cand):
            return cand
    return None


def to_color(mask):
    """Use ONLY the fixed RailSem19 palette."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for lab, rgb in enumerate(RAILSEM19_PALETTE_RGB):
        bgr = (rgb[2], rgb[1], rgb[0])
        out[mask == lab] = bgr
    out[mask == IGNORE_LABEL] = (0, 0, 0)
    return out


def to_black_gray(mask):
    g = mask.astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def resize_nn(mask, shape_hw):
    h, w = shape_hw
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def make_triptych(left, mid, right,
                  labels=("Original image", "Predicted mask", "Ground truth mask")):
    h, w = left.shape[:2]

    if mid.shape[:2] != (h, w):
        mid = resize_nn(mid, (h, w))
    if right.shape[:2] != (h, w):
        right = resize_nn(right, (h, w))

    header_h = 36
    canvas = np.full((header_h + h, 3 * w, 3), 255, dtype=np.uint8)

    canvas[header_h:header_h + h, 0:w] = left
    canvas[header_h:header_h + h, w:2 * w] = mid
    canvas[header_h:header_h + h, 2 * w:3 * w] = right

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.5
    th = 1
    color = (0, 0, 0)
    xs = [w // 2, w + w // 2, 2 * w + w // 2]

    for x, txt in zip(xs, labels):
        (tw, _), _ = cv2.getTextSize(txt, font, fs, th)
        cv2.putText(canvas, txt, (x - tw // 2, 22), font, fs, color, th, cv2.LINE_AA)

    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="Path to mmseg config .py")
    ap.add_argument("checkpoint", help="Path to checkpoint .pth")
    ap.add_argument("--in-dir", required=True, help="Input images directory (used for inference)")
    ap.add_argument("--raw-dir", required=True, help="Original raw RGB images directory (used for LEFT panel)")
    ap.add_argument("--gt-dir", required=True, help="Ground-truth masks directory")
    ap.add_argument("--out-dir", required=True, help="Output base directory")
    ap.add_argument("--max-images", type=int, default=10, help="Number of images to process")
    ap.add_argument("--device", default="cuda:0", help="Device (e.g. cuda:0)")
    args = ap.parse_args()

    out_black = os.path.join(args.out_dir, "triptych_black")
    out_color = os.path.join(args.out_dir, "triptych_color")
    os.makedirs(out_black, exist_ok=True)
    os.makedirs(out_color, exist_ok=True)

    for p in [args.in_dir, args.raw_dir, args.gt_dir]:
        if not os.path.isdir(p):
            raise SystemExit(f"[ERROR] Directory not found: {p}")

    imgs = list_images(args.in_dir)
    imgs = imgs[:args.max_images]
    if not imgs:
        raise SystemExit(f"[ERROR] No images found in: {args.in_dir}")

    print("[INFO] Initializing model...")
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    n_done = 0
    for img_path in imgs:
        # LEFT panel = original raw RGB frame
        raw_path = find_raw_image_by_basename(img_path, args.raw_dir)
        if raw_path is None:
            print("[WARN] Raw image not found for:", os.path.basename(img_path))
            continue

        raw_img = cv2.imread(raw_path, cv2.IMREAD_COLOR)
        if raw_img is None:
            print("[WARN] Cannot read raw image:", raw_path)
            continue

        gt_path = find_gt_mask_by_basename(img_path, args.gt_dir)
        if not gt_path:
            print("[WARN] GT not found for:", os.path.basename(img_path))
            continue

        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt is None:
            print("[WARN] Cannot read GT:", gt_path)
            continue
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt = gt.astype(np.uint8)

        # Inference on the augmented/current image
        result = inference_segmentor(model, img_path)
        pred = result[0] if isinstance(result, (list, tuple)) else result
        pred = pred.astype(np.uint8)

        h, w = raw_img.shape[:2]
        if pred.shape != (h, w):
            pred = resize_nn(pred, (h, w))
        if gt.shape != (h, w):
            gt = resize_nn(gt, (h, w))

        # SAME fixed palette for pred + GT
        pred_color = to_color(pred)
        gt_color = to_color(gt)
        pred_black = to_black_gray(pred)
        gt_black = to_black_gray(gt)

        tri_color = make_triptych(raw_img, pred_color, gt_color)
        tri_black = make_triptych(raw_img, pred_black, gt_black)

        token = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(out_color, f"{token}_triptych_color.png"), tri_color)
        cv2.imwrite(os.path.join(out_black, f"{token}_triptych_black.png"), tri_black)

        n_done += 1
        print(f"[{n_done}/{len(imgs)}] Wrote: {token}")

    print(f"[OK] Wrote {n_done} color triptychs -> {out_color}")
    print(f"[OK] Wrote {n_done} black triptychs -> {out_black}")


if __name__ == "__main__":
    main()

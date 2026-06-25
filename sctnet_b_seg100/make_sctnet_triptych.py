#!/usr/bin/env python3
import os
import glob
import random
import argparse
import numpy as np
import cv2

from mmseg.apis import init_segmentor, inference_segmentor

# 6-class palette (RGB)
# 0 = sky
# 1 = terrain
# 2 = nature
# 3 = car
# 4 = building
# 5 = railway
PALETTE_RGB = [
    [70, 130, 180],   # sky
    [152, 251, 152],  # terrain
    [107, 142, 35],   # nature
    [0, 0, 142],      # car
    [70, 70, 70],     # building
    [230, 150, 140],  # railway
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


def sanitize_mask(mask):
    mask = mask.copy().astype(np.uint8)
    invalid = ~np.isin(mask, [0, 1, 2, 3, 4, 5, IGNORE_LABEL])
    mask[invalid] = IGNORE_LABEL
    return mask


def to_color(mask):
    mask = sanitize_mask(mask)
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for lab, rgb in enumerate(PALETTE_RGB):
        bgr = (rgb[2], rgb[1], rgb[0])
        out[mask == lab] = bgr
    out[mask == IGNORE_LABEL] = (0, 0, 0)
    return out


def to_black_gray(mask):
    mask = sanitize_mask(mask)
    g = np.zeros_like(mask, dtype=np.uint8)
    g[mask == 0] = 0
    g[mask == 1] = 10
    g[mask == 2] = 20
    g[mask == 3] = 30
    g[mask == 4] = 40
    g[mask == 5] = 50
    g[mask == IGNORE_LABEL] = 255
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

    header_h = 44
    canvas = np.full((header_h + h, 3 * w, 3), 255, dtype=np.uint8)

    canvas[header_h:header_h + h, 0:w] = left
    canvas[header_h:header_h + h, w:2 * w] = mid
    canvas[header_h:header_h + h, 2 * w:3 * w] = right

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.7
    th = 2
    color = (0, 0, 0)
    xs = [w // 2, w + w // 2, 2 * w + w // 2]

    for x, txt in zip(xs, labels):
        (tw, _), _ = cv2.getTextSize(txt, font, fs, th)
        cv2.putText(canvas, txt, (x - tw // 2, 30), font, fs, color, th, cv2.LINE_AA)

    return canvas


def load_model(config_path, checkpoint_path, device):
    model = init_segmentor(config_path, checkpoint_path, device=device)
    return model


def infer_mask(model, img_path):
    result = inference_segmentor(model, img_path)
    if isinstance(result, (list, tuple)):
        pred = result[0]
    else:
        pred = result
    return pred.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to SCTNet config .py")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth")
    ap.add_argument("--in-dir", required=True, help="Input images directory")
    ap.add_argument("--gt-dir", required=True, help="Ground-truth masks directory")
    ap.add_argument("--out-dir", required=True, help="Output base directory")
    ap.add_argument("--max-images", type=int, default=10, help="Number of images to process")
    ap.add_argument("--device", default="cuda:0", help="Device (e.g. cuda:0)")
    ap.add_argument("--random-sample", action="store_true", help="Randomly sample images instead of first N")
    ap.add_argument("--seed", type=int, default=42, help="Random seed when using --random-sample")
    args = ap.parse_args()

    out_black = os.path.join(args.out_dir, "triptych_black")
    out_color = os.path.join(args.out_dir, "triptych_color")
    os.makedirs(out_black, exist_ok=True)
    os.makedirs(out_color, exist_ok=True)

    if not os.path.isdir(args.in_dir):
        raise SystemExit(f"[ERROR] Input dir not found: {args.in_dir}")
    if not os.path.isdir(args.gt_dir):
        raise SystemExit(f"[ERROR] GT dir not found: {args.gt_dir}")

    imgs = list_images(args.in_dir)
    if not imgs:
        raise SystemExit(f"[ERROR] No images found in: {args.in_dir}")

    if args.random_sample:
        random.seed(args.seed)
        random.shuffle(imgs)

    imgs = imgs[:args.max_images]

    print("[INFO] Loading model...")
    model = load_model(args.config, args.checkpoint, device=args.device)

    n_done = 0
    for img_path in imgs:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print("[WARN] Cannot read image:", img_path)
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
        gt = sanitize_mask(gt)

        pred = infer_mask(model, img_path)
        pred = sanitize_mask(pred)

        h, w = img.shape[:2]
        if pred.shape != (h, w):
            pred = resize_nn(pred, (h, w))
        if gt.shape != (h, w):
            gt = resize_nn(gt, (h, w))

        pred_black = to_black_gray(pred)
        gt_black = to_black_gray(gt)
        pred_color = to_color(pred)
        gt_color = to_color(gt)

        tri_black = make_triptych(img, pred_black, gt_black)
        tri_color = make_triptych(img, pred_color, gt_color)

        token = os.path.splitext(os.path.basename(img_path))[0]
        black_out = os.path.join(out_black, f"{token}_triptych_black.png")
        color_out = os.path.join(out_color, f"{token}_triptych_color.png")

        cv2.imwrite(black_out, tri_black)
        cv2.imwrite(color_out, tri_color)

        n_done += 1
        print(f"[{n_done}/{len(imgs)}] Wrote: {token}")

    print(f"[OK] Wrote {n_done} black triptychs -> {out_black}")
    print(f"[OK] Wrote {n_done} color triptychs -> {out_color}")


if __name__ == "__main__":
    main()

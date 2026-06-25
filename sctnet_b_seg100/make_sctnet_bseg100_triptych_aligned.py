#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import cv2

from mmseg.apis import init_segmentor, inference_segmentor

RAILSEM19_PALETTE_RGB = [
    [128, 64,128], [244, 35,232], [70, 70, 70], [102,102,156], [190,153,153],
    [153,153,153], [250,170, 30], [220,220,  0], [107,142, 35], [152,251,152],
    [70,130,180], [220, 20, 60], [255,  0,  0], [  0,  0,142], [  0,  0, 70],
    [  0, 60,100], [  0, 80,100], [  0,  0,230], [119, 11, 32],
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


def to_color(mask):
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


def fit_img(img, shape_hw, is_mask=False):
    h, w = shape_hw
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def make_triptych(left, mid, right,
                  labels=("Original image", "Predicted mask", "ground truth mask")):
    h, w = left.shape[:2]
    mid = fit_img(mid, (h, w), is_mask=True)
    right = fit_img(right, (h, w), is_mask=True)

    header_h = 32
    canvas = np.full((header_h + h, 3 * w, 3), 255, dtype=np.uint8)

    canvas[header_h:header_h+h, 0:w]     = left
    canvas[header_h:header_h+h, w:2*w]   = mid
    canvas[header_h:header_h+h, 2*w:3*w] = right

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45
    th = 1
    color = (0, 0, 0)
    xs = [w // 2, w + w // 2, 2 * w + w // 2]

    for x, txt in zip(xs, labels):
        (tw, _), _ = cv2.getTextSize(txt, font, fs, th)
        cv2.putText(canvas, txt, (x - tw // 2, 20), font, fs, color, th, cv2.LINE_AA)

    return canvas


def title_prefix_from_filename(sample_path):
    name = os.path.splitext(os.path.basename(sample_path))[0]

    if "_bri" in name:
        val = name.split("_bri", 1)[1].replace("p", ".")
        return f"Brightness-{val}"

    if "_con" in name:
        val = name.split("_con", 1)[1].replace("p", ".")
        return f"Contrast-{val}"

    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="config .py")
    ap.add_argument("checkpoint", help="checkpoint .pth")
    ap.add_argument("--in-dir", required=True, help="sample image dir (same images used for inference + display)")
    ap.add_argument("--gt-dir", required=True, help="ground truth mask dir")
    ap.add_argument("--out-dir", required=True, help="output dir")
    ap.add_argument("--max-images", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_color = os.path.join(args.out_dir, "color")
    out_gray = os.path.join(args.out_dir, "gray")
    os.makedirs(out_color, exist_ok=True)
    os.makedirs(out_gray, exist_ok=True)

    for p in [args.in_dir, args.gt_dir]:
        if not os.path.isdir(p):
            raise SystemExit(f"[ERROR] directory not found: {p}")

    imgs = list_images(args.in_dir)[:args.max_images]
    if not imgs:
        raise SystemExit(f"[ERROR] No images found in: {args.in_dir}")

    print("[INFO] Initializing model...")
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    n_done = 0
    for img_path in imgs:
        gt_path = find_gt_mask_by_basename(img_path, args.gt_dir)
        if gt_path is None:
            print("[WARN] GT not found for:", os.path.basename(img_path))
            continue

        # LEFT PANEL = exact same sample image used for inference
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print("[WARN] Cannot read image:", img_path)
            continue

        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt is None:
            print("[WARN] Cannot read GT:", gt_path)
            continue
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt = gt.astype(np.uint8)

        # Inference on this exact same image
        result = inference_segmentor(model, img_path)
        pred = result[0] if isinstance(result, (list, tuple)) else result
        pred = pred.astype(np.uint8)

        h, w = img.shape[:2]
        if pred.shape != (h, w):
            pred = resize_nn(pred, (h, w))
        if gt.shape != (h, w):
            gt = resize_nn(gt, (h, w))

        pred_col = to_color(pred)
        gt_col   = to_color(gt)
        pred_gray = to_black_gray(pred)
        gt_gray   = to_black_gray(gt)

        strip_color = make_triptych(img, pred_col, gt_col)
        strip_gray  = make_triptych(img, pred_gray, gt_gray)

        title = title_prefix_from_filename(img_path)
        token = os.path.splitext(os.path.basename(img_path))[0]

        out_color_path = os.path.join(out_color, f"{title}_{token}_color.png")
        out_gray_path  = os.path.join(out_gray,  f"{title}_{token}_gray.png")

        cv2.imwrite(out_color_path, strip_color)
        cv2.imwrite(out_gray_path, strip_gray)

        n_done += 1
        print(f"[{n_done}/{len(imgs)}] wrote {token}")

    print(f"[OK] wrote {n_done} color strips -> {out_color}")
    print(f"[OK] wrote {n_done} gray strips  -> {out_gray}")


if __name__ == "__main__":
    main()

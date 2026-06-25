#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import cv2

from mmseg.apis import init_segmentor, inference_segmentor

# RailSem19 palette (RGB) -> convert to BGR for OpenCV
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
    # Dark grayscale class-id style
    g = mask.astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def resize_nn(mask, shape_hw):
    h, w = shape_hw
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def fit_to_size(img, target_hw, is_mask=False):
    """Resize to exact target size."""
    th, tw = target_hw
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(img, (tw, th), interpolation=interp)


def make_two_row_panel(raw_img, pred_color, gt_color, pred_gray, gt_gray, title_text):
    """
    Final layout like your example:
      title
      row 1: raw | pred color | gt color
      row 2: raw | pred gray  | gt gray
    """
    h, w = raw_img.shape[:2]

    # Ensure all panels same size
    pred_color = fit_to_size(pred_color, (h, w), is_mask=True)
    gt_color   = fit_to_size(gt_color,   (h, w), is_mask=True)
    pred_gray  = fit_to_size(pred_gray,  (h, w), is_mask=True)
    gt_gray    = fit_to_size(gt_gray,    (h, w), is_mask=True)

    title_h = 42
    header_h = 28
    gap_h = 20
    canvas_h = title_h + header_h + h + gap_h + header_h + h
    canvas_w = 3 * w

    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # Title
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title_text, (10, 28), font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    # Row 1 header
    row1_y = title_h
    row1_img_y = row1_y + header_h

    # Row 2 header
    row2_y = row1_img_y + h + gap_h
    row2_img_y = row2_y + header_h

    labels = ["Original image", "Predicted mask", "ground truth mask"]
    xs = [w // 2, w + w // 2, 2 * w + w // 2]

    for x, txt in zip(xs, labels):
        (tw, _), _ = cv2.getTextSize(txt, font, 0.45, 1)
        cv2.putText(canvas, txt, (x - tw // 2, row1_y + 18), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, txt, (x - tw // 2, row2_y + 18), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Row 1: color
    canvas[row1_img_y:row1_img_y+h, 0:w]     = raw_img
    canvas[row1_img_y:row1_img_y+h, w:2*w]   = pred_color
    canvas[row1_img_y:row1_img_y+h, 2*w:3*w] = gt_color

    # Row 2: grayscale
    canvas[row2_img_y:row2_img_y+h, 0:w]     = raw_img
    canvas[row2_img_y:row2_img_y+h, w:2*w]   = pred_gray
    canvas[row2_img_y:row2_img_y+h, 2*w:3*w] = gt_gray

    return canvas


def title_from_filename(img_path):
    """
    Example:
      rs00029_bri0p5.jpg -> Brightness-0.5
      rs00029_bri0p6.jpg -> Brightness-0.6
      rs00029_con0p7.jpg -> Contrast-0.7
      otherwise -> basename
    """
    name = os.path.splitext(os.path.basename(img_path))[0]

    if "_bri" in name:
        val = name.split("_bri", 1)[1]
        val = val.replace("p", ".")
        return f"Brightness-{val}"

    if "_con" in name:
        val = name.split("_con", 1)[1]
        val = val.replace("p", ".")
        return f"Contrast-{val}"

    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="config .py")
    ap.add_argument("checkpoint", help="checkpoint .pth")
    ap.add_argument("--in-dir", required=True, help="input images dir")
    ap.add_argument("--gt-dir", required=True, help="ground truth masks dir")
    ap.add_argument("--out-dir", required=True, help="output dir")
    ap.add_argument("--max-images", type=int, default=5, help="number of images to process")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.isdir(args.in_dir):
        raise SystemExit(f"[ERROR] in-dir not found: {args.in_dir}")
    if not os.path.isdir(args.gt_dir):
        raise SystemExit(f"[ERROR] gt-dir not found: {args.gt_dir}")

    imgs = list_images(args.in_dir)
    imgs = imgs[:args.max_images]

    if not imgs:
        raise SystemExit(f"[ERROR] No images found in: {args.in_dir}")

    print("[INFO] Initializing model...")
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

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
        gt = gt.astype(np.uint8)

        result = inference_segmentor(model, img_path)
        pred = result[0] if isinstance(result, (list, tuple)) else result
        pred = pred.astype(np.uint8)

        h, w = img.shape[:2]
        if pred.shape != (h, w):
            pred = resize_nn(pred, (h, w))
        if gt.shape != (h, w):
            gt = resize_nn(gt, (h, w))

        pred_col  = to_color(pred)
        gt_col    = to_color(gt)
        pred_gray = to_black_gray(pred)
        gt_gray   = to_black_gray(gt)

        title_text = title_from_filename(img_path)
        panel = make_two_row_panel(img, pred_col, gt_col, pred_gray, gt_gray, title_text)

        token = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.out_dir, f"{token}_panel.png")
        cv2.imwrite(out_path, panel)

        n_done += 1
        print(f"[{n_done}/{len(imgs)}] wrote {out_path}")

    print(f"[OK] wrote {n_done} panels -> {args.out_dir}")


if __name__ == "__main__":
    main()

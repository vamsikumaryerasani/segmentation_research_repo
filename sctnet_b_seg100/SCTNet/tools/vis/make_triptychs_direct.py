#!/usr/bin/env python3
import os, glob, argparse
import numpy as np
import cv2

from mmseg.apis import init_segmentor, inference_segmentor

# RailSem19 palette (RGB) -> we’ll convert to BGR for OpenCV when applying
RAILSEM19_PALETTE_RGB = [
    [128, 64,128],  # 0: Road
    [244, 35,232],  # 1: Sidewalk
    [70, 70, 70],   # 2: Building
    [102,102,156],  # 3: Wall
    [190,153,153],  # 4: Fence
    [153,153,153],  # 5: Pole
    [250,170, 30],  # 6: Traffic Light
    [220,220,  0],  # 7: Traffic Sign
    [107,142, 35],  # 8: Vegetation
    [152,251,152],  # 9: Terrain
    [70,130,180],   #10: Sky
    [220, 20, 60],  #11: Person
    [255,  0,  0],  #12: Rider
    [  0,  0,142],  #13: Car
    [  0,  0, 70],  #14: Truck
    [  0, 60,100],  #15: Bus
    [  0, 80,100],  #16: Train
    [  0,  0,230],  #17: Motorcycle
    [119, 11, 32],  #18: Bicycle
]

def list_images(in_dir):
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.webp")
    imgs = []
    for e in exts:
        imgs += sorted(glob.glob(os.path.join(in_dir, e)))
    return imgs

def find_gt_mask(img_path, mask_root):
    token = os.path.splitext(os.path.basename(img_path))[0]
    cands = []
    for e in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.webp"):
        cands += glob.glob(os.path.join(mask_root, "**", token + os.path.splitext(e)[0].replace("*","") ), recursive=True)
    # The above is not reliable across extensions; do robust glob:
    cands = glob.glob(os.path.join(mask_root, "**", token + ".*"), recursive=True)
    cands = [p for p in cands if os.path.isfile(p)]
    return cands[0] if cands else None

def to_color(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    # ignore label 255 -> black
    for lab, rgb in enumerate(RAILSEM19_PALETTE_RGB):
        bgr = (rgb[2], rgb[1], rgb[0])
        out[mask == lab] = bgr
    out[mask == 255] = (0,0,0)
    return out

def to_black_gray(mask):
    """
    This makes the “black annotated” look:
    keep RAW class IDs (0..18) without scaling to 0..255.
    That stays dark/black, with subtle structure.
    Keep 255 as 255 (white) if present.
    """
    g = mask.astype(np.uint8)  # 0..18 (dark) and maybe 255
    g3 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return g3

def resize_nn(mask, shape_hw):
    h, w = shape_hw
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

def make_triptych(left, mid, right, labels=("Original images","Predicted outputs","ground-truth masks")):
    # all BGR, same size
    h, w = left.shape[:2]
    header_h = 44
    canvas = np.full((header_h + h, 3*w, 3), 255, dtype=np.uint8)

    canvas[header_h:header_h+h, 0:w]     = left
    canvas[header_h:header_h+h, w:2*w]   = mid
    canvas[header_h:header_h+h, 2*w:3*w] = right

    # header text
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.7
    th = 2
    color = (0,0,0)
    xs = [w//2, w + w//2, 2*w + w//2]
    for x, txt in zip(xs, labels):
        (tw, _), _ = cv2.getTextSize(txt, font, fs, th)
        cv2.putText(canvas, txt, (x - tw//2, 30), font, fs, color, th, cv2.LINE_AA)

    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="config .py")
    ap.add_argument("checkpoint", help="checkpoint .pth")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_black = os.path.join(args.out_dir, "triptych_black")
    out_color = os.path.join(args.out_dir, "triptych_color")
    os.makedirs(out_black, exist_ok=True)
    os.makedirs(out_color, exist_ok=True)

    # mask root from your config expectation
    mask_root = "data/railsem19_mmseg_90_10/masks/val"
    if not os.path.isdir(mask_root):
        raise SystemExit(f"[ERROR] mask_root not found: {mask_root}")

    imgs = list_images(args.in_dir)[:args.max_images]
    if not imgs:
        raise SystemExit(f"[ERROR] No images found in: {args.in_dir}")

    print("[INFO] init model...")
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    n_done = 0
    for img_path in imgs:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            print("[WARN] cannot read:", img_path)
            continue

        gt_path = find_gt_mask(img_path, mask_root)
        if not gt_path:
            print("[WARN] GT not found for:", os.path.basename(img_path))
            continue

        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt is None:
            print("[WARN] cannot read GT:", gt_path)
            continue
        if gt.ndim == 3:
            gt = gt[...,0]
        gt = gt.astype(np.uint8)

        # inference
        result = inference_segmentor(model, img_path)
        pred = result[0] if isinstance(result, (list, tuple)) else result
        pred = pred.astype(np.uint8)

        h, w = img.shape[:2]
        if pred.shape != (h, w):
            pred = resize_nn(pred, (h, w))
        if gt.shape != (h, w):
            gt = resize_nn(gt, (h, w))

        # build visuals
        pred_black = to_black_gray(pred)
        gt_black   = to_black_gray(gt)
        pred_col   = to_color(pred)
        gt_col     = to_color(gt)

        # triptychs
        tri_black = make_triptych(img, pred_black, gt_black)
        tri_color = make_triptych(img, pred_col, gt_col)

        token = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(out_black, f"{token}_triptych_black.png"), tri_black)
        cv2.imwrite(os.path.join(out_color, f"{token}_triptych_color.png"), tri_color)

        n_done += 1

    print(f"[OK] wrote {n_done} black triptychs -> {out_black}")
    print(f"[OK] wrote {n_done} color triptychs -> {out_color}")

if __name__ == "__main__":
    main()

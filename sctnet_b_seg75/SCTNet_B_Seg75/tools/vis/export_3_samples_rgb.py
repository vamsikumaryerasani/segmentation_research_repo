#!/usr/bin/env python3
import os
import argparse
import numpy as np
import cv2
from mmseg.apis import init_segmentor, inference_segmentor

# RailSem19 palette (RGB) -> convert to BGR for OpenCV
RAILSEM19_PALETTE_RGB = [
    [128,  64, 128],  # 0 road
    [244,  35, 232],  # 1 sidewalk
    [ 70,  70,  70],  # 2 building
    [102, 102, 156],  # 3 wall
    [190, 153, 153],  # 4 fence
    [153, 153, 153],  # 5 pole
    [250, 170,  30],  # 6 traffic light
    [220, 220,   0],  # 7 traffic sign
    [107, 142,  35],  # 8 vegetation
    [152, 251, 152],  # 9 terrain
    [ 70, 130, 180],  # 10 sky
    [220,  20,  60],  # 11 person
    [255,   0,   0],  # 12 rider
    [  0,   0, 142],  # 13 car
    [  0,   0,  70],  # 14 truck
    [  0,  60, 100],  # 15 bus
    [  0,  80, 100],  # 16 train
    [  0,   0, 230],  # 17 motorcycle
    [119,  11,  32],  # 18 bicycle
]
IGNORE_LABEL = 255

def colorize(mask):
    """mask uint8 (H,W) -> BGR (H,W,3)"""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, rgb in enumerate(RAILSEM19_PALETTE_RGB):
        bgr = (rgb[2], rgb[1], rgb[0])
        out[mask == i] = bgr
    out[mask == IGNORE_LABEL] = (0, 0, 0)
    return out

def read_mask_png(path):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Fixed list requested by Omar
    names = ["rs06139_bri0p7", "rs00326", "rs08008"]

    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    model.eval()

    for n in names:
        img_path = os.path.join(args.img_dir, n + ".jpg")
        gt_path  = os.path.join(args.gt_dir,  n + ".png")

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Missing GT: {gt_path}")

        # Original (BGR) -> save as PNG lossless
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Cannot read: {img_path}")

        # Predict
        res = inference_segmentor(model, img_path)
        pred = res[0] if isinstance(res, (list, tuple)) else res
        pred = pred.astype(np.uint8)

        # GT
        gt = read_mask_png(gt_path)

        # Ensure same size
        h, w = img_bgr.shape[:2]
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        if gt.shape != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

        # Colorize (BGR)
        pred_col = colorize(pred)
        gt_col = colorize(gt)

        # Save
        cv2.imwrite(os.path.join(args.out_dir, f"{n}_ORIG.png"), img_bgr)
        cv2.imwrite(os.path.join(args.out_dir, f"{n}_GT_RGB.png"), gt_col)
        cv2.imwrite(os.path.join(args.out_dir, f"{n}_PRED_RGB.png"), pred_col)

        print(f"[OK] saved {n}: ORIG / GT_RGB / PRED_RGB")

if __name__ == "__main__":
    main()

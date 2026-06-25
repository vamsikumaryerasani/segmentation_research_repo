#!/usr/bin/env python3
import os
import shutil
import argparse
import cv2
import numpy as np

from mmseg.apis import init_segmentor, inference_segmentor

# RailSem19 palette in RGB
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


def mask_to_rgb(mask):
    """Convert uint8 class-index mask to RGB color mask."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for lab, rgb in enumerate(RAILSEM19_PALETTE_RGB):
        out[mask == lab] = rgb
    out[mask == IGNORE_LABEL] = [0, 0, 0]
    return out


def resize_nn(mask, shape_hw):
    h, w = shape_hw
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def find_gt_path(img_path, gt_root):
    """
    Example:
      rs06139_bri0p7.jpg -> rs06139_bri0p7.png
    """
    stem = os.path.splitext(os.path.basename(img_path))[0]
    gt_path = os.path.join(gt_root, stem + ".png")
    if os.path.isfile(gt_path):
        return gt_path
    return None


def save_rgb_png(rgb_img, out_path):
    # cv2.imwrite expects BGR, so convert RGB -> BGR
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="Path to mmseg config")
    ap.add_argument("checkpoint", help="Path to checkpoint .pth")
    ap.add_argument("--gt-root", required=True, help="Directory containing GT png masks")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    ap.add_argument("--images", nargs="+", required=True, help="List of image paths")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading model...")
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    for img_path in args.images:
        if not os.path.isfile(img_path):
            print(f"[WARN] Image not found: {img_path}")
            continue

        stem = os.path.splitext(os.path.basename(img_path))[0]
        sample_dir = os.path.join(args.out_dir, stem)
        os.makedirs(sample_dir, exist_ok=True)

        gt_path = find_gt_path(img_path, args.gt_root)
        if gt_path is None:
            print(f"[WARN] GT not found for {img_path}")
            continue

        # Read original image
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue
        h, w = img_bgr.shape[:2]

        # Read GT
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt is None:
            print(f"[WARN] Could not read GT: {gt_path}")
            continue
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        gt = gt.astype(np.uint8)

        if gt.shape != (h, w):
            gt = resize_nn(gt, (h, w))

        # Prediction
        result = inference_segmentor(model, img_path)
        pred = result[0] if isinstance(result, (list, tuple)) else result
        pred = pred.astype(np.uint8)

        if pred.shape != (h, w):
            pred = resize_nn(pred, (h, w))

        # Convert masks to RGB
        pred_rgb = mask_to_rgb(pred)
        gt_rgb = mask_to_rgb(gt)

        # Save outputs
        # 1) original raw image (copied as-is)
        orig_out = os.path.join(sample_dir, f"{stem}_original{os.path.splitext(img_path)[1]}")
        shutil.copy2(img_path, orig_out)

        # 2) predicted RGB mask
        pred_out = os.path.join(sample_dir, f"{stem}_pred_rgb.png")
        save_rgb_png(pred_rgb, pred_out)

        # 3) GT RGB mask
        gt_out = os.path.join(sample_dir, f"{stem}_gt_rgb.png")
        save_rgb_png(gt_rgb, gt_out)

        print(f"[OK] Saved sample: {stem}")
        print(f"     original : {orig_out}")
        print(f"     pred_rgb : {pred_out}")
        print(f"     gt_rgb   : {gt_out}")

    print("\n[DONE] All requested samples processed.")


if __name__ == "__main__":
    main()

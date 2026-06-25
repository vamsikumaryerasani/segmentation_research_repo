import os, glob, argparse
import numpy as np
import cv2
import mmcv
from mmseg.apis import init_segmentor, inference_segmentor

# RailSem19 19-class color palette (RGB)
RAILSEM19_PALETTE = [
    [128, 64,128], [244, 35,232], [70, 70, 70], [102,102,156], [190,153,153],
    [153,153,153], [250,170, 30], [220,220,  0], [107,142, 35], [152,251,152],
    [70,130,180], [220, 20, 60], [255,  0,  0], [0,  0,142], [0,  0, 70],
    [0, 60,100], [0, 80,100], [0,  0,230], [119, 11, 32],
]

def colorize_mask(mask):
    """mask (H,W) -> color mask (BGR) using RailSem19 palette"""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for label, rgb in enumerate(RAILSEM19_PALETTE):
        bgr = (rgb[2], rgb[1], rgb[0])
        out[mask == label] = bgr
    return out

def save_gray_masks(mask, out_raw_path, out_vis_path):
    """
    Option 1:
      - raw grayscale: class IDs (0..18) saved as 8-bit single channel
      - visible grayscale: normalized to 0..255 so it looks correct when opened
    """
    gray_raw = mask.astype(np.uint8)              # values 0..18
    cv2.imwrite(out_raw_path, gray_raw)

    # for visualization (so it doesn't look black)
    gray_vis = (mask.astype(np.float32) * (255.0 / 18.0)).clip(0,255).astype(np.uint8)
    cv2.imwrite(out_vis_path, gray_vis)

def read_gt_mask(gt_path):
    gt = mmcv.imread(gt_path, flag="unchanged")
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    return gt.astype(np.int32)

def find_gt(ann_dir, img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    cand = os.path.join(ann_dir, base + ".png")
    if os.path.isfile(cand):
        return cand
    hits = glob.glob(os.path.join(ann_dir, base + ".*"))
    return hits[0] if hits else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--ann-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--glob", default="*")
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--opacity", type=float, default=0.6)  # for color overlay only
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    # collect images
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.webp")
    imgs = []
    if args.glob != "*" and args.glob:
        imgs = sorted(glob.glob(os.path.join(args.in_dir, "**", args.glob), recursive=True))
    else:
        for e in exts:
            imgs += glob.glob(os.path.join(args.in_dir, "**", e), recursive=True)
        imgs = sorted(imgs)

    if not imgs:
        raise RuntimeError(f"No images found in {args.in_dir}")

    imgs = imgs[:args.max_images]

    for i, img_path in enumerate(imgs):
        img = mmcv.imread(img_path)  # BGR

        # inference
        pred = inference_segmentor(model, img_path)[0].astype(np.int32)

        # GT
        gt_path = find_gt(args.ann_dir, img_path)
        if gt_path is None:
            print(f"[WARN] GT not found for {img_path}, skipping.")
            continue
        gt = read_gt_mask(gt_path)

        base = os.path.splitext(os.path.basename(img_path))[0]

        # ---------- GRAYSCALE (Option 1) ----------
        # raw IDs (0..18) + visible normalized
        pred_gray_raw = os.path.join(args.out_dir, f"{i:02d}_{base}_pred_gray_RAW_ids.png")
        pred_gray_vis = os.path.join(args.out_dir, f"{i:02d}_{base}_pred_gray_VIS_0_255.png")
        gt_gray_raw   = os.path.join(args.out_dir, f"{i:02d}_{base}_gt_gray_RAW_ids.png")
        gt_gray_vis   = os.path.join(args.out_dir, f"{i:02d}_{base}_gt_gray_VIS_0_255.png")

        save_gray_masks(pred, pred_gray_raw, pred_gray_vis)
        save_gray_masks(gt,   gt_gray_raw,   gt_gray_vis)

        # panel like you want: original | pred_gray | gt_gray (use VIS so it opens correctly)
        pred_vis = cv2.imread(pred_gray_vis, cv2.IMREAD_GRAYSCALE)
        gt_vis   = cv2.imread(gt_gray_vis, cv2.IMREAD_GRAYSCALE)
        pred_vis3 = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)
        gt_vis3   = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)
        panel_gray = np.concatenate([img, pred_vis3, gt_vis3], axis=1)
        mmcv.imwrite(panel_gray, os.path.join(args.out_dir, f"{i:02d}_{base}_PANEL_gray.png"))

        # ---------- COLOR MASK + OVERLAY ----------
        pred_color = colorize_mask(pred)
        gt_color   = colorize_mask(gt)

        mmcv.imwrite(pred_color, os.path.join(args.out_dir, f"{i:02d}_{base}_pred_COLOR.png"))
        mmcv.imwrite(gt_color,   os.path.join(args.out_dir, f"{i:02d}_{base}_gt_COLOR.png"))

        overlay_pred = (args.opacity * img + (1.0 - args.opacity) * pred_color).clip(0,255).astype(np.uint8)
        overlay_gt   = (args.opacity * img + (1.0 - args.opacity) * gt_color).clip(0,255).astype(np.uint8)

        mmcv.imwrite(overlay_pred, os.path.join(args.out_dir, f"{i:02d}_{base}_overlay_pred.png"))
        mmcv.imwrite(overlay_gt,   os.path.join(args.out_dir, f"{i:02d}_{base}_overlay_gt.png"))

        # panel: original | pred_color | gt_color
        panel_color = np.concatenate([img, pred_color, gt_color], axis=1)
        mmcv.imwrite(panel_color, os.path.join(args.out_dir, f"{i:02d}_{base}_PANEL_color.png"))

        print(f"done: {i:02d} {base}")

if __name__ == "__main__":
    main()

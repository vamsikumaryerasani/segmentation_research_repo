#!/usr/bin/env python3
import os, glob, argparse
import cv2
import numpy as np
from mmcv import Config
from mmseg.apis import init_segmentor, inference_segmentor

# RailSem19 palette (RGB) -> convert to BGR for OpenCV
RAILSEM19_PALETTE_RGB = [
    [128, 64,128],[244, 35,232],[70, 70, 70],[102,102,156],[190,153,153],
    [153,153,153],[250,170, 30],[220,220,  0],[107,142, 35],[152,251,152],
    [70,130,180],[220, 20, 60],[255,  0,  0],[  0,  0,142],[  0,  0, 70],
    [  0, 60,100],[  0, 80,100],[  0,  0,230],[119, 11, 32],
]
PALETTE_BGR = np.array(RAILSEM19_PALETTE_RGB, dtype=np.uint8)[:, ::-1]

IMG_EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp")

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def imread_bgr(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Could not read image: {p}")
    return im

def read_gt_mask(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"Could not read GT mask: {mask_path}")
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)

def colorize_ids(ids_u8):
    h, w = ids_u8.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k in range(len(PALETTE_BGR)):
        out[ids_u8 == k] = PALETTE_BGR[k]
    return out

def ids_to_gray_vis(ids_u8, max_id=18):
    ids = ids_u8.astype(np.float32)
    vis = np.round(ids * (255.0 / float(max_id))).clip(0, 255).astype(np.uint8)
    return vis

def overlay(img_bgr, color_bgr, alpha_img=0.6):
    # alpha_img = weight for original image
    return cv2.addWeighted(img_bgr, alpha_img, color_bgr, 1.0 - alpha_img, 0)

def to_bgr(gray_u8):
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

def add_header(tri_bgr):
    titles=("Original images","Predicted outputs","ground-truth masks")
    h, w = tri_bgr.shape[:2]
    header_h = max(70, int(0.10 * h))
    canvas = np.full((h + header_h, w, 3), 255, dtype=np.uint8)
    canvas[header_h:] = tri_bgr

    col_w = w // 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.1
    thick = 2
    for i, t in enumerate(titles):
        x0 = i * col_w
        x1 = (i + 1) * col_w if i < 2 else w
        (tw, th), _ = cv2.getTextSize(t, font, scale, thick)
        cx = (x0 + x1)//2 - tw//2
        cy = header_h//2 + th//2
        cv2.putText(canvas, t, (cx, cy), font, scale, (0,0,0), thick, cv2.LINE_AA)
    return canvas

def find_gt_for_image(gt_root, img_path):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    direct = os.path.join(gt_root, stem + ".png")
    if os.path.exists(direct):
        return direct
    hits = glob.glob(os.path.join(gt_root, "**", stem + ".png"), recursive=True)
    if hits:
        return hits[0]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--opacity", type=float, default=0.6, help="weight for original image in overlay (0..1)")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    val = cfg.data.val
    if isinstance(val, (list, tuple)):
        val = val[0]

    # build GT root from config (relative to repo root)
    repo_root = os.getcwd()
    data_root = val.get("data_root", cfg.data.get("data_root", "")) or ""
    ann_dir = val.get("ann_dir", "")
    gt_root = os.path.normpath(os.path.join(repo_root, data_root, ann_dir))

    if not os.path.isdir(gt_root):
        raise RuntimeError(f"GT mask directory not found: {gt_root}")

    # gather input images
    imgs = []
    for ext in IMG_EXTS:
        imgs += sorted(glob.glob(os.path.join(args.in_dir, f"*{ext}")))
    imgs = imgs[:args.max_images]
    if not imgs:
        raise RuntimeError(f"No input images found in: {args.in_dir}")

    # output dirs
    OUT = args.out_dir
    d_orig = os.path.join(OUT, "orig")
    d_pred_gray = os.path.join(OUT, "pred_gray")
    d_gt_gray = os.path.join(OUT, "gt_gray")
    d_overlay = os.path.join(OUT, "overlay_pred")
    d_gt_color = os.path.join(OUT, "gt_color")
    d_tri_black = os.path.join(OUT, "triptych_black")
    d_tri_color = os.path.join(OUT, "triptych_color")
    for d in [d_orig, d_pred_gray, d_gt_gray, d_overlay, d_gt_color, d_tri_black, d_tri_color]:
        ensure_dir(d)

    # init model
    model = init_segmentor(args.config, args.checkpoint, device="cuda:0")

    made = 0
    for i, img_path in enumerate(imgs):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        tag = f"{i:02d}_{stem}"

        img = imread_bgr(img_path)

        # inference
        result = inference_segmentor(model, img_path)
        pred = result[0].astype(np.uint8)

        # gt
        gt_path = find_gt_for_image(gt_root, img_path)
        if gt_path is None:
            print(f"[WARN] GT not found for {img_path} (looked in {gt_root}) -> skipping")
            continue
        gt = read_gt_mask(gt_path)

        # resize masks to image size (just in case)
        H, W = img.shape[:2]
        if pred.shape[:2] != (H, W):
            pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
        if gt.shape[:2] != (H, W):
            gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)

        # black/grayscale visuals
        pred_gray_vis = ids_to_gray_vis(pred, max_id=18)
        gt_gray_vis = ids_to_gray_vis(gt, max_id=18)

        # color visuals
        pred_color = colorize_ids(pred)
        gt_color = colorize_ids(gt)
        pred_overlay = overlay(img, pred_color, alpha_img=args.opacity)

        # save single images
        cv2.imwrite(os.path.join(d_orig, f"{tag}_orig.png"), img)
        cv2.imwrite(os.path.join(d_pred_gray, f"{tag}_pred_gray_VIS_0_255.png"), pred_gray_vis)
        cv2.imwrite(os.path.join(d_gt_gray, f"{tag}_gt_gray_VIS_0_255.png"), gt_gray_vis)
        cv2.imwrite(os.path.join(d_overlay, f"{tag}_overlay_pred.png"), pred_overlay)
        cv2.imwrite(os.path.join(d_gt_color, f"{tag}_gt_color.png"), gt_color)

        # triptych_black: Original | Pred(gray) | GT(gray)
        tri_b = np.concatenate([img, to_bgr(pred_gray_vis), to_bgr(gt_gray_vis)], axis=1)
        tri_b = add_header(tri_b)
        cv2.imwrite(os.path.join(d_tri_black, f"{tag}_triptych_black.png"), tri_b)

        # triptych_color: Original | Pred(overlay) | GT(color mask)
        tri_c = np.concatenate([img, pred_overlay, gt_color], axis=1)
        tri_c = add_header(tri_c)
        cv2.imwrite(os.path.join(d_tri_color, f"{tag}_triptych_color.png"), tri_c)

        made += 1

    print(f"[OK] made {made} samples in: {OUT}")
    print("[OK] triptychs:")
    print(" -", d_tri_black)
    print(" -", d_tri_color)

if __name__ == "__main__":
    main()

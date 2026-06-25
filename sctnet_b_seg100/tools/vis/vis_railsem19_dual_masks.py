import os, glob, argparse
import numpy as np
import mmcv
from mmseg.apis import init_segmentor, inference_segmentor

# RailSem19 / Cityscapes-style palette (19 classes)
RAILSEM19_PALETTE = [
    [128,  64,128],  # 0 Road
    [244,  35,232],  # 1 Sidewalk
    [ 70,  70, 70],  # 2 Building
    [102,102,156],   # 3 Wall
    [190,153,153],   # 4 Fence
    [153,153,153],   # 5 Pole
    [250,170, 30],   # 6 Traffic Light
    [220,220,  0],   # 7 Traffic Sign
    [107,142, 35],   # 8 Vegetation
    [152,251,152],   # 9 Terrain
    [ 70,130,180],   # 10 Sky
    [220, 20, 60],   # 11 Person
    [255,  0,  0],   # 12 Rider
    [  0,  0,142],   # 13 Car
    [  0,  0, 70],   # 14 Truck
    [  0, 60,100],   # 15 Bus
    [  0, 80,100],   # 16 Train
    [  0,  0,230],   # 17 Motorcycle
    [119, 11, 32],   # 18 Bicycle
]

def colorize_mask(mask: np.ndarray, palette):
    """mask: (H,W) int -> BGR uint8 image"""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for label, rgb in enumerate(palette):
        # palette is RGB; mmcv.imwrite expects BGR, so flip
        bgr = (rgb[2], rgb[1], rgb[0])
        out[mask == label] = bgr
    return out

def grayize_mask(mask: np.ndarray, num_classes: int):
    """map class ids -> 0..255 grayscale (then 3ch BGR)"""
    if num_classes <= 1:
        g = np.zeros_like(mask, dtype=np.uint8)
    else:
        g = (mask.astype(np.float32) * (255.0 / (num_classes - 1))).clip(0, 255).astype(np.uint8)
    g3 = np.stack([g, g, g], axis=-1)  # BGR
    return g3

def find_images(in_dir, pattern):
    exts = ["*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.webp"]
    if pattern and pattern != "*":
        return sorted(glob.glob(os.path.join(in_dir, "**", pattern), recursive=True))
    imgs = []
    for e in exts:
        imgs += glob.glob(os.path.join(in_dir, "**", e), recursive=True)
    return sorted(imgs)

def find_gt(ann_dir, img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    # RailSem19 masks are usually .png
    cand = os.path.join(ann_dir, base + ".png")
    if os.path.isfile(cand):
        return cand
    # fallback: any extension
    hits = glob.glob(os.path.join(ann_dir, base + ".*"))
    return hits[0] if hits else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--ann-dir", default=None, help="GT masks dir. If omitted, tries to infer from config.val.ann_dir + data_root")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--glob", default="*", help="image glob (e.g. '*.png'). Default: auto-detect common image exts")
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # init model
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    # infer ann_dir from config if not provided
    ann_dir = args.ann_dir
    if ann_dir is None:
        from mmcv import Config
        cfg = Config.fromfile(args.config)
        val = cfg.data.val[0] if isinstance(cfg.data.val, (list, tuple)) else cfg.data.val
        data_root = val.get("data_root", cfg.data.get("data_root", "")) or ""
        ann_dir_cfg = val.get("ann_dir", None)
        if ann_dir_cfg is None:
            raise RuntimeError("Couldn't infer ann_dir from config. Pass --ann-dir explicitly.")
        ann_dir = os.path.normpath(os.path.join(data_root, ann_dir_cfg))

    imgs = find_images(args.in_dir, args.glob)
    if not imgs:
        raise RuntimeError(f"No images found in {args.in_dir} with pattern {args.glob}")

    imgs = imgs[:args.max_images]
    num_classes = 19

    for idx, img_path in enumerate(imgs):
        img_bgr = mmcv.imread(img_path)  # BGR
        pred = inference_segmentor(model, img_path)
        pred_mask = pred[0].astype(np.int32)

        gt_path = find_gt(ann_dir, img_path)
        if gt_path is None:
            print(f"[WARN] GT not found for {img_path} (searched in {ann_dir}). Skipping GT panels.")
            continue
        gt_mask = mmcv.imread(gt_path, flag="unchanged")
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[:, :, 0]
        gt_mask = gt_mask.astype(np.int32)

        # build visuals
        pred_color = colorize_mask(pred_mask, RAILSEM19_PALETTE)
        gt_color   = colorize_mask(gt_mask,   RAILSEM19_PALETTE)

        pred_gray = grayize_mask(pred_mask, num_classes)
        gt_gray   = grayize_mask(gt_mask,   num_classes)

        # 2 rows x 3 cols (matches your screenshot)
        row1 = np.concatenate([img_bgr, pred_gray, gt_gray], axis=1)
        row2 = np.concatenate([img_bgr, pred_color, gt_color], axis=1)
        panel = np.concatenate([row1, row2], axis=0)

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_panel = os.path.join(args.out_dir, f"{idx:02d}_{base}_panel.png")
        mmcv.imwrite(panel, out_panel)

        # also save individual outputs (optional, handy)
        mmcv.imwrite(pred_color, os.path.join(args.out_dir, f"{idx:02d}_{base}_pred_color.png"))
        mmcv.imwrite(gt_color,   os.path.join(args.out_dir, f"{idx:02d}_{base}_gt_color.png"))
        mmcv.imwrite(pred_gray,  os.path.join(args.out_dir, f"{idx:02d}_{base}_pred_gray.png"))
        mmcv.imwrite(gt_gray,    os.path.join(args.out_dir, f"{idx:02d}_{base}_gt_gray.png"))

        print("saved:", out_panel)

if __name__ == "__main__":
    main()

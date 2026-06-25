import os, glob, random, argparse
import numpy as np
import mmcv
from mmseg.apis import init_segmentor, inference_segmentor

# RailSem19 19-class color palette (RGB)
RAILSEM19_PALETTE = [
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
NUM_CLASSES = 19
IGNORE_LABEL = 255


def list_images(root, exts=("png","jpg","jpeg","bmp","tif","tiff","webp")):
    imgs = []
    for e in exts:
        imgs += glob.glob(os.path.join(root, "**", f"*.{e}"), recursive=True)
    return sorted(imgs)


def colorize_mask(mask, palette):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(palette):
        out[mask == i] = c
    return out


def to_grayscale(mask, num_classes=NUM_CLASSES):
    """Map class ids [0..C-1] to grayscale [0..255]. Keep ignore=255 as 255."""
    mask = mask.astype(np.int32)
    gray = np.zeros_like(mask, dtype=np.uint8)
    scale = 255.0 / float(num_classes - 1)
    valid = (mask >= 0) & (mask < num_classes)
    gray[valid] = np.clip(np.round(mask[valid] * scale), 0, 255).astype(np.uint8)
    gray[mask == IGNORE_LABEL] = 255
    return gray


def find_gt_path(img_path, gt_dir):
    """
    Best-effort mapping:
    - Uses same basename
    - tries .png
    """
    base = os.path.splitext(os.path.basename(img_path))[0]
    cand = os.path.join(gt_dir, base + ".png")
    return cand if os.path.isfile(cand) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="mmseg config .py")
    ap.add_argument("checkpoint", help="checkpoint .pth")
    ap.add_argument("--in-dir", required=True, help="folder with input images")
    ap.add_argument("--gt-dir", default=None, help="folder with GT masks (.png), optional")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-weight", type=float, default=0.6, help="overlay weight for original image")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    imgs = list_images(args.in_dir)
    if not imgs:
        raise RuntimeError(f"No images found in: {args.in_dir}")

    random.seed(args.seed)
    random.shuffle(imgs)
    imgs = imgs[:args.max_images]

    model = init_segmentor(args.config, args.checkpoint, device="cuda:0")
    # mmseg uses PALETTE for some internal visualization; harmless to set
    model.PALETTE = RAILSEM19_PALETTE

    iw = float(args.img_weight)
    mw = 1.0 - iw

    print(f"[OK] Using {len(imgs)} images")
    print(f"[OK] Saving to: {args.out_dir}")
    print(f"[OK] Overlay = {iw:.2f}*image + {mw:.2f}*mask")

    for idx, img_path in enumerate(imgs, 1):
        img_bgr = mmcv.imread(img_path)  # BGR uint8

        result = inference_segmentor(model, img_path)
        pred = result[0] if isinstance(result, (list, tuple)) else result
        pred = pred.astype(np.uint8)

        # pred color + gray
        pred_rgb = colorize_mask(pred, RAILSEM19_PALETTE)
        pred_bgr = pred_rgb[:, :, ::-1]
        pred_gray = to_grayscale(pred)

        overlay = (iw * img_bgr.astype(np.float32) + mw * pred_bgr.astype(np.float32)).astype(np.uint8)

        # optional GT
        gt_bgr = None
        gt_gray = None
        if args.gt_dir:
            gt_path = find_gt_path(img_path, args.gt_dir)
            if gt_path:
                gt = mmcv.imread(gt_path, flag="unchanged")
                if gt.ndim == 3:
                    gt = gt[:, :, 0]
                gt = gt.astype(np.uint8)

                gt_rgb = colorize_mask(gt, RAILSEM19_PALETTE)
                gt_bgr = gt_rgb[:, :, ::-1]
                gt_gray = to_grayscale(gt)

        base = os.path.splitext(os.path.basename(img_path))[0]

        # save individual
        mmcv.imwrite(pred_bgr,  os.path.join(args.out_dir, f"{idx:02d}_{base}_pred_color.png"))
        mmcv.imwrite(pred_gray, os.path.join(args.out_dir, f"{idx:02d}_{base}_pred_gray.png"))
        mmcv.imwrite(overlay,   os.path.join(args.out_dir, f"{idx:02d}_{base}_overlay.png"))

        # build strip
        pred_gray_bgr = np.stack([pred_gray]*3, axis=2)

        if gt_bgr is not None:
            gt_gray_bgr = np.stack([gt_gray]*3, axis=2)

            mmcv.imwrite(gt_bgr,  os.path.join(args.out_dir, f"{idx:02d}_{base}_gt_color.png"))
            mmcv.imwrite(gt_gray, os.path.join(args.out_dir, f"{idx:02d}_{base}_gt_gray.png"))

            strip = np.concatenate([img_bgr, pred_gray_bgr, pred_bgr, gt_gray_bgr, gt_bgr], axis=1)
            mmcv.imwrite(strip, os.path.join(args.out_dir, f"{idx:02d}_{base}_strip.png"))
            print(f"[{idx}/{len(imgs)}] {base} (gt: yes)")
        else:
            strip = np.concatenate([img_bgr, pred_gray_bgr, pred_bgr], axis=1)
            mmcv.imwrite(strip, os.path.join(args.out_dir, f"{idx:02d}_{base}_strip.png"))
            print(f"[{idx}/{len(imgs)}] {base} (gt: no)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from models.ddrnet23 import get_ddrnet23


RAILSEM19_PALETTE_RGB = [
    [128, 64, 128],   # 0
    [244, 35, 232],   # 1
    [70, 70, 70],     # 2
    [102, 102, 156],  # 3
    [190, 153, 153],  # 4
    [153, 153, 153],  # 5
    [250, 170, 30],   # 6
    [220, 220, 0],    # 7
    [107, 142, 35],   # 8
    [152, 251, 152],  # 9
    [70, 130, 180],   # 10
    [220, 20, 60],    # 11
    [255, 0, 0],      # 12
    [0, 0, 142],      # 13
    [0, 0, 70],       # 14
    [0, 60, 100],     # 15
    [0, 80, 100],     # 16
    [0, 0, 230],      # 17
    [119, 11, 32],    # 18
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


def find_raw_image_by_basename(img_path, raw_dir):
    base = os.path.splitext(os.path.basename(img_path))[0]

    root = base
    for tag in ["_bri", "_con"]:
        if tag in root:
            root = root.split(tag, 1)[0]
            break

    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
        cand = os.path.join(raw_dir, root + ext)
        if os.path.isfile(cand):
            return cand
    return None


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


def make_triptych(left, mid, right,
                  labels=("Original image", "Predicted mask", "Ground truth mask")):
    h, w = left.shape[:2]

    if mid.shape[:2] != (h, w):
        mid = resize_nn(mid, (h, w))
    if right.shape[:2] != (h, w):
        right = resize_nn(right, (h, w))

    header_h = 36
    canvas = np.full((header_h + h, 3 * w, 3), 255, dtype=np.uint8)

    canvas[header_h:header_h + h, 0:w] = left
    canvas[header_h:header_h + h, w:2 * w] = mid
    canvas[header_h:header_h + h, 2 * w:3 * w] = right

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.5
    th = 1
    color = (0, 0, 0)
    xs = [w // 2, w + w // 2, 2 * w + w // 2]

    for x, txt in zip(xs, labels):
        (tw, _), _ = cv2.getTextSize(txt, font, fs, th)
        cv2.putText(canvas, txt, (x - tw // 2, 22), font, fs, color, th, cv2.LINE_AA)

    return canvas


def preprocess_bgr_image(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0)
    return img


def load_checkpoint_model(checkpoint_path, device, num_classes=19):
    model = get_ddrnet23(num_classes=num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer_mask(model, img_bgr, device):
    h, w = img_bgr.shape[:2]
    x = preprocess_bgr_image(img_bgr).to(device)

    y = model(x)
    if isinstance(y, (list, tuple)):
        y = y[0]

    if y.shape[-2:] != (h, w):
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)

    pred = torch.argmax(y, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to DDRNet checkpoint .pth")
    ap.add_argument("--in-dir", required=True, help="Input images directory")
    ap.add_argument("--raw-dir", default="", help="Original raw RGB images directory")
    ap.add_argument("--gt-dir", default="", help="Ground-truth masks directory")
    ap.add_argument("--split-file", default="", help="Optional txt file with stems to process")
    ap.add_argument("--out-dir", required=True, help="Output base directory")
    ap.add_argument("--start-index", type=int, default=0, help="Start index in sorted image list")
    ap.add_argument("--max-images", type=int, default=10, help="Number of images to process")
    ap.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0")
    ap.add_argument("--save-triptych", action="store_true", help="Also save triptych outputs")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_uint8 = os.path.join(args.out_dir, "pred_uint8")
    out_color = os.path.join(args.out_dir, "pred_color")
    out_black = os.path.join(args.out_dir, "pred_black")
    out_trip_black = os.path.join(args.out_dir, "triptych_black")
    out_trip_color = os.path.join(args.out_dir, "triptych_color")

    os.makedirs(out_uint8, exist_ok=True)
    os.makedirs(out_color, exist_ok=True)
    os.makedirs(out_black, exist_ok=True)
    if args.save_triptych:
        os.makedirs(out_trip_black, exist_ok=True)
        os.makedirs(out_trip_color, exist_ok=True)

    if not os.path.isdir(args.in_dir):
        raise SystemExit(f"[ERROR] Input directory not found: {args.in_dir}")

    if args.save_triptych:
        if not args.raw_dir or not os.path.isdir(args.raw_dir):
            raise SystemExit("[ERROR] --raw-dir is required and must exist when using --save-triptych")
        if not args.gt_dir or not os.path.isdir(args.gt_dir):
            raise SystemExit("[ERROR] --gt-dir is required and must exist when using --save-triptych")

    if args.split_file:
        if not os.path.isfile(args.split_file):
            raise SystemExit(f"[ERROR] Split file not found: {args.split_file}")

        with open(args.split_file, "r") as f:
            stems = [x.strip() for x in f if x.strip()]

        selected = stems[args.start_index:args.start_index + args.max_images]
        imgs = []
        for stem in selected:
            found = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
                cand = os.path.join(args.in_dir, stem + ext)
                if os.path.isfile(cand):
                    found = cand
                    break
            if found is not None:
                imgs.append(found)
            else:
                print("[WARN] Missing image for stem:", stem)
    else:
        imgs = list_images(args.in_dir)
        imgs = imgs[args.start_index:args.start_index + args.max_images]

    if not imgs:
        raise SystemExit("[ERROR] No images found to process.")

    print("[INFO] Loading model...")
    model = load_checkpoint_model(args.checkpoint, device=device, num_classes=19)

    n_done = 0
    for img_path in imgs:
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print("[WARN] Cannot read image:", img_path)
            continue

        pred = infer_mask(model, img_bgr, device=device)

        pred_color = to_color(pred)
        pred_black = to_black_gray(pred)

        token = os.path.splitext(os.path.basename(img_path))[0]

        cv2.imwrite(os.path.join(out_uint8, f"{token}.png"), pred)
        cv2.imwrite(os.path.join(out_color, f"{token}_color.png"), pred_color)
        cv2.imwrite(os.path.join(out_black, f"{token}_black.png"), pred_black)

        if args.save_triptych:
            raw_path = find_raw_image_by_basename(img_path, args.raw_dir)
            if raw_path is None:
                print("[WARN] Raw image not found for:", os.path.basename(img_path))
                continue

            raw_img = cv2.imread(raw_path, cv2.IMREAD_COLOR)
            if raw_img is None:
                print("[WARN] Cannot read raw image:", raw_path)
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

            h, w = raw_img.shape[:2]
            if pred.shape != (h, w):
                pred = resize_nn(pred, (h, w))
                pred_color = to_color(pred)
                pred_black = to_black_gray(pred)
            if gt.shape != (h, w):
                gt = resize_nn(gt, (h, w))

            gt_color = to_color(gt)
            gt_black = to_black_gray(gt)

            tri_color = make_triptych(raw_img, pred_color, gt_color)
            tri_black = make_triptych(raw_img, pred_black, gt_black)

            cv2.imwrite(os.path.join(out_trip_color, f"{token}_triptych_color.png"), tri_color)
            cv2.imwrite(os.path.join(out_trip_black, f"{token}_triptych_black.png"), tri_black)

        n_done += 1
        print(f"[{n_done}/{len(imgs)}] Wrote: {token}")

    print(f"[OK] Wrote {n_done} uint8 masks -> {out_uint8}")
    print(f"[OK] Wrote {n_done} color masks -> {out_color}")
    print(f"[OK] Wrote {n_done} black masks -> {out_black}")
    if args.save_triptych:
        print(f"[OK] Wrote triptychs -> {out_trip_color}")
        print(f"[OK] Wrote triptychs -> {out_trip_black}")


if __name__ == "__main__":
    main()

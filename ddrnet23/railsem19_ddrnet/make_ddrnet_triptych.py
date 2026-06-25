#!/usr/bin/env python3
import os
import glob
import random
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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for lab, rgb in enumerate(RAILSEM19_PALETTE_RGB):
        bgr = (rgb[2], rgb[1], rgb[0])
        out[mask == lab] = bgr
    out[mask == IGNORE_LABEL] = (0, 0, 0)
    return out


def to_black_gray(mask):
    g = mask.astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def resize_nn(img, shape_hw):
    h, w = shape_hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)


def make_triptych(left, mid, right, labels):
    h, w = left.shape[:2]

    if mid.shape[:2] != (h, w):
        mid = resize_nn(mid, (h, w))
    if right.shape[:2] != (h, w):
        right = resize_nn(right, (h, w))

    header_h = 44
    canvas = np.full((header_h + h, 3 * w, 3), 255, dtype=np.uint8)

    canvas[header_h:header_h + h, 0:w] = left
    canvas[header_h:header_h + h, w:2 * w] = mid
    canvas[header_h:header_h + h, 2 * w:3 * w] = right

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.7
    th = 2
    color = (0, 0, 0)
    xs = [w // 2, w + w // 2, 2 * w + w // 2]

    for x, txt in zip(xs, labels):
        (tw, _), _ = cv2.getTextSize(txt, font, fs, th)
        cv2.putText(canvas, txt, (x - tw // 2, 30), font, fs, color, th, cv2.LINE_AA)

    return canvas


def preprocess_image(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0)
    return img


def load_model(checkpoint_path, device, num_classes=19):
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
def infer_mask(model, img_bgr, device, infer_height=512, infer_width=1024):
    orig_h, orig_w = img_bgr.shape[:2]

    img_resized = cv2.resize(
        img_bgr,
        (infer_width, infer_height),
        interpolation=cv2.INTER_LINEAR,
    )

    x = preprocess_image(img_resized).to(device)

    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]

    if out.shape[-2:] != (orig_h, orig_w):
        out = F.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

    pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--random-sample", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--infer-width", type=int, default=1024)
    ap.add_argument("--infer-height", type=int, default=512)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    ensure_dir(args.out_dir)
    pred_uint8_dir = os.path.join(args.out_dir, "pred_uint8")
    pred_color_dir = os.path.join(args.out_dir, "pred_color")
    pred_black_dir = os.path.join(args.out_dir, "pred_black")
    triptychs_color_dir = os.path.join(args.out_dir, "triptychs_color")
    triptychs_black_dir = os.path.join(args.out_dir, "triptychs_black")

    ensure_dir(pred_uint8_dir)
    ensure_dir(pred_color_dir)
    ensure_dir(pred_black_dir)
    ensure_dir(triptychs_color_dir)
    ensure_dir(triptychs_black_dir)

    images = list_images(args.in_dir)
    if not images:
        raise RuntimeError(f"No images found in: {args.in_dir}")

    if args.random_sample:
        random.seed(args.seed)
        if args.max_images < len(images):
            images = random.sample(images, args.max_images)
    else:
        images = images[:args.max_images]

    print(f"Loading model from: {args.checkpoint}")
    print(f"Using device: {args.device}")
    print(f"Inference resize: {args.infer_width}x{args.infer_height}")

    model = load_model(args.checkpoint, args.device, num_classes=19)

    processed = 0
    skipped = 0

    for img_path in images:
        gt_path = find_gt_mask_by_basename(img_path, args.gt_dir)
        if gt_path is None:
            print(f"[SKIP] GT not found for {os.path.basename(img_path)}")
            skipped += 1
            continue

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

        if img_bgr is None:
            print(f"[SKIP] Could not read image: {img_path}")
            skipped += 1
            continue

        if gt_mask is None:
            print(f"[SKIP] Could not read GT: {gt_path}")
            skipped += 1
            continue

        if gt_mask.ndim == 3:
            gt_mask = gt_mask[:, :, 0]

        pred_mask = infer_mask(
            model,
            img_bgr,
            args.device,
            infer_height=args.infer_height,
            infer_width=args.infer_width,
        )

        pred_color = to_color(pred_mask)
        pred_black = to_black_gray(pred_mask)
        gt_color = to_color(gt_mask)
        gt_black = to_black_gray(gt_mask)

        color_triptych = make_triptych(
            img_bgr,
            pred_color,
            gt_color,
            labels=("Original image", "Predicted mask", "Ground truth mask"),
        )

        black_triptych = make_triptych(
            img_bgr,
            pred_black,
            gt_black,
            labels=("Original image", "Predicted mask", "Ground truth mask"),
        )

        base = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(pred_uint8_dir, base + ".png"), pred_mask)
        cv2.imwrite(os.path.join(pred_color_dir, base + ".png"), pred_color)
        cv2.imwrite(os.path.join(pred_black_dir, base + ".png"), pred_black)
        cv2.imwrite(os.path.join(triptychs_color_dir, base + ".png"), color_triptych)
        cv2.imwrite(os.path.join(triptychs_black_dir, base + ".png"), black_triptych)

        print(f"[OK] {base}")
        processed += 1

    print(f"Done. Processed={processed}, Skipped={skipped}")
    print(f"Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import argparse
import numpy as np
import cv2
import torch

from models.ddrnet23 import get_ddrnet23

PALETTE = np.array([
    [128,  64, 128],  # 0
    [244,  35, 232],  # 1
    [ 70,  70,  70],  # 2
    [102, 102, 156],  # 3
    [190, 153, 153],  # 4
    [153, 153, 153],  # 5
    [250, 170,  30],  # 6
    [220, 220,   0],  # 7
    [107, 142,  35],  # 8
    [152, 251, 152],  # 9
    [ 70, 130, 180],  # 10
    [220,  20,  60],  # 11
    [255,   0,   0],  # 12
    [  0,   0, 142],  # 13
    [  0,   0,  70],  # 14
    [  0,  60, 100],  # 15
    [  0,  80, 100],  # 16
    [  0,   0, 230],  # 17
    [119,  11,  32],  # 18
], dtype=np.uint8)

IGNORE_INDEX = 255


def colorize_mask(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(PALETTE))
    color[valid] = PALETTE[mask[valid]]
    color[mask == IGNORE_INDEX] = [0, 0, 0]
    return color


def load_checkpoint_model(checkpoint_path, device, num_classes=19):
    model = get_ddrnet23(num_classes=num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        new_state[nk] = v

    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model


def preprocess_image(img_bgr, width=1024, height=512):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    resized = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)
    x = torch.from_numpy(x).unsqueeze(0).float()
    return x, (orig_h, orig_w)


@torch.no_grad()
def predict_mask(model, img_path, device):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    x, (orig_h, orig_w) = preprocess_image(img_bgr)
    x = x.to(device)

    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]

    pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return img_bgr, pred


def load_gt_mask(gt_path, target_hw=None):
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    if gt is None:
        raise RuntimeError(f"Could not read GT mask: {gt_path}")
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    gt = gt.astype(np.uint8)

    if target_hw is not None and gt.shape != target_hw:
        gt = cv2.resize(gt, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)

    return gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--in-dir", required=True, type=str)
    parser.add_argument("--gt-dir", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num-classes", default=19, type=int)
    parser.add_argument("--stems", nargs="+", required=True,
                        help="Example: rs06139_bri0p7 rs00326 rs08008")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading model...")
    model = load_checkpoint_model(
        checkpoint_path=args.checkpoint,
        device=device,
        num_classes=args.num_classes
    )

    for stem in args.stems:
        img_path = os.path.join(args.in_dir, stem + ".jpg")
        gt_path = os.path.join(args.gt_dir, stem + ".png")

        if not os.path.isfile(img_path):
            print(f"[WARN] Missing image: {img_path}")
            continue
        if not os.path.isfile(gt_path):
            print(f"[WARN] Missing GT: {gt_path}")
            continue

        print(f"[INFO] Processing: {stem}")
        orig_bgr, pred_mask = predict_mask(model, img_path, device)
        gt_mask = load_gt_mask(gt_path, target_hw=pred_mask.shape)

        pred_color = colorize_mask(pred_mask)
        gt_color = colorize_mask(gt_mask)

        # save all in SAME folder
        orig_out = os.path.join(args.out_dir, f"{stem}_original.jpg")
        pred_out = os.path.join(args.out_dir, f"{stem}_pred.png")
        gt_out   = os.path.join(args.out_dir, f"{stem}_gt.png")

        cv2.imwrite(orig_out, orig_bgr)
        cv2.imwrite(pred_out, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(gt_out, cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR))

        print(f"  saved: {orig_out}")
        print(f"  saved: {pred_out}")
        print(f"  saved: {gt_out}")

    print(f"[OK] Done. Outputs saved in: {args.out_dir}")


if __name__ == "__main__":
    main()

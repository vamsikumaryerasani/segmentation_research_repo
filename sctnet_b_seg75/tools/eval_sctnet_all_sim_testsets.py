#!/usr/bin/env python3
import os
import csv
import argparse
import tempfile
import numpy as np
from mmcv import Config
from mmseg.apis import init_segmentor, inference_segmentor
from PIL import Image


SIM_DATASETS = [
    ("Different Lighting Conditions", "/data/pool/qmc-41b/dataset_diff_lightning_conditions"),
    ("Fog Mist or Shallow Fog 500m Visibility", "/data/pool/qmc-41b/fog_mist_500m_visibility"),
    ("Moderate Fog 200m visibility", "/data/pool/qmc-41b/moderate_fog_200m_visibility"),
    ("Rain_light 0.4 mm per hour", "/data/pool/qmc-41b/rain_light_0p4mm_per_hour"),
    ("Snow_middle 4 mm per hour", "/data/pool/qmc-41b/snow_middle_4mm_per_hour"),
    ("Snow_Heavy 20 mm per hour", "/data/pool/qmc-41b/snow_heavy_20mm_per_hour"),
    ("Snow_Light Less than 0.5 mm per hour", "/data/pool/qmc-41b/snow_light_less_0p5mm_per_hour"),
]


def read_ids(split_file):
    with open(split_file, "r") as f:
        return [x.strip() for x in f if x.strip()]


def find_file(folder, sample_id):
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
    for ext in exts:
        p = os.path.join(folder, sample_id + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Could not find file for id={sample_id} in {folder}")


def load_mask(mask_path):
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64)


def compute_metrics_from_confmat(confmat):
    eps = 1e-10
    intersection = np.diag(confmat)
    gt_sum = confmat.sum(axis=1)
    pred_sum = confmat.sum(axis=0)
    union = gt_sum + pred_sum - intersection

    iou = intersection / np.maximum(union, eps)
    acc = intersection / np.maximum(gt_sum, eps)
    aacc = intersection.sum() / np.maximum(gt_sum.sum(), eps)
    miou = np.nanmean(iou)
    macc = np.nanmean(acc)

    return {
        "IoU": iou,
        "Acc": acc,
        "aAcc": aacc,
        "mIoU": miou,
        "mAcc": macc,
    }


def evaluate_dataset(model, dataset_root, num_classes=6, ignore_index=255):
    split_file = os.path.join(dataset_root, "splits", "test.txt")
    img_dir = os.path.join(dataset_root, "images")
    mask_dir = os.path.join(dataset_root, "masks")

    ids = read_ids(split_file)
    confmat = np.zeros((num_classes, num_classes), dtype=np.int64)

    total = len(ids)
    for i, sample_id in enumerate(ids, 1):
        img_path = find_file(img_dir, sample_id)
        mask_path = find_file(mask_dir, sample_id)

        gt = load_mask(mask_path)
        result = inference_segmentor(model, img_path)
        pred = result[0] if isinstance(result, list) else result
        pred = np.array(pred, dtype=np.int64)

        if pred.shape != gt.shape:
            from cv2 import resize, INTER_NEAREST
            pred = resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=INTER_NEAREST).astype(np.int64)

        valid = gt != ignore_index
        gt_valid = gt[valid]
        pred_valid = pred[valid]

        inds = gt_valid * num_classes + pred_valid
        bincount = np.bincount(inds, minlength=num_classes * num_classes)
        confmat += bincount.reshape(num_classes, num_classes)

        if i % 50 == 0 or i == total:
            print(f"[{i}/{total}] {os.path.basename(dataset_root)}")

    return compute_metrics_from_confmat(confmat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to SCTNet config")
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    ap.add_argument("--out-csv", required=True, help="Where to save CSV results")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    ap.add_argument("--num-classes", type=int, default=6)
    args = ap.parse_args()

    print("[INFO] Loading model...")
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    rows = []
    for name, root in SIM_DATASETS:
        print(f"\n[INFO] Evaluating on: {name}")
        metrics = evaluate_dataset(model, root, num_classes=args.num_classes)
        row = {
            "dataset": name,
            "aAcc": metrics["aAcc"] * 100.0,
            "mIoU": metrics["mIoU"] * 100.0,
            "mAcc": metrics["mAcc"] * 100.0,
        }
        rows.append(row)
        print(f"[RESULT] {name} -> aAcc={row['aAcc']:.2f}, mIoU={row['mIoU']:.2f}, mAcc={row['mAcc']:.2f}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "aAcc", "mIoU", "mAcc"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Saved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mydatasets.new_seg_dataset import NewSegDataset
from models.ddrnet23 import get_ddrnet23

NUM_CLASSES = 6
IGNORE_INDEX = 255

DATASETS = [
    ("Different Lighting Conditions", "/data/pool/qmc-41b/dataset_diff_lightning_conditions"),
    ("Fog Mist or Shallow Fog 500m Visibility", "/data/pool/qmc-41b/fog_mist_500m_visibility"),
    ("Moderate Fog 200m visibility", "/data/pool/qmc-41b/moderate_fog_200m_visibility"),
    ("Rain_light 0.4 mm mm2 per hour", "/data/pool/qmc-41b/rain_light_0p4mm_per_hour"),
    ("Snow_middle 4 mm mm2 per hour", "/data/pool/qmc-41b/snow_middle_4mm_per_hour"),
    ("Snow_Heavy 20 mm mm2 per hour", "/data/pool/qmc-41b/snow_heavy_20mm_per_hour"),
    ("Snow_Light Less than 0.5 mm2 per hour", "/data/pool/qmc-41b/snow_light_less_0p5mm_per_hour"),
]

def update_confusion_matrix(confmat, preds, labels, num_classes, ignore_index=255):
    valid = labels != ignore_index
    preds = preds[valid]
    labels = labels[valid]
    if preds.numel() == 0:
        return confmat
    inds = labels * num_classes + preds
    bincount = torch.bincount(inds, minlength=num_classes * num_classes)
    confmat += bincount.reshape(num_classes, num_classes).cpu().numpy()
    return confmat

def compute_segmentation_metrics(confmat):
    eps = 1e-10
    intersection = np.diag(confmat)
    gt_sum = confmat.sum(axis=1)
    pred_sum = confmat.sum(axis=0)
    union = gt_sum + pred_sum - intersection
    iou = intersection / np.maximum(union, eps)
    acc = intersection / np.maximum(gt_sum, eps)
    total_correct = intersection.sum()
    total_pixels = gt_sum.sum()
    aacc = total_correct / np.maximum(total_pixels, eps)
    miou = np.nanmean(iou)
    macc = np.nanmean(acc)
    return {"IoU": iou, "Acc": acc, "aAcc": aacc, "mIoU": miou, "mAcc": macc}

@torch.no_grad()
def evaluate_dataset(model, data_root, device, batch_size, num_workers, width, height):
    split_file = os.path.join(data_root, "splits", "test.txt")
    ds = NewSegDataset(
        data_root=data_root,
        split_file=split_file,
        image_size=(width, height),
        augment=False,
        ignore_index=IGNORE_INDEX,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    confmat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_images = 0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        if outputs.shape[-2:] != labels.shape[-2:]:
            outputs = F.interpolate(outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)

        preds = torch.argmax(outputs, dim=1)
        confmat = update_confusion_matrix(confmat, preds, labels, NUM_CLASSES, IGNORE_INDEX)
        total_images += images.shape[0]

    return total_images, compute_segmentation_metrics(confmat)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = get_ddrnet23(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "split", "num_images", "aAcc", "mIoU", "mAcc"])

        for dataset_name, data_root in DATASETS:
            print(f"Evaluating {args.model_name} on {dataset_name}")
            total_images, m = evaluate_dataset(
                model, data_root, device,
                args.batch_size, args.num_workers, args.width, args.height
            )
            w.writerow([
                args.model_name,
                dataset_name,
                "test",
                total_images,
                m["aAcc"] * 100,
                m["mIoU"] * 100,
                m["mAcc"] * 100,
            ])
            f.flush()
            print(f"mIoU = {m['mIoU'] * 100:.2f}")

    print("Saved:", args.out_csv)

if __name__ == "__main__":
    main()

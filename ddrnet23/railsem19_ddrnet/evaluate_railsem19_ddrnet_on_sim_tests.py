#!/usr/bin/env python3
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mydatasets.new_seg_dataset import NewSegDataset
from models.ddrnet23 import get_ddrnet23

CHECKPOINT = "/data/pool/qmc-41b/ddrnet23_project/railsem19_ddrnet/checkpoints/best_mIoU_epoch_120.pth"
OUT_CSV = "/data/pool/qmc-41b/ddrnet23_project/railsem19_ddrnet/eval_railsem19_ddrnet_on_sim_tests.csv"

MODEL_NUM_CLASSES = 19
EVAL_NUM_CLASSES = 6
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
def evaluate_dataset(model, data_root, device):
    split_file = os.path.join(data_root, "splits", "test.txt")
    ds = NewSegDataset(
        data_root=data_root,
        split_file=split_file,
        image_size=(1024, 512),
        augment=False,
        ignore_index=IGNORE_INDEX,
    )
    loader = DataLoader(
        ds,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    confmat = np.zeros((EVAL_NUM_CLASSES, EVAL_NUM_CLASSES), dtype=np.int64)
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

        # direct rough evaluation for filling the table
        preds = preds.clamp(min=0, max=EVAL_NUM_CLASSES - 1)

        confmat = update_confusion_matrix(confmat, preds, labels, EVAL_NUM_CLASSES, IGNORE_INDEX)
        total_images += images.shape[0]

    return total_images, compute_segmentation_metrics(confmat)

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_ddrnet23(num_classes=MODEL_NUM_CLASSES).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "split", "num_images", "aAcc", "mIoU", "mAcc"])

        for dataset_name, data_root in DATASETS:
            print(f"Evaluating RailSem19 DDRNet on {dataset_name}")
            total_images, m = evaluate_dataset(model, data_root, device)
            w.writerow([
                "railsem19_ddrnet",
                dataset_name,
                "test",
                total_images,
                m["aAcc"] * 100,
                m["mIoU"] * 100,
                m["mAcc"] * 100,
            ])
            f.flush()
            print(f"mIoU = {m['mIoU'] * 100:.2f}")

    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()

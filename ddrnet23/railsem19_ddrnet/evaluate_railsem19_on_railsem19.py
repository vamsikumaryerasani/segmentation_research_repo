#!/usr/bin/env python3
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mydatasets.railsem19 import RailSem19Dataset
from models.ddrnet23 import get_ddrnet23

NUM_CLASSES = 19
IGNORE_INDEX = 255
DATA_ROOT = "/data/pool/qmc-41b/rs19_val_aug_bri_0p5_0p6_0p7_con_0p6_0p7"
SPLIT_FILE = "/data/pool/qmc-41b/railsem19_eval_fixed_splits/test.txt"
CHECKPOINT = "/data/pool/qmc-41b/ddrnet23_project/railsem19_ddrnet/checkpoints/best_mIoU_epoch_120.pth"
OUT_CSV = "/data/pool/qmc-41b/ddrnet23_project/railsem19_ddrnet/eval_railsem19_on_railsem19_test.csv"

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
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RailSem19Dataset(
        data_root=DATA_ROOT,
        split_file=SPLIT_FILE,
        image_size=(1024, 512),
        augment=False,
        ignore_index=IGNORE_INDEX,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    model = get_ddrnet23(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()

    confmat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_images = 0

    for batch in tqdm(loader):
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

    m = compute_segmentation_metrics(confmat)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","dataset","split","num_images","aAcc","mIoU","mAcc"])
        w.writerow(["railsem19_19class","railsem19","test",total_images,m["aAcc"]*100,m["mIoU"]*100,m["mAcc"]*100])

    print("aAcc:", round(m["aAcc"]*100, 2))
    print("mIoU:", round(m["mIoU"]*100, 2))
    print("mAcc:", round(m["mAcc"]*100, 2))
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()

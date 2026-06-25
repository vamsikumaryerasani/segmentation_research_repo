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


def pixel_accuracy(pred, target, ignore_index=255):
    valid = target != ignore_index
    if valid.sum() == 0:
        return 0.0
    correct = (pred[valid] == target[valid]).sum().item()
    total = valid.sum().item()
    return correct / total


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

    return {
        "IoU": iou,
        "Acc": acc,
        "aAcc": aacc,
        "mIoU": miou,
        "mAcc": macc,
    }


@torch.no_grad()
def evaluate_one(model, loader, device, num_classes, ignore_index=255):
    model.eval()

    total_acc = 0.0
    count = 0
    confmat = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_images = 0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        if outputs.shape[-2:] != labels.shape[-2:]:
            outputs = F.interpolate(
                outputs,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        preds = torch.argmax(outputs, dim=1)

        total_acc += pixel_accuracy(preds, labels, ignore_index)
        count += 1
        total_images += images.shape[0]

        confmat = update_confusion_matrix(
            confmat, preds, labels, num_classes=num_classes, ignore_index=ignore_index
        )

    metrics = compute_segmentation_metrics(confmat)

    return {
        "acc_batch_avg": total_acc / max(count, 1),
        "metrics": metrics,
        "num_batches": count,
        "num_images": total_images,
    }


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    return model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--num_classes", type=int, default=6)
    ap.add_argument("--ignore_index", type=int, default=255)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--out_csv",
        type=str,
        default="/data/pool/qmc-41b/ddrnet23_project/railsem19_ddrnet/eval_results_test_only.csv",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    models_to_eval = [
        ("diff_lightning", "/data/pool/qmc-41b/ddrnet23_project/railsem19_ddrnet/checkpoints_diff_lightning/latest.pth"),
        ("snow_light", "/data/pool/qmc-41b/ddrnet23_project/railsem19_ddrnet/checkpoints_snow_light/latest.pth"),
        ("snow_middle_4mm", "/data/pool/qmc-41b/ddrnet23_project/railsem19_ddrnet/checkpoints_snow_middle_4mm/latest.pth"),
        ("rain_light_0p4mm", "/data/pool/qmc-41b/ddrnet23_project/railsem19_ddrnet/checkpoints_rain_light_0p4mm/latest.pth"),
        ("fog_mist_500m", "/data/pool/qmc-41b/ddrnet23_project/railsem19_ddrnet/checkpoints_fog_mist_500m/latest.pth"),
        ("snow_heavy_20mm", "/data/pool/qmc-41b/ddrnet23_project/railsem19_ddrnet/checkpoints_snow_heavy_20mm/latest.pth"),
        ("moderate_fog_200m", "/data/pool/qmc-41b/ddrnet23_project/railsem19_ddrnet/checkpoints_moderate_fog_200m/latest.pth"),
        ("combined_weather", "/data/pool/qmc-41b/ddrnet23_project/railsem19_ddrnet/checkpoints_combined_weather/latest.pth"),
    ]

    datasets_to_eval = [
        ("diff_lightning", "/data/pool/qmc-41b/dataset_diff_lightning_conditions"),
        ("snow_light", "/data/pool/qmc-41b/snow_light_less_0p5mm_per_hour"),
        ("snow_middle_4mm", "/data/pool/qmc-41b/snow_middle_4mm_per_hour"),
        ("rain_light_0p4mm", "/data/pool/qmc-41b/rain_light_0p4mm_per_hour"),
        ("fog_mist_500m", "/data/pool/qmc-41b/fog_mist_500m_visibility"),
        ("snow_heavy_20mm", "/data/pool/qmc-41b/snow_heavy_20mm_per_hour"),
        ("moderate_fog_200m", "/data/pool/qmc-41b/moderate_fog_200m_visibility"),
    ]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "dataset",
            "split",
            "num_images",
            "num_batches",
            "aAcc",
            "mIoU",
            "mAcc",
            "class0_IoU",
            "class1_IoU",
            "class2_IoU",
            "class3_IoU",
            "class4_IoU",
            "class5_IoU",
        ])

        for model_name, ckpt_path in models_to_eval:
            if not os.path.isfile(ckpt_path):
                print(f"[WARN] Missing checkpoint, skipping: {ckpt_path}")
                continue

            print(f"\n=== Loading model: {model_name} ===")
            model = get_ddrnet23(num_classes=args.num_classes).to(device)
            model = load_checkpoint(model, ckpt_path, device)

            for dataset_name, data_root in datasets_to_eval:
                split_file = os.path.join(data_root, "splits", "test.txt")
                if not os.path.isfile(split_file):
                    print(f"[WARN] Missing split, skipping: {split_file}")
                    continue

                print(f"Evaluating model={model_name} on dataset={dataset_name} split=test")

                test_set = NewSegDataset(
                    data_root=data_root,
                    split_file=split_file,
                    image_size=(args.width, args.height),
                    augment=False,
                    ignore_index=args.ignore_index,
                )

                test_loader = DataLoader(
                    test_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=False,
                )

                out = evaluate_one(
                    model=model,
                    loader=test_loader,
                    device=device,
                    num_classes=args.num_classes,
                    ignore_index=args.ignore_index,
                )

                m = out["metrics"]
                row = [
                    model_name,
                    dataset_name,
                    "test",
                    out["num_images"],
                    out["num_batches"],
                    float(m["aAcc"]) * 100.0,
                    float(m["mIoU"]) * 100.0,
                    float(m["mAcc"]) * 100.0,
                    float(m["IoU"][0]) * 100.0,
                    float(m["IoU"][1]) * 100.0,
                    float(m["IoU"][2]) * 100.0,
                    float(m["IoU"][3]) * 100.0,
                    float(m["IoU"][4]) * 100.0,
                    float(m["IoU"][5]) * 100.0,
                ]
                writer.writerow(row)
                f.flush()

                print(
                    f"[OK] model={model_name} dataset={dataset_name} "
                    f"aAcc={row[5]:.2f} mIoU={row[6]:.2f} mAcc={row[7]:.2f}"
                )

    print(f"\nSaved results to: {args.out_csv}")


if __name__ == "__main__":
    main()

import os
import csv
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mydatasets.railsem19 import RailSem19Dataset
from models.ddrnet23 import get_ddrnet23


LOG_FH = None


def init_log_file(save_dir: str):
    global LOG_FH
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train.log")
    LOG_FH = open(log_path, "a", buffering=1)


def close_log_file():
    global LOG_FH
    if LOG_FH is not None:
        LOG_FH.close()
        LOG_FH = None


def log_info(message: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    line = f"{now} - mmseg - INFO - {message}"
    print(line)
    if LOG_FH is not None:
        LOG_FH.write(line + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=16)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--early_stop_start_epoch", type=int, default=1)

    parser.add_argument("--resume_from", type=str, default="")

    return parser.parse_args()


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


CLASS_NAMES = [
    "class_0", "class_1", "class_2", "class_3", "class_4",
    "class_5", "class_6", "class_7", "class_8", "class_9",
    "class_10", "class_11", "class_12", "class_13", "class_14",
    "class_15", "class_16", "class_17", "class_18",
]

def print_class_table(metrics, num_classes):
    log_info("+----------------+--------+--------+")
    log_info("| Class Name     |  IoU   |  Acc   |")
    log_info("+----------------+--------+--------+")
    for cls in range(num_classes):
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        iou = metrics["IoU"][cls] * 100.0
        acc = metrics["Acc"][cls] * 100.0
        log_info(f"| {name:<14} | {iou:>6.2f} | {acc:>6.2f} |")
    log_info("+----------------+--------+--------+")

def print_summary_table(metrics):
    log_info("+--------+--------+--------+")
    log_info("|  aAcc  |  mIoU  |  mAcc  |")
    log_info("+--------+--------+--------+")
    log_info(
        f"| {metrics['aAcc']*100:>6.2f} | {metrics['mIoU']*100:>6.2f} | {metrics['mAcc']*100:>6.2f} |"
    )
    log_info("+--------+--------+--------+")


def init_csv(csv_path: str):
    if os.path.exists(csv_path):
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "aAcc",
            "mIoU",
            "mAcc",
            "best_val_loss",
            "best_mIoU",
            "early_stop_bad",
            "val_images",
            "val_batches",
        ])


def append_csv_row(csv_path: str, row: dict):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["epoch"],
            row["train_loss"],
            row["train_acc"],
            row["val_loss"],
            row["val_acc"],
            row["aAcc"],
            row["mIoU"],
            row["mAcc"],
            row["best_val_loss"],
            row["best_mIoU"],
            row["early_stop_bad"],
            row["val_images"],
            row["val_batches"],
        ])


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, ignore_index=255):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    count = 0
    confmat = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_images = 0

    for batch in tqdm(loader, desc="Val", leave=False):
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

        loss = criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        total_loss += loss.item()
        total_acc += pixel_accuracy(preds, labels, ignore_index)
        count += 1
        total_images += images.shape[0]

        confmat = update_confusion_matrix(
            confmat, preds, labels, num_classes=num_classes, ignore_index=ignore_index
        )

    metrics = compute_segmentation_metrics(confmat)

    return {
        "val_loss": total_loss / max(count, 1),
        "val_acc": total_acc / max(count, 1),
        "metrics": metrics,
        "num_batches_used": count,
        "num_images_used": total_images,
    }


def save_checkpoint(path, epoch, model, optimizer, best_val_loss, best_miou, epochs_without_improvement):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_miou": best_miou,
            "epochs_without_improvement": epochs_without_improvement,
        },
        path,
    )


def update_latest_symlink(save_dir: str, target_filename: str):
    latest_path = os.path.join(save_dir, "latest.pth")
    target_path = os.path.join(save_dir, target_filename)

    try:
        if os.path.islink(latest_path) or os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(target_filename, latest_path)
    except OSError:
        import shutil
        shutil.copy2(target_path, latest_path)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    init_log_file(args.save_dir)

    csv_path = os.path.join(args.save_dir, "metrics.csv")
    init_csv(csv_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_set = RailSem19Dataset(
        data_root=args.data_root,
        split_file=os.path.join(args.split_dir, "train.txt"),
        image_size=(args.width, args.height),
        augment=True,
        ignore_index=args.ignore_index,
    )

    val_set = RailSem19Dataset(
        data_root=args.data_root,
        split_file=os.path.join(args.split_dir, "val.txt"),
        image_size=(args.width, args.height),
        augment=False,
        ignore_index=args.ignore_index,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = get_ddrnet23(num_classes=args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_miou = -1.0
    epochs_without_improvement = 0
    best_ckpt_path = None
    start_epoch = 0

    if args.resume_from:
        log_info(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_miou = ckpt.get("best_miou", -1.0)
        epochs_without_improvement = 0

        log_info(
            f"Resumed state | start_epoch={start_epoch} "
            f"best_val_loss={best_val_loss:.6f} best_mIoU={best_miou:.4f} "
            f"early_stop_bad={epochs_without_improvement}/{args.early_stop_patience}"
        )

    log_info(
        f"Starting training | monitor=val.loss rule=less patience={args.early_stop_patience} | "
        f"train_images={len(train_set)} val_images={len(val_set)} "
        f"train_bs={args.batch_size} val_bs={args.val_batch_size}"
    )

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            step_count = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for batch in pbar:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                optimizer.zero_grad()

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

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs, dim=1)
                acc = pixel_accuracy(preds, labels, args.ignore_index)

                epoch_loss += loss.item()
                epoch_acc += acc
                step_count += 1

                pbar.set_postfix(
                    acc=f"{epoch_acc/step_count:.4f}",
                    loss=f"{epoch_loss/step_count:.4f}",
                )

            train_loss = epoch_loss / max(step_count, 1)
            train_acc = epoch_acc / max(step_count, 1)

            eval_out = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
            )

            val_loss = eval_out["val_loss"]
            val_acc = eval_out["val_acc"]
            metrics = eval_out["metrics"]
            val_batches = eval_out["num_batches_used"]
            val_images = eval_out["num_images_used"]

            print_class_table(metrics, args.num_classes)
            print_summary_table(metrics)

            current_miou = metrics["mIoU"]
            improved_miou = current_miou > best_miou
            if improved_miou:
                old_best = best_ckpt_path
                best_miou = current_miou

                best_ckpt_path = os.path.join(args.save_dir, f"best_mIoU_epoch_{epoch+1}.pth")
                save_checkpoint(
                    best_ckpt_path,
                    epoch + 1,
                    model,
                    optimizer,
                    best_val_loss,
                    best_miou,
                    epochs_without_improvement,
                )

                if old_best is not None and old_best != best_ckpt_path and os.path.exists(old_best):
                    os.remove(old_best)

            improved = val_loss < (best_val_loss - args.early_stop_min_delta)

            if epoch + 1 >= args.early_stop_start_epoch:
                if improved:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            epoch_filename = f"epoch_{epoch+1}.pth"
            epoch_ckpt_path = os.path.join(args.save_dir, epoch_filename)
            save_checkpoint(
                epoch_ckpt_path,
                epoch + 1,
                model,
                optimizer,
                best_val_loss,
                best_miou,
                epochs_without_improvement,
            )

            update_latest_symlink(args.save_dir, epoch_filename)

            append_csv_row(csv_path, {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "aAcc": metrics["aAcc"],
                "mIoU": metrics["mIoU"],
                "mAcc": metrics["mAcc"],
                "best_val_loss": best_val_loss,
                "best_mIoU": best_miou,
                "early_stop_bad": epochs_without_improvement,
                "val_images": val_images,
                "val_batches": val_batches,
            })

            log_info(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"train_loss={train_loss:.6f} train_acc={train_acc*100:.4f} | "
                f"val_loss={val_loss:.6f} val_acc={val_acc*100:.4f} | "
                f"mIoU={metrics['mIoU']*100:.2f} mAcc={metrics['mAcc']*100:.2f} aAcc={metrics['aAcc']*100:.2f} | "
                f"val_images={val_images} val_batches={val_batches} | "
                f"best_val_loss={best_val_loss:.6f} best_mIoU={best_miou:.4f} | "
                f"early_stop_bad={epochs_without_improvement}/{args.early_stop_patience}"
            )

            log_info(f"Saved epoch checkpoint: {epoch_filename}")
            log_info("Updated latest.pth")

            if improved_miou:
                log_info(f"New best checkpoint: {os.path.basename(best_ckpt_path)}")

            if epochs_without_improvement >= args.early_stop_patience:
                log_info(f"[EarlyStopping] Stop triggered at epoch={epoch+1}")
                break

        log_info("Training finished")
        log_info(f"CSV metrics saved to: {csv_path}")
        log_info(f"Log file saved to: {os.path.join(args.save_dir, 'train.log')}")

    finally:
        close_log_file()


if __name__ == "__main__":
    main()

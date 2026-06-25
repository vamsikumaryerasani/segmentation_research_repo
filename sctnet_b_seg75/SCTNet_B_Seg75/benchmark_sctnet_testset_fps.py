#!/usr/bin/env python3
import time
import argparse
import torch
torch.backends.cudnn.benchmark = True

from mmcv import Config
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor


def load_model_from_config(config_path, checkpoint_path, device):
    cfg = Config.fromfile(config_path)
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, cfg


def unwrap_imgs(x):
    # img can be DataContainer -> [tensor]
    if hasattr(x, "data"):
        x = x.data[0]
    elif isinstance(x, list) and len(x) == 1 and torch.is_tensor(x[0]):
        x = x[0]
    return x


def unwrap_img_metas(x):
    # img_metas can be DataContainer -> [[dict, dict, ...]]
    if hasattr(x, "data"):
        x = x.data[0]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model_from_config(args.config, args.checkpoint, device)

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=args.num_workers,
        dist=False,
        shuffle=False,
    )

    total_images = 0

    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for batch in loader:
            imgs = unwrap_imgs(batch["img"])
            img_metas = unwrap_img_metas(batch["img_metas"])

            imgs = imgs.to(device, non_blocking=True)
            bs = imgs.shape[0]

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                _ = model(img=[imgs], img_metas=[img_metas], return_loss=False)

            total_images += bs

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    total_time = end - start
    fps = total_images / total_time
    ms_per_image = (total_time / total_images) * 1000

    print(f"Device: {device}")
    print(f"Total images: {total_images}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total time: {total_time:.6f} s")
    print(f"Average time per image: {ms_per_image:.3f} ms")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    main()

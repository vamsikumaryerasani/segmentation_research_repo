#!/usr/bin/env python3
import os
import csv
import json
import argparse
import tempfile

from mmcv import Config
from mmseg.apis import init_segmentor, single_gpu_test
from mmseg.datasets import build_dataset, build_dataloader
from mmcv.parallel import MMDataParallel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--img-dir", default="images_new")
    ap.add_argument("--ann-dir", default="masks_new")
    ap.add_argument("--split", default="splits_85_10_5/test.txt")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = Config.fromfile(args.base_config)

    cfg.data.test.data_root = args.data_root
    cfg.data.test.img_dir = args.img_dir
    cfg.data.test.ann_dir = args.ann_dir
    cfg.data.test.split = args.split
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 2
    cfg.model.train_cfg = None

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        cfg.dump(f.name)
        tmp_cfg_path = f.name

    print("[INFO] Using config:", tmp_cfg_path)
    print("[INFO] Using checkpoint:", args.checkpoint)
    print("[INFO] Evaluating RailSem19 test split:", args.split)

    model = init_segmentor(tmp_cfg_path, args.checkpoint, device=args.device)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False
    )

    model = MMDataParallel(model, device_ids=[0])
    results = single_gpu_test(model, data_loader, show=False)
    metrics = dataset.evaluate(results, metric='mIoU')

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "aAcc", "mIoU", "mAcc"])
        writer.writerow(["RailSem19 test", metrics.get("aAcc", ""), metrics.get("mIoU", ""), metrics.get("mAcc", "")])

    print("[RESULT]", json.dumps({
        "dataset": "RailSem19 test",
        "aAcc": metrics.get("aAcc", ""),
        "mIoU": metrics.get("mIoU", ""),
        "mAcc": metrics.get("mAcc", ""),
    }, indent=2))
    print("[OK] Wrote CSV:", args.out_csv)

    os.remove(tmp_cfg_path)

if __name__ == "__main__":
    main()

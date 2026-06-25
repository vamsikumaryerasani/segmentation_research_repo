#!/usr/bin/env python3
import os
import csv
import argparse
import tempfile

from mmcv import Config
from mmseg.apis import init_segmentor, single_gpu_test
from mmseg.datasets import build_dataset, build_dataloader
from mmcv.parallel import MMDataParallel

SIM_DATASETS = [
    ("Different Lighting conditions", "/data/pool/qmc-41b/dataset_diff_lightning_conditions"),
    ("Fog Mist or Shallow Fog 500m Visibility", "/data/pool/qmc-41b/fog_mist_500m_visibility"),
    ("Moderate Fog 200m visibility", "/data/pool/qmc-41b/moderate_fog_200m_visibility"),
    ("Rain_light 0.4 mm mm2 per hour", "/data/pool/qmc-41b/rain_light_0p4mm_per_hour"),
    ("Snow_middle 4 mm mm2 per hour", "/data/pool/qmc-41b/snow_middle_4mm_per_hour"),
    ("Snow_Heavy 20 mm mm2 per hour", "/data/pool/qmc-41b/snow_heavy_20mm_per_hour"),
    ("Snow_Light Less than 0.5 mm2 per hour", "/data/pool/qmc-41b/snow_light_less_0p5mm_per_hour"),
]

CLASSES = ('sky', 'terrain', 'nature', 'car', 'building', 'railway')
PALETTE = [
    [0, 0, 0],
    [128, 64, 128],
    [107, 142, 35],
    [0, 0, 142],
    [70, 70, 70],
    [153, 153, 153],
]

def make_eval_cfg(base_config_path, data_root):
    cfg = Config.fromfile(base_config_path)

    cfg.data.test.type = 'CustomDataset'
    cfg.data.test.data_root = data_root
    cfg.data.test.img_dir = 'images'
    cfg.data.test.ann_dir = 'masks'
    cfg.data.test.split = 'splits/test.txt'
    cfg.data.test.img_suffix = '.jpg'
    cfg.data.test.seg_map_suffix = '.png'
    cfg.data.test.classes = CLASSES
    cfg.data.test.palette = PALETTE

    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 2

    cfg.model.train_cfg = None
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True, help="Any 6-class SCTNet config")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate")
    ap.add_argument("--out-csv", required=True, help="CSV output path")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    rows = []

    for dataset_name, data_root in SIM_DATASETS:
        print(f"\n[INFO] Evaluating on: {dataset_name}")
        cfg = make_eval_cfg(args.base_config, data_root)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            cfg.dump(f.name)
            tmp_cfg_path = f.name

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
        row = {
            "dataset": dataset_name,
            "aAcc": metrics.get("aAcc", ""),
            "mIoU": metrics.get("mIoU", ""),
            "mAcc": metrics.get("mAcc", ""),
        }
        print(f"[RESULT] {dataset_name} | aAcc={row['aAcc']} mIoU={row['mIoU']} mAcc={row['mAcc']}")
        rows.append(row)

        os.remove(tmp_cfg_path)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "aAcc", "mIoU", "mAcc"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Wrote CSV: {args.out_csv}")

if __name__ == "__main__":
    main()

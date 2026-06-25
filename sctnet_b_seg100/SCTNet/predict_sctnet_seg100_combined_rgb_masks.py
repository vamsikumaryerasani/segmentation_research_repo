#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import torch

from mmcv import Config
from mmseg.models import build_segmentor
from mmseg.apis import inference_segmentor

PALETTE = np.array([
    [0, 0, 0],        # 0 sky
    [152, 251, 152],  # 1 terrain
    [107, 142, 35],   # 2 nature
    [0, 0, 142],      # 3 car
    [70, 70, 70],     # 4 building
    [185, 117, 131],  # 5 railway track
], dtype=np.uint8)

IMAGES = [
    "/data/pool/qmc-41b/dataset_diff_lightning_conditions/images/image_0006.jpg",
    "/data/pool/qmc-41b/snow_middle_4mm_per_hour/images/image_0094.jpg",
    "/data/pool/qmc-41b/snow_light_less_0p5mm_per_hour/images/image_0110.jpg",
    "/data/pool/qmc-41b/fog_mist_500m_visibility/images/image_0280.jpg",
    "/data/pool/qmc-41b/snow_heavy_20mm_per_hour/images/image_0280.jpg",
    "/data/pool/qmc-41b/moderate_fog_200m_visibility/images/image_0348.jpg",
]

CONFIG = "/data/pool/qmc-41b/SCTNet-B-Seg100/SCTNet/configs/sctnet_newdatasets/sctnet_b_seg100_combined_weather.py"
CHECKPOINT = "/data/pool/qmc-41b/SCTNet-B-Seg100/SCTNet/work_dirs/sctnet_b_seg100_combined_weather/best_mIoU_epoch_28.pth"
OUT_DIR = Path("/data/pool/qmc-41b/predictions_triptych_style/sctnet_b_seg100")

def colorize(mask):
    return PALETTE[mask]

def get_mask_path(img_path: Path) -> Path:
    return img_path.parent.parent / "masks" / f"{img_path.stem}.png"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = Config.fromfile(CONFIG)
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    model.cfg = cfg
    ckpt = torch.load(CHECKPOINT, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    for img_path_str in IMAGES:
        img_path = Path(img_path_str)
        mask_path = get_mask_path(img_path)

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Failed to read image: {img_path}")
            continue

        if not mask_path.exists():
            print(f"Missing GT mask: {mask_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        gt_mask = np.array(Image.open(mask_path))
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[..., 0]

        pred = inference_segmentor(model, str(img_path))[0].astype(np.uint8)

        gt_rgb = colorize(gt_mask.astype(np.uint8))
        pred_rgb = colorize(pred)

        prefix = f"{img_path.parent.parent.name}_{img_path.stem}"
        Image.fromarray(img_rgb).save(OUT_DIR / f"{prefix}_original.png")
        Image.fromarray(gt_rgb).save(OUT_DIR / f"{prefix}_gt_rgb.png")
        Image.fromarray(pred_rgb).save(OUT_DIR / f"{prefix}_pred_rgb.png")
        print("saved", prefix)

if __name__ == "__main__":
    main()

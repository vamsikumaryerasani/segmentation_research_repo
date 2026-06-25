#!/usr/bin/env python3
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/pool/qmc-41b/ddrnet23_project/railsem19_ddrnet")
from models.ddrnet23 import get_ddrnet23

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

CHECKPOINT = "/data/pool/qmc-41b/ddrnet23_project/railsem19_ddrnet/checkpoints_combined_weather/latest.pth"
OUT_DIR = Path("/data/pool/qmc-41b/predictions_triptych_style/ddrnet23")

def colorize(mask):
    return PALETTE[mask]

def get_mask_path(img_path: Path) -> Path:
    return img_path.parent.parent / "masks" / f"{img_path.stem}.png"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_ddrnet23(num_classes=6).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()

    with torch.inference_mode():
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

            gt_mask = np.array(Image.open(mask_path))
            if gt_mask.ndim == 3:
                gt_mask = gt_mask[..., 0]

            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            x = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            x = x.to(device)
            x_in = F.interpolate(x, size=(512, 1024), mode="bilinear", align_corners=False)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                y = model(x_in)
                if isinstance(y, (list, tuple)):
                    y = y[0]

            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            pred = y.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

            gt_rgb = colorize(gt_mask.astype(np.uint8))
            pred_rgb = colorize(pred)

            prefix = f"{img_path.parent.parent.name}_{img_path.stem}"
            Image.fromarray(img_rgb).save(OUT_DIR / f"{prefix}_original.png")
            Image.fromarray(gt_rgb).save(OUT_DIR / f"{prefix}_gt_rgb.png")
            Image.fromarray(pred_rgb).save(OUT_DIR / f"{prefix}_pred_rgb.png")
            print("saved", prefix)

if __name__ == "__main__":
    main()

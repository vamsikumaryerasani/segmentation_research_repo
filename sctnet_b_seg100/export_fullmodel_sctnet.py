import torch
from mmcv import Config
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint

config_path = "configs/sctnet_railsem/sctnet_b_seg100_rs19_aug_85_10_5.py"
ckpt_path = "work_dirs/sctnet_b_seg100_rs19_aug_85_10_5_bs32_e150_val5100_test2550_fresh/best_mIoU_epoch_98.pth"
out_path = "work_dirs/sctnet_b_seg100_export_for_ros2/sctnet_b_seg100_FULLMODEL.pth"

cfg = Config.fromfile(config_path)
cfg.model.pretrained = None

model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, ckpt_path, map_location="cpu")
model.eval()

torch.save(model, out_path)
print("Saved:", out_path)

#!/usr/bin/env python3
import torch
import torch.nn as nn
from mmcv import Config
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint

config_path = "configs/sctnet_railsem/sctnet_b_seg100_rs19_aug_85_10_5.py"
ckpt_path = "work_dirs/sctnet_b_seg100_rs19_aug_85_10_5_bs32_e150_val5100_test2550_fresh/best_mIoU_epoch_98.pth"
out_path = "work_dirs/sctnet_b_seg100_export_for_ros2/sctnet_b_seg100.ts.pt"


def convert_syncbn_to_bn(module):
    """Recursively replace SyncBatchNorm with BatchNorm2d."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(
            module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight.copy_(module.weight)
                module_output.bias.copy_(module.bias)
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked

    for name, child in module.named_children():
        module_output.add_module(name, convert_syncbn_to_bn(child))
    return module_output


cfg = Config.fromfile(config_path)
cfg.model.pretrained = None

model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, ckpt_path, map_location="cpu")
model = convert_syncbn_to_bn(model)
model.eval()


class SegWrapper(nn.Module):
    def __init__(self, seg_model):
        super().__init__()
        self.seg_model = seg_model

    def forward(self, img):
        n, c, h, w = img.shape
        img_metas = []
        for _ in range(n):
            img_metas.append(
                dict(
                    ori_shape=(h, w, c),
                    img_shape=(h, w, c),
                    pad_shape=(h, w, c),
                    scale_factor=1.0,
                    flip=False,
                    flip_direction=None,
                )
            )
        return self.seg_model.encode_decode(img, img_metas)


wrapper = SegWrapper(model).eval()
dummy = torch.randn(1, 3, 540, 960)

with torch.no_grad():
    traced = torch.jit.trace(wrapper, dummy, strict=False)
    traced.save(out_path)

print("Saved:", out_path)

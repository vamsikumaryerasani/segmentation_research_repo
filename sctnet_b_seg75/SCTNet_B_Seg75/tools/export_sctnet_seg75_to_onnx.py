import torch
import torch.nn as nn
from mmcv import Config
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint

CONFIG = "/data/pool/qmc-41b/work_dirs/sctnet_b_seg75_rs19_aug_85_10_5_epoch400_es_valloss/sctnet-b_seg75_railsem19_aug_85_10_5.py"
CKPT = "/data/pool/qmc-41b/work_dirs/sctnet_b_seg75_rs19_aug_85_10_5_epoch400_es_valloss/best_mIoU_epoch_66.pth"
OUT = "/data/pool/qmc-41b/share_sctnet_b_seg75/sctnet_b_seg75_railsem19.onnx"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        img_metas = [[{
            "img_shape": (h, w, 3),
            "ori_shape": (h, w, 3),
            "pad_shape": (h, w, 3),
            "filename": "dummy.jpg",
            "scale_factor": 1.0,
            "flip": False
        }]]
        out = self.model.encode_decode(x, img_metas)
        return out


def main():
    cfg = Config.fromfile(CONFIG)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, CKPT, map_location="cpu")

    model.to(DEVICE)
    model.eval()

    wrapped = ONNXWrapper(model).to(DEVICE)
    wrapped.eval()

    dummy = torch.randn(1, 3, 720, 1280, device=DEVICE)

    # warmup forward once
    with torch.no_grad():
        _ = wrapped(dummy)

    torch.onnx.export(
        wrapped,
        dummy,
        OUT,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
        export_params=True,
    )

    print("Saved:", OUT)


if __name__ == "__main__":
    main()

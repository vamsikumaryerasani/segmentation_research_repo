import torch
import torch.nn as nn

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class SCTHead(BaseDecodeHead):
    """mmseg-compatible SCTHead.

    forward() -> seg_logits (Tensor)
    forward_train() -> dict(losses)
    """

    def __init__(self, in_channels=256, channels=128, num_classes=19, **kwargs):
        # BaseDecodeHead handles in_index/input_transform/etc through kwargs
        super().__init__(in_channels=in_channels, channels=channels, num_classes=num_classes, **kwargs)

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        # BaseDecodeHead helper: select/transform backbone outputs
        x = self._transform_inputs(inputs)

        # Safety: if transform still returns a list/tuple, pick the first tensor
        if isinstance(x, (list, tuple)):
            picked = None
            for item in x:
                if torch.is_tensor(item):
                    picked = item
                    break
            if picked is None:
                raise TypeError(f"SCTHead.forward expected Tensor, got {type(x)} with no tensor elements")
            x = picked

        # Correct order: Conv (in_channels->channels) then BN over 'channels'
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.relu(self.bn2(x))
        seg_logits = self.cls_seg(x)
        return seg_logits

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        return self.forward(inputs)

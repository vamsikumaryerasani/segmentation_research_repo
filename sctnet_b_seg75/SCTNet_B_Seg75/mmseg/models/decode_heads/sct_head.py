import torch
import torch.nn as nn

from mmcv.cnn import ConvModule
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class SCTHead(BaseDecodeHead):
    """SCTHead for SCTNet.

    NOTE:
    MMSeg passes a list/tuple of multi-level backbone features to decode heads.
    This head selects the feature map whose channel dimension matches in_channels.
    """

    def __init__(self, **kwargs):
        super(SCTHead, self).__init__(**kwargs)

        # Simple head (as in repo): two conv blocks + classifier
        self.conv1 = nn.Conv2d(self.in_channels, self.channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.channels)

        self.cls_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (list[Tensor] | tuple[Tensor] | Tensor): multi-level features.

        Returns:
            decoder_feature (Tensor), seg_logits (Tensor)
        """
        # Select a single feature tensor
        if isinstance(inputs, (list, tuple)):
            x = None
            # Prefer the feature whose channels match self.in_channels (e.g., 256)
            for t in inputs[::-1]:
                if hasattr(t, "dim") and t.dim() == 4 and t.size(1) == self.in_channels:
                    x = t
                    break
            # Fallback: last feature map
            if x is None:
                x = inputs[-1]
        else:
            x = inputs

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        seg_logits = self.cls_seg(x)

        decoder_feature = x
        return decoder_feature, seg_logits

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward train.
        Must return (losses, decoder_feature, seg_logits) because encoder_decoder_distill expects that.
        """
        decoder_feature, seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses, decoder_feature, seg_logits

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward test/inference. Must return seg_logits."""
        _, seg_logits = self.forward(inputs)
        return seg_logits

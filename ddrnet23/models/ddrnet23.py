import os
import sys

DDRNET_REPO = "/data/pool/qmc-41b/ddrnet23_project/DDRNet"
SEG_DIR = os.path.join(DDRNET_REPO, "segmentation")

if SEG_DIR not in sys.path:
    sys.path.insert(0, SEG_DIR)


def get_ddrnet23(num_classes=19):
    from DDRNet_23 import DualResNet, BasicBlock

    model = DualResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        planes=64,
        spp_planes=128,
        head_planes=128,
        augment=False,
    )
    return model

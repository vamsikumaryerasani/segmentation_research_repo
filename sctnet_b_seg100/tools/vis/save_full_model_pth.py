import os
import torch
from mmseg.apis import init_segmentor

def main():
    cfg = "configs/sctnet_railsem/sctnet_b_seg100_railsem90_10_400e.py"
    ckpt = "work_dirs/SCTNet-B-Seg100_railsem90_10_400e_1gpu/best_mIoU_epoch_34.pth"
    out_dir = "work_dirs/SCTNet-B-Seg100_railsem90_10_400e_1gpu/export_pretrained"
    os.makedirs(out_dir, exist_ok=True)

    out_full = os.path.join(out_dir, "sctnet_b_seg100_FULLMODEL.pth")

    # Build model from config + checkpoint
    model = init_segmentor(cfg, ckpt, device="cpu")
    model.eval()

    # Save full pickled model (architecture + weights)
    # NOTE: this requires same codebase/imports to load elsewhere.
    torch.save(model, out_full)
    print("[OK] saved full model to:", out_full)

if __name__ == "__main__":
    main()

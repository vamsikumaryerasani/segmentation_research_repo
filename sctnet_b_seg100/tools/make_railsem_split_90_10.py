import os
import random
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.9

# Source (original dataset you already have)
SRC_ROOT = Path("/data/pool/qmc-41b/sctnet_clean_ws/SCTNet/data/railsem19_mmseg")
SRC_IMG_TRAIN = SRC_ROOT / "images_new" / "train"
SRC_IMG_VAL   = SRC_ROOT / "images_new" / "val"
SRC_MSK_TRAIN = SRC_ROOT / "masks_new"  / "train"
SRC_MSK_VAL   = SRC_ROOT / "masks_new"  / "val"

# Destination (NEW split folders inside this repo)
DST_ROOT = Path("data/railsem19_mmseg_90_10")
DST_IMG_TRAIN = DST_ROOT / "images" / "train"
DST_IMG_VAL   = DST_ROOT / "images" / "val"
DST_MSK_TRAIN = DST_ROOT / "masks"  / "train"
DST_MSK_VAL   = DST_ROOT / "masks"  / "val"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def gather_images(img_dir: Path):
    if not img_dir.exists():
        return []
    return [p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def find_mask_for_image(img_path: Path):
    stem = img_path.stem
    # masks are usually png; search in both original mask folders
    candidates = [
        SRC_MSK_TRAIN / f"{stem}.png",
        SRC_MSK_VAL   / f"{stem}.png",
        SRC_MSK_TRAIN / f"{stem}{img_path.suffix}",
        SRC_MSK_VAL   / f"{stem}{img_path.suffix}",
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback: slow search by stem
    for root in (SRC_MSK_TRAIN, SRC_MSK_VAL):
        hit = list(root.glob(stem + ".*"))
        if hit:
            return hit[0]
    return None

def safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.as_posix(), dst.as_posix())

def main():
    random.seed(SEED)

    imgs = gather_images(SRC_IMG_TRAIN) + gather_images(SRC_IMG_VAL)
    imgs = sorted(set(imgs))

    pairs = []
    missing = 0
    for img in imgs:
        msk = find_mask_for_image(img)
        if msk is None:
            missing += 1
            continue
        pairs.append((img, msk))

    if not pairs:
        raise SystemExit("No (image, mask) pairs found. Check paths and filenames.")

    random.shuffle(pairs)
    n_total = len(pairs)
    n_train = int(n_total * TRAIN_RATIO)
    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:]

    # Clear old split (only our generated folder)
    if DST_ROOT.exists():
        # remove only symlinks/files we created
        for p in DST_ROOT.rglob("*"):
            if p.is_symlink() or p.is_file():
                p.unlink()
        # remove empty dirs
        for p in sorted(DST_ROOT.rglob("*"), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass

    # Write symlinks
    for img, msk in train_pairs:
        safe_symlink(img, DST_IMG_TRAIN / img.name)
        safe_symlink(msk, DST_MSK_TRAIN / msk.name)

    for img, msk in val_pairs:
        safe_symlink(img, DST_IMG_VAL / img.name)
        safe_symlink(msk, DST_MSK_VAL / msk.name)

    print("Done.")
    print(f"Missing masks (skipped): {missing}")
    print(f"Total paired samples: {n_total}")
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")
    print(f"Output: {DST_ROOT.resolve()}")

if __name__ == "__main__":
    main()

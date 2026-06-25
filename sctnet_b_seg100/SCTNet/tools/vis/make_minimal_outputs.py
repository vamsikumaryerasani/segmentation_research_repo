import os, glob, shutil
import cv2
import numpy as np

# ---- EDIT THESE 3 PATHS IF NEEDED ----
SRC = "work_dirs/SCTNet-B-Seg100_railsem90_10_400e_1gpu/vis_final"      # where your current outputs are
INP = "work_dirs/SCTNet-B-Seg100_railsem90_10_400e_1gpu/vis_input_10"     # original images you ran inference on
OUT = "work_dirs/SCTNet-B-Seg100_railsem90_10_400e_1gpu/vis_minimal_only" # new clean folder
# --------------------------------------

NUM_CLASSES = 19
MAX_ID = NUM_CLASSES - 1
exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp")

def pick_one(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None

def find_original(prefix):
    token = prefix.strip("_").split("_")[-1]  # e.g. rs06678
    for e in exts:
        hits = glob.glob(os.path.join(INP, f"*{token}*{e}"))
        if hits:
            return hits[0]
    return None

# collect overlay_pred files in a VERY forgiving way
overlay_files = []
overlay_files += glob.glob(os.path.join(SRC, "*overlay*pred*.png"))
overlay_files += glob.glob(os.path.join(SRC, "*overlay*pred*.jpg"))
overlay_files += glob.glob(os.path.join(SRC, "*overlay*pred*.jpeg"))
overlay_files = sorted(set(overlay_files))

if not overlay_files:
    print(f"[ERROR] No overlay_pred files found in: {SRC}")
    print("[HINT] Run: ls -lah", SRC)
    print("[HINT] Or search: find work_dirs/... -iname '*overlay*pred*'")
    raise SystemExit(1)

# recreate OUT cleanly
if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT, exist_ok=True)

for ov in overlay_files:
    base = os.path.basename(ov)

    # derive prefix from filename:
    # handles: 09_rs06678_overlay_pred.png  OR  09_rs06678_pred_overlay.png etc
    prefix = base
    for cut in ["overlay_pred", "pred_overlay", "overlaypred", "predoverlay"]:
        prefix = prefix.replace(cut, "")
    prefix = prefix.replace(".png","").replace(".jpg","").replace(".jpeg","")
    if not prefix.endswith("_"):
        prefix = prefix + "_"

    # 1) original
    orig = find_original(prefix)
    if orig:
        shutil.copy2(orig, os.path.join(OUT, f"{prefix}orig{os.path.splitext(orig)[1]}"))
    else:
        print(f"[WARN] original not found for {prefix} inside {INP}")

    # 2) colored overlay pred
    shutil.copy2(ov, os.path.join(OUT, f"{prefix}overlay_pred{os.path.splitext(ov)[1]}"))

    # 3) grayscale VIS (0..255) (preferred) else create from RAW ids
    vis = pick_one([
        os.path.join(SRC, f"{prefix}pred_gray_VIS_0_255.png"),
        os.path.join(SRC, f"{prefix}*_pred_gray_VIS_0_255.png"),
    ])
    raw = pick_one([
        os.path.join(SRC, f"{prefix}pred_gray_RAW_ids.png"),
        os.path.join(SRC, f"{prefix}*_pred_gray_RAW_ids.png"),
    ])

    out_vis = os.path.join(OUT, f"{prefix}pred_gray_VIS_0_255.png")

    if vis and os.path.exists(vis):
        shutil.copy2(vis, out_vis)
    elif raw and os.path.exists(raw):
        m = cv2.imread(raw, cv2.IMREAD_UNCHANGED)
        if m is None:
            print(f"[WARN] could not read RAW mask: {raw}")
            continue
        if m.ndim == 3:
            m = m[:,:,0]
        m = m.astype(np.float32)
        m_vis = (m * (255.0 / MAX_ID)).clip(0,255).astype(np.uint8)
        m_vis_bgr = cv2.cvtColor(m_vis, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(out_vis, m_vis_bgr)
    else:
        print(f"[WARN] no gray VIS/RAW found for {prefix}")

print("✅ Done.")
print("SRC:", SRC)
print("OUT:", OUT)
print("Files in OUT:", len(glob.glob(os.path.join(OUT, '*'))))

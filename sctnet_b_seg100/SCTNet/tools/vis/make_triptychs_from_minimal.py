#!/usr/bin/env python3
import os, glob, argparse
import cv2
import numpy as np

# RailSem19 palette (RGB) -> BGR for cv2
PALETTE_RGB = [
    [128, 64,128],[244, 35,232],[70, 70, 70],[102,102,156],[190,153,153],
    [153,153,153],[250,170, 30],[220,220,  0],[107,142, 35],[152,251,152],
    [70,130,180],[220, 20, 60],[255,  0,  0],[  0,  0,142],[  0,  0, 70],
    [  0, 60,100],[  0, 80,100],[  0,  0,230],[119, 11, 32],
]
PALETTE_BGR = np.array(PALETTE_RGB, dtype=np.uint8)[:, ::-1]
IMG_EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp")

# Folder-name heuristics (robust)
ORIG_DIR_KEYS = {"orig","original","image","images","input"}
PRED_GRAY_DIR_KEYS = {"pred_gray","predgrey","gray_pred","predmask_gray","pred_mask_gray","pred_mask","pred"}
GT_GRAY_DIR_KEYS = {"gt_gray","gtgrey","gray_gt","gtmask_gray","gt_mask_gray","gt"}
OVERLAY_PRED_DIR_KEYS = {"overlay_pred","pred_overlay","overlay","overlaycolor","overlay_color"}
GT_COLOR_DIR_KEYS = {"gt_color","gtcolour","color_gt","gtmask_color","gt_mask_color"}

# Filename suffix heuristics
SUF_ORIG = ("_orig", "_original", "_img", "_image")
SUF_PRED_GRAY = ("_pred_gray", "_pred_grey", "_gray_pred", "_predmask_gray", "_pred_mask_gray", "_predmask", "_pred_mask", "_pred")
SUF_GT_GRAY = ("_gt_gray", "_gt_grey", "_gray_gt", "_gtmask_gray", "_gt_mask_gray", "_gtmask", "_gt_mask", "_gt")
SUF_OVERLAY_PRED = ("_overlay_pred", "_pred_overlay", "_overlay", "_overlaycolor", "_overlay_color")
SUF_GT_COLOR = ("_gt_color", "_gt_colour", "_color_gt", "_gtmask_color", "_gt_mask_color")

def to_bgr(im):
    if im is None:
        return None
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        return cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    return im

def read_img(p):
    return cv2.imread(p, cv2.IMREAD_UNCHANGED)

def strip_known_suffixes(stem: str) -> str:
    for suf_list in [SUF_ORIG, SUF_PRED_GRAY, SUF_GT_GRAY, SUF_OVERLAY_PRED, SUF_GT_COLOR]:
        for suf in suf_list:
            if stem.endswith(suf):
                return stem[: -len(suf)]
    return stem

def key_from_path(p):
    stem = os.path.splitext(os.path.basename(p))[0]
    return strip_known_suffixes(stem)

def is_id_mask(im):
    return im is not None and im.ndim == 2 and im.size > 0 and int(im.max()) <= 18

def gray_vis_from_ids(ids_u8, max_id=18):
    ids = ids_u8.astype(np.float32)
    vis = np.round(ids * (255.0 / float(max_id))).clip(0, 255).astype(np.uint8)
    return vis

def colorize_ids(ids_u8):
    h, w = ids_u8.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k in range(len(PALETTE_BGR)):
        out[ids_u8 == k] = PALETTE_BGR[k]
    return out

def overlay(img_bgr, color_bgr, alpha=0.6):
    return cv2.addWeighted(img_bgr, alpha, color_bgr, 1.0-alpha, 0)

def resize_like(im, H, W, interp):
    if im.shape[0] == H and im.shape[1] == W:
        return im
    return cv2.resize(im, (W, H), interpolation=interp)

def add_header(tri_bgr):
    titles=("Original images","Predicted outputs","ground-truth masks")
    h, w = tri_bgr.shape[:2]
    header_h = max(70, int(0.10 * h))
    canvas = np.full((h + header_h, w, 3), 255, dtype=np.uint8)
    canvas[header_h:] = tri_bgr

    col_w = w // 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.1
    thick = 2
    for i, t in enumerate(titles):
        x0 = i * col_w
        x1 = (i + 1) * col_w if i < 2 else w
        tw, th = cv2.getTextSize(t, font, scale, thick)[0]
        cx = (x0 + x1)//2 - tw//2
        cy = header_h//2 + th//2
        cv2.putText(canvas, t, (cx, cy), font, scale, (0,0,0), thick, cv2.LINE_AA)
    return canvas

def classify_file(path):
    # Prefer folder name classification
    parent = os.path.basename(os.path.dirname(path)).lower()
    if parent in ORIG_DIR_KEYS: return "orig"
    if parent in PRED_GRAY_DIR_KEYS: return "pred_gray"
    if parent in GT_GRAY_DIR_KEYS: return "gt_gray"
    if parent in OVERLAY_PRED_DIR_KEYS: return "overlay_pred"
    if parent in GT_COLOR_DIR_KEYS: return "gt_color"

    # Fallback: filename suffix classification
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    for suf in SUF_ORIG:
        if stem.endswith(suf): return "orig"
    for suf in SUF_PRED_GRAY:
        if stem.endswith(suf): return "pred_gray"
    for suf in SUF_GT_GRAY:
        if stem.endswith(suf): return "gt_gray"
    for suf in SUF_OVERLAY_PRED:
        if stem.endswith(suf): return "overlay_pred"
    for suf in SUF_GT_COLOR:
        if stem.endswith(suf): return "gt_color"

    return None

def collect_maps(in_dir):
    all_files = []
    for ext in IMG_EXTS:
        all_files += glob.glob(os.path.join(in_dir, "**", f"*{ext}"), recursive=True)

    maps = {k: {} for k in ["orig","pred_gray","gt_gray","overlay_pred","gt_color"]}
    unknown = []

    for p in all_files:
        cls = classify_file(p)
        if cls is None:
            unknown.append(p)
            continue
        maps[cls][key_from_path(p)] = p

    return maps, unknown

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--opacity", type=float, default=0.6)
    args = ap.parse_args()

    IN = args.in_dir
    OUT = args.out_dir
    os.makedirs(OUT, exist_ok=True)

    out_black = os.path.join(OUT, "triptych_black")
    out_color = os.path.join(OUT, "triptych_color")
    os.makedirs(out_black, exist_ok=True)
    os.makedirs(out_color, exist_ok=True)

    maps, unknown = collect_maps(IN)
    orig_map = maps["orig"]
    pred_gray_map = maps["pred_gray"]
    gt_gray_map = maps["gt_gray"]
    overlay_pred_map = maps["overlay_pred"]
    gt_color_map = maps["gt_color"]

    print("[INFO] bucket sizes:",
          "orig", len(orig_map),
          "pred_gray", len(pred_gray_map),
          "gt_gray", len(gt_gray_map),
          "overlay_pred", len(overlay_pred_map),
          "gt_color", len(gt_color_map))
    if unknown:
        print(f"[INFO] Unclassified files (showing up to 10):")
        for p in unknown[:10]:
            print("  -", p)

    # BLACK needs: orig + pred_gray + gt_gray
    keys_black = sorted(set(orig_map) & set(pred_gray_map) & set(gt_gray_map))[:args.max_images]

    # COLOR needs: orig + (overlay_pred OR pred_gray) + (gt_color OR gt_gray)
    keys_color = sorted(
        set(orig_map)
        & (set(overlay_pred_map) | set(pred_gray_map))
        & (set(gt_color_map) | set(gt_gray_map))
    )[:args.max_images]

    made_b = 0
    made_c = 0

    # ---------- BLACK triptychs ----------
    for k in keys_black:
        orig = to_bgr(read_img(orig_map[k]))
        pred = read_img(pred_gray_map[k])
        gt   = read_img(gt_gray_map[k])
        if orig is None or pred is None or gt is None:
            continue
        H, W = orig.shape[:2]

        if is_id_mask(pred):
            pred = gray_vis_from_ids(pred.astype(np.uint8))
        if is_id_mask(gt):
            gt = gray_vis_from_ids(gt.astype(np.uint8))

        pred_bgr = to_bgr(pred)
        gt_bgr   = to_bgr(gt)
        pred_bgr = resize_like(pred_bgr, H, W, cv2.INTER_NEAREST)
        gt_bgr   = resize_like(gt_bgr, H, W, cv2.INTER_NEAREST)

        tri = np.concatenate([orig, pred_bgr, gt_bgr], axis=1)
        tri = add_header(tri)
        cv2.imwrite(os.path.join(out_black, f"{k}_triptych_black.png"), tri)
        made_b += 1

    # ---------- COLOR triptychs ----------
    for k in keys_color:
        orig = to_bgr(read_img(orig_map[k]))
        if orig is None:
            continue
        H, W = orig.shape[:2]

        # Prediction output: prefer overlay_pred; else build overlay from pred_gray if it's raw IDs
        if k in overlay_pred_map:
            pred_out = to_bgr(read_img(overlay_pred_map[k]))
        else:
            pred_raw = read_img(pred_gray_map.get(k, ""))
            if pred_raw is None:
                continue
            if is_id_mask(pred_raw):
                col = colorize_ids(pred_raw.astype(np.uint8))
                col = resize_like(col, H, W, cv2.INTER_NEAREST)
                pred_out = overlay(orig, col, alpha=args.opacity)
            else:
                pred_out = to_bgr(pred_raw)

        # GT output: prefer gt_color; else colorize gt_gray if raw IDs
        if k in gt_color_map:
            gt_out = to_bgr(read_img(gt_color_map[k]))
        else:
            gt_raw = read_img(gt_gray_map.get(k, ""))
            if gt_raw is None:
                continue
            if is_id_mask(gt_raw):
                gt_out = colorize_ids(gt_raw.astype(np.uint8))
            else:
                gt_out = to_bgr(gt_raw)

        pred_out = resize_like(pred_out, H, W, cv2.INTER_NEAREST)
        gt_out   = resize_like(gt_out,   H, W, cv2.INTER_NEAREST)

        tri = np.concatenate([orig, pred_out, gt_out], axis=1)
        tri = add_header(tri)
        cv2.imwrite(os.path.join(out_color, f"{k}_triptych_color.png"), tri)
        made_c += 1

    print(f"[OK] made {made_b} black triptychs in: {out_black}")
    print(f"[OK] made {made_c} color triptychs in: {out_color}")

    if made_b == 0 and made_c == 0:
        print("[HINT] No matched triplets were found.")
        print("       This usually means your files are not named/classified as orig/pred/gt,")
        print("       or the folder is empty / wrong path.")
        print("       Run:")
        print(f"         find {IN} -maxdepth 3 -type f | head -n 50")

if __name__ == "__main__":
    main()

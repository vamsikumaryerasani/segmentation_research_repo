#!/usr/bin/env python3
import os, glob, argparse
import cv2
import numpy as np

def imread_any(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img

def to_bgr(img):
    # Convert to 3-channel BGR for stacking
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def resize_like(img, ref_h, ref_w, interp=cv2.INTER_NEAREST):
    if img.shape[0] == ref_h and img.shape[1] == ref_w:
        return img
    return cv2.resize(img, (ref_w, ref_h), interpolation=interp)

def add_header(triptych_bgr, titles=("Original images","Predicted outputs","ground-truth masks")):
    h, w = triptych_bgr.shape[:2]
    col_w = w // 3
    header_h = max(60, int(0.10 * h))
    header = np.full((header_h, w, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    y = int(header_h * 0.70)

    for i, t in enumerate(titles):
        x0 = i * col_w
        x1 = (i + 1) * col_w
        (tw, th), _ = cv2.getTextSize(t, font, scale, thickness)
        cx = x0 + (x1 - x0 - tw) // 2
        cv2.putText(header, t, (max(5, cx), y), font, scale, (0,0,0), thickness, cv2.LINE_AA)

    return np.vstack([header, triptych_bgr])

def make_triptych(orig_path, pred_path, gt_path, add_titles=True):
    orig = to_bgr(imread_any(orig_path))
    pred = to_bgr(imread_any(pred_path))
    gt   = to_bgr(imread_any(gt_path))

    H, W = orig.shape[:2]
    pred = resize_like(pred, H, W, interp=cv2.INTER_NEAREST)
    gt   = resize_like(gt,   H, W, interp=cv2.INTER_NEAREST)

    trip = np.concatenate([orig, pred, gt], axis=1)
    if add_titles:
        trip = add_header(trip)
    return trip

def find_orig(input_dir, token):
    # token example: 09_rs06678
    # try rs06678 first, then token
    keys = [token.split("_", 1)[-1], token]
    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp")
    for k in keys:
        for e in exts:
            hits = glob.glob(os.path.join(input_dir, f"*{k}*{e}"))
            if hits:
                return hits[0]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis-final", required=True, help="folder containing *_pred_COLOR.png etc")
    ap.add_argument("--input-dir", required=True, help="folder containing the original RGB images")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-images", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_color = os.path.join(args.out_dir, "panels_color")
    out_gray  = os.path.join(args.out_dir, "panels_gray")
    os.makedirs(out_color, exist_ok=True)
    os.makedirs(out_gray, exist_ok=True)

    # tokens from overlay_pred (most reliable index)
    overlays = sorted(glob.glob(os.path.join(args.vis_final, "*_overlay_pred.png")))
    if not overlays:
        raise SystemExit(f"[ERROR] No *_overlay_pred.png found in {args.vis_final}")

    overlays = overlays[:args.max_images]

    color_panels = []
    gray_panels  = []

    for ov in overlays:
        token = os.path.basename(ov).split("_overlay_pred")[0]

        orig = find_orig(args.input_dir, token)
        if orig is None:
            print("[WARN] original not found for", token, "skipping")
            continue

        pred_color = os.path.join(args.vis_final, f"{token}_pred_COLOR.png")
        gt_color   = os.path.join(args.vis_final, f"{token}_gt_COLOR.png")

        # grayscale (RAW ids)
        pred_gray  = os.path.join(args.vis_final, f"{token}_pred_gray_RAW_ids.png")
        gt_gray    = os.path.join(args.vis_final, f"{token}_gt_gray_RAW_ids.png")

        # build color triptych
        if os.path.exists(pred_color) and os.path.exists(gt_color):
            trip_color = make_triptych(orig, pred_color, gt_color, add_titles=True)
            outp = os.path.join(out_color, f"{token}_PANEL_color.png")
            cv2.imwrite(outp, trip_color)
            color_panels.append(trip_color)
        else:
            print("[WARN] missing pred/gt COLOR for", token)

        # build gray triptych (black-ish annotated ids)
        if os.path.exists(pred_gray) and os.path.exists(gt_gray):
            trip_gray = make_triptych(orig, pred_gray, gt_gray, add_titles=True)
            outp = os.path.join(out_gray, f"{token}_PANEL_gray.png")
            cv2.imwrite(outp, trip_gray)
            gray_panels.append(trip_gray)
        else:
            print("[WARN] missing pred/gt gray RAW ids for", token)

    # also save ONE combined grid image for each type (like your example)
    def save_grid(panels, out_path):
        if not panels:
            return
        # panels already include titles; for grid we keep titles only on first panel
        panels2 = [panels[0]] + [p[int(p.shape[0]*0.10):,:,:] for p in panels[1:]]  # remove header from subsequent
        gap = 10
        w = max(p.shape[1] for p in panels2)
        rows = []
        for p in panels2:
            if p.shape[1] < w:
                pad = np.full((p.shape[0], w - p.shape[1], 3), 255, dtype=np.uint8)
                p = np.hstack([p, pad])
            rows.append(p)
        grid = rows[0]
        for r in rows[1:]:
            spacer = np.full((gap, w, 3), 255, dtype=np.uint8)
            grid = np.vstack([grid, spacer, r])
        cv2.imwrite(out_path, grid)

    save_grid(color_panels, os.path.join(args.out_dir, "GRID_color.png"))
    save_grid(gray_panels,  os.path.join(args.out_dir, "GRID_gray.png"))

    print("✅ DONE")
    print("Color panels:", out_color)
    print("Gray panels :", out_gray)
    print("Grid color  :", os.path.join(args.out_dir, "GRID_color.png"))
    print("Grid gray   :", os.path.join(args.out_dir, "GRID_gray.png"))

if __name__ == "__main__":
    main()

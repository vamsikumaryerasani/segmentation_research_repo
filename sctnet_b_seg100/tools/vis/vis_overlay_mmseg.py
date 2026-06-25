#!/usr/bin/env python3
import argparse
import os
import glob
from mmseg.apis import init_segmentor, inference_segmentor

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

def list_images(in_dir, pattern):
    paths = sorted(glob.glob(os.path.join(in_dir, pattern), recursive=True))
    paths = [p for p in paths if os.path.isfile(p) and p.lower().endswith(IMG_EXTS)]
    return paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config .py")
    parser.add_argument("checkpoint", help="path to .pth")
    parser.add_argument("--in-dir", required=True, help="folder with images")
    parser.add_argument("--glob", default="**/*", help='glob pattern (default: "**/*")')
    parser.add_argument("--out-dir", required=True, help="output folder")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opacity", type=float, default=0.6)
    parser.add_argument("--max-images", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    imgs = list_images(args.in_dir, args.glob)
    if not imgs:
        raise RuntimeError(f"No images found in {args.in_dir} with pattern {args.glob}")

    imgs = imgs[: args.max_images]
    print(f"[INFO] Found {len(imgs)} images (showing first {len(imgs)}).")

    # Load model
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    # Run inference + save overlay
    for i, img_path in enumerate(imgs, 1):
        result = inference_segmentor(model, img_path)

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_file = os.path.join(args.out_dir, f"{base}_pred.png")

        # Overlay visualization
        model.show_result(
            img_path,
            result,
            out_file=out_file,
            opacity=args.opacity,
            show=False
        )

        print(f"[{i}/{len(imgs)}] saved: {out_file}")

    print("[DONE] All visualizations saved.")

if __name__ == "__main__":
    main()

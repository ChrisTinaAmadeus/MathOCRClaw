import os
import sys
import argparse
from pathlib import Path
from typing import List

from PIL import Image
from paddleocr import LayoutDetection


def collect_images(image_path: str = None, image_dir: str = None) -> List[Path]:
    if image_path:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        return [p]

    if image_dir:
        root = Path(image_dir)
        if not root.exists():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        imgs = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in exts]
        if not imgs:
            raise RuntimeError(f"No images were found under directory {image_dir}")
        return imgs

    raise ValueError("You must specify either --image-path or --image-dir")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        "Batch inference with PP-DocLayout_plus-L (LayoutDetection) and save JSON + visualizations"
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image-path", type=str)
    g.add_argument("--image-dir", type=str)

    ap.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )
    ap.add_argument(
        "--model-name",
        type=str,
        default="PP-DocLayout_plus-L",
    )
    ap.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("MTC_DOCLAYOUT_MODEL_DIR", ""),
        help="Optional local PaddleX model directory. If set, it is used instead of downloading by model-name.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    ap.add_argument(
        "--no-nms",
        action="store_true",
    )
    ap.add_argument(
        "--enable-mkldnn",
        action="store_true",
        help="Enable MKLDNN/oneDNN CPU acceleration. Disabled by default for Windows stability.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    out_root = Path(args.output_dir)
    vis_dir = out_root / "vis_img"
    json_dir = out_root / "json"
    vis_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    try:
        images = collect_images(args.image_path, args.image_dir)
    except Exception as e:
        print(f"[ERROR] Failed to collect images: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(images)} image(s) in total")
    print(f"[INFO] Model: {args.model_name}, Device: {args.device}")
    if args.model_dir:
        print(f"[INFO] Model dir: {args.model_dir}")
    print(f"[INFO] Output directory: {out_root} (vis_img/, json/)")

    model_kwargs = {"device": args.device, "enable_mkldnn": bool(args.enable_mkldnn)}
    if args.model_dir:
        model_kwargs["model_dir"] = args.model_dir
    else:
        model_kwargs["model_name"] = args.model_name
    model = LayoutDetection(**model_kwargs)
    img_paths_str = [str(p) for p in images]
    try:
        outs = model.predict(
            img_paths_str,
            batch_size=args.batch_size,
            layout_nms=(not args.no_nms),
        )
    except Exception as e:
        print(f"[ERROR] LayoutDetection.predict failed: {e}", file=sys.stderr)
        sys.exit(1)

    for img_path, res in zip(images, outs):
        stem = img_path.stem
        print(f"[DocLayout] Processing completed: {img_path}")

        res.save_to_img(save_path=str(vis_dir))

        json_save_dir = str(json_dir)
        res.save_to_json(save_path=json_save_dir)

        print(
            f"  -> vis:  {vis_dir}/{stem}.png or .jpg (determined by Paddle)\n"
            f"  -> json: {json_dir}/{stem}.json"
        )


if __name__ == "__main__":
    main()

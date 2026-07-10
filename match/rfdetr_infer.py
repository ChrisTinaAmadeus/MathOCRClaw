import os
import sys
import json
import math
import argparse
import shutil
import warnings
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image

import time
os.environ.setdefault("LOG_LEVEL", "ERROR")
warnings.filterwarnings("ignore", message=r".*target=True is deprecated.*")
logging.getLogger("rf-detr").setLevel(logging.ERROR)
logging.getLogger("rfdetr").setLevel(logging.ERROR)
try:
    with open('/home/tangcheng/an/.cursor/debug.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps({
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "A",
            "location": "match/rfdetr_infer.py:15",
            "message": "Before supervision import - checking Python env",
            "data": {
                "python_executable": sys.executable,
                "python_version": sys.version,
                "sys_path": sys.path[:5]  # First 5 entries
            },
            "timestamp": int(time.time() * 1000)
        }, ensure_ascii=False) + "\n")
except Exception:
    pass


try:
    import supervision as sv

    try:
        with open('/home/tangcheng/an/.cursor/debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "A",
                "location": "match/rfdetr_infer.py:15",
                "message": "supervision import SUCCESS",
                "data": {"supervision_version": getattr(sv, '__version__', 'unknown')},
                "timestamp": int(time.time() * 1000)
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass

except ImportError as e:
    try:
        import subprocess
        import shutil
        with open('/home/tangcheng/an/.cursor/debug.log', 'a', encoding='utf-8') as f:
            pip_check = shutil.which('pip')
            pip3_check = shutil.which('pip3')
            python_check = sys.executable
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "A,B,C,D,E",
                "location": "match/rfdetr_infer.py:15",
                "message": "supervision import FAILED",
                "data": {
                    "error": str(e),
                    "python_executable": python_check,
                    "pip_path": pip_check,
                    "pip3_path": pip3_check,
                    "sys_path": sys.path
                },
                "timestamp": int(time.time() * 1000)
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass
    raise

from rfdetr import RFDETRMedium


DEFAULT_CLASS_NAMES: Dict[int, str] = {
    2: "problem_solving_question",
    3: "partial_question",
}


def load_class_names(classes_json: Optional[str]) -> Optional[Dict[int, str]]:
    if classes_json is None:
        return None
    if not os.path.exists(classes_json):
        print(f"[WARN] classes_json file does not exist: {classes_json}. Only class_id will be used.", file=sys.stderr)
        return None

    with open(classes_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    mapping: Dict[int, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                idx = int(k)
                mapping[idx] = str(v)
            except ValueError:
                continue
    elif isinstance(raw, list):
        for idx, name in enumerate(raw):
            mapping[idx] = str(name)
    else:
        print("[WARN] Invalid classes_json format. Ignoring it.", file=sys.stderr)
        return None

    return mapping


def collect_images(image_path: Optional[str], image_dir: Optional[str]) -> List[Path]:
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
        images = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in exts]
        if not images:
            raise RuntimeError(f"No images were found under directory: {image_dir}.")
        return images

    raise ValueError("You must specify either --image-path or --image-dir.")


def expand_and_clip(xyxy, w: int, h: int, pad_ratio: float):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_ratio, bh * pad_ratio

    nx1 = max(0, int(math.floor(x1 - px)))
    ny1 = max(0, int(math.floor(y1 - py)))
    nx2 = min(w, int(math.ceil(x2 + px)))
    ny2 = min(h, int(math.ceil(y2 + py)))

    if nx2 <= nx1:
        nx2 = min(w, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(h, ny1 + 1)
    return nx1, ny1, nx2, ny2


def get_class_name(
    cid: int,
    class_name_map: Optional[Dict[int, str]],
    rf_model: RFDETRMedium,
) -> str:
    if class_name_map is not None and cid in class_name_map:
        return class_name_map[cid]

    if hasattr(rf_model, "class_names") and isinstance(rf_model.class_names, dict):
        # Compatible with both 1-based and 0-based indexing
        name = rf_model.class_names.get(int(cid) + 1)
        if name is None:
            name = rf_model.class_names.get(int(cid))
        if name is not None:
            return str(name)

    if cid in DEFAULT_CLASS_NAMES:
        return DEFAULT_CLASS_NAMES[cid]

    return str(cid)


def run_inference(
    checkpoint: str,
    image_paths: List[Path],
    output_dir: str,
    classes_json: Optional[str] = None,
    threshold: float = 0.3,
    pad_ratio: float = 0.02,
    min_area: float = 0.0,
    overwrite_jsonl: bool = False,
    clean_output: bool = False,
    num_classes: Optional[int] = 3,
    optimize_for_inference: bool = False,
) -> None:
    output_root = Path(output_dir)
    overlay_dir = output_root / "overlay"
    crops_dir = output_root / "crops"
    output_root.mkdir(parents=True, exist_ok=True)

    if clean_output:
        if overlay_dir.exists():
            shutil.rmtree(overlay_dir)
        if crops_dir.exists():
            shutil.rmtree(crops_dir)

    overlay_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    class_name_map = load_class_names(classes_json)

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    checkpoint = str(checkpoint_path)

    print(f"[INFO] Using checkpoint: {checkpoint}")
    print(f"[INFO] Output directory: {output_root}")
    print(f"[INFO] Confidence threshold: {threshold}, pad_ratio: {pad_ratio}, min_area: {min_area}")
    print(f"[INFO] overwrite_jsonl: {overwrite_jsonl}, clean_output: {clean_output}")
    print(f"[INFO] num_classes: {num_classes}, optimize_for_inference: {optimize_for_inference}")

    model_kwargs: Dict[str, Any] = {"pretrain_weights": checkpoint}
    if num_classes is not None and int(num_classes) > 0:
        model_kwargs["num_classes"] = int(num_classes)
    try:
        rf_model = RFDETRMedium(**model_kwargs)
    except TypeError:
        model_kwargs.pop("num_classes", None)
        rf_model = RFDETRMedium(**model_kwargs)

    if hasattr(rf_model, "eval") and callable(getattr(rf_model, "eval")):
        rf_model.eval()

    if optimize_for_inference and hasattr(rf_model, "optimize_for_inference") and callable(
        getattr(rf_model, "optimize_for_inference")
    ):
        try:
            rf_model.optimize_for_inference()
        except Exception as e:
            print(f"[WARN] optimize_for_inference failed (can be ignored): {e}", file=sys.stderr)

    jsonl_path = output_root / "rfdetr_infer_results.jsonl"

    if overwrite_jsonl or (not jsonl_path.exists()):
        json_mode = "w"
    else:
        json_mode = "a"

    print(f"[INFO] JSONL path: {jsonl_path} (mode={json_mode})")

    try:
        from tqdm import tqdm
        iterator = tqdm(image_paths, desc="Running RF-DETR inference")
    except Exception:
        iterator = image_paths

    with jsonl_path.open(json_mode, encoding="utf-8") as f_jsonl:
        for img_path in iterator:
            img_path = Path(img_path)
            stem = img_path.stem

            try:
                with Image.open(img_path) as im:
                    image = im.convert("RGB")
                    width, height = image.size

                    raw_out = rf_model.predict([image], threshold=threshold)
            except Exception as e:
                print(f"[ERROR] Failed to open or run inference on image: {img_path}, error: {e}", file=sys.stderr)
                continue

            detections = raw_out[0] if isinstance(raw_out, list) else raw_out

            if min_area > 0:
                keep_idx = []
                for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
                    if (x2 - x1) * (y2 - y1) >= min_area:
                        keep_idx.append(i)
                if len(keep_idx) != len(detections):
                    removed = len(detections) - len(keep_idx)
                    print(
                        f"[INFO] {img_path.name}: filtered out {removed} small boxes (min_area={min_area})",
                        file=sys.stderr,
                    )
                    detections = detections[keep_idx]

            canvas = np.array(image)
            box_annotator = sv.BoxAnnotator(thickness=2)
            label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
            labels = []
            for idx, (cid, conf) in enumerate(zip(detections.class_id, detections.confidence), start=1):
                cid_int = int(cid)
                cname = get_class_name(cid_int, class_name_map, rf_model)
                labels.append(f"#{idx} {cname} {conf:.2f}")

            canvas = box_annotator.annotate(scene=canvas, detections=detections)
            canvas = label_annotator.annotate(scene=canvas, detections=detections, labels=labels)

            overlay_path = overlay_dir / f"{stem}_overlay.jpg"
            Image.fromarray(canvas).save(overlay_path)
            img_crop_dir = crops_dir / stem
            img_crop_dir.mkdir(parents=True, exist_ok=True)

            det_list: List[Dict[str, Any]] = []
            for idx, (xyxy, cid, conf) in enumerate(
                zip(detections.xyxy, detections.class_id, detections.confidence), start=1
            ):
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                cname = get_class_name(int(cid), class_name_map, rf_model)
                score = float(conf)

                px1, py1, px2, py2 = expand_and_clip(xyxy, width, height, pad_ratio)
                crop = image.crop((px1, py1, px2, py2))

                crop_name = f"{stem}_det{idx:02d}_cls{int(cid)}_{cname}_s{score:.2f}.png"
                crop_name = crop_name.replace(" ", "_")
                crop_path = img_crop_dir / crop_name
                crop.save(crop_path)

                det_list.append(
                    {
                        "index": idx,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "bbox_xyxy_padded": [px1, py1, px2, py2],
                        "score": score,
                        "class_id": int(cid),
                        "class_name": cname,
                        "crop_path": str(crop_path),
                    }
                )

            record: Dict[str, Any] = {
                "image_path": str(img_path),
                "file_name": img_path.name,
                "width": width,
                "height": height,
                "overlay_path": str(overlay_path),
                "detections": det_list,
            }

            f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] Inference and visualization completed, total {len(image_paths)} image(s)")
    print(f"[OK] Detection results JSONL: {jsonl_path}")
    print(f"[OK] Overlay directory: {overlay_dir}")
    print(f"[OK] Crops directory: {crops_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RF-DETR inference: save JSONL + draw overlay boxes + crop bounding boxes"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image-path", type=str, help="Path to a single image")
    group.add_argument("--image-dir", type=str, help="Directory containing multiple images (recursively traversed)")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the RF-DETR checkpoint")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory; overlay/, crops/, and rfdetr_infer_results.jsonl will be created inside it",
    )
    parser.add_argument(
        "--classes-json",
        type=str,
        default=None,
        help="Optional: class mapping JSON file",
    )
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument(
        "--pad-ratio",
        type=float,
        default=0.02,
        help="Relative padding ratio when cropping bounding boxes (default: 0.02)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="Minimum area for filtering small boxes (pixel^2, default: 0 means no filtering)",
    )
    parser.add_argument(
        "--overwrite-jsonl",
        action="store_true",
        help="If set, overwrite rfdetr_infer_results.jsonl; otherwise append to the existing file",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="If set, clear the overlay/ and crops/ directories before running",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Number of classes for RF-DETR model (default: 3; use 0 to let RF-DETR infer it)",
    )
    parser.add_argument(
        "--optimize-for-inference",
        action="store_true",
        help="Run RF-DETR tracing optimization. Useful for large batches, slow/noisy for quick single-image runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        imgs = collect_images(args.image_path, args.image_dir)
    except Exception as e:
        print(f"[ERROR] Failed to collect images: {e}", file=sys.stderr)
        sys.exit(1)

    run_inference(
        checkpoint=args.checkpoint,
        image_paths=imgs,
        output_dir=args.output_dir,
        classes_json=args.classes_json,
        threshold=args.threshold,
        pad_ratio=args.pad_ratio,
        min_area=args.min_area,
        overwrite_jsonl=args.overwrite_jsonl,
        clean_output=args.clean_output,
        num_classes=args.num_classes,
        optimize_for_inference=args.optimize_for_inference,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .common import sha1_bytes

def safe_open_image(path) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None

def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def img_to_data_url(img: Image.Image, *, jpeg_quality: int = 85) -> Tuple[str, str]:

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_clamp_int(jpeg_quality, 40, 95), optimize=True)
    b = buf.getvalue()
    h = sha1_bytes(b)
    data_url = "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")
    return data_url, h

def blank_frac(img: Image.Image, *, white_thr: int = 248) -> float:

    g = img.convert("L")
    hist = g.histogram()
    total = sum(hist) or 1
    white = sum(hist[white_thr:])  # >= thr
    return float(white) / float(total)


def scan_document_for_ocr(img: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    """Normalize page illumination while preserving all ink colors.

    Background division removes slow illumination changes while retaining both
    printed content and colored annotations for downstream OCR/VLM calls.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]

    blur_sigma = max(8.0, min(50.0, min(width, height) * 0.02))
    background = cv2.GaussianBlur(
        rgb,
        (0, 0),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma,
        borderType=cv2.BORDER_REPLICATE,
    ).astype(np.float32)
    corrected_float = rgb.astype(np.float32) * 255.0 / np.maximum(background, 1.0)
    # Restore stroke contrast after flattening the background illumination.
    corrected_float = 255.0 - (255.0 - corrected_float) * 1.35
    corrected = np.clip(corrected_float, 0, 255).astype(np.uint8)

    metadata: Dict[str, Any] = {
        "method": "background_division",
        "width": width,
        "height": height,
        "shadow_blur_sigma": round(blur_sigma, 2),
        "ink_contrast_gain": 1.35,
        "colored_ink_preserved": True,
    }
    return Image.fromarray(corrected), metadata


def enhance_handwriting_ink(img: Image.Image, *, max_edge: int = 1600) -> Image.Image:
    """Create a grayscale companion view that exposes faint local pen strokes.

    This is intentionally a companion to the color context, not a replacement:
    color remains necessary for separating student writing from teacher marks.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    local = clahe.apply(gray)
    blurred = cv2.GaussianBlur(local, (0, 0), sigmaX=1.0, sigmaY=1.0)
    sharpened = cv2.addWeighted(local, 1.65, blurred, -0.65, 0)
    output = Image.fromarray(sharpened).convert("RGB")

    width, height = output.size
    longest = max(width, height)
    if 0 < longest < max_edge:
        scale = min(2.0, max_edge / float(longest))
        if scale > 1.05:
            output = output.resize(
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                Image.Resampling.LANCZOS,
            )
    elif longest > max_edge:
        scale = max_edge / float(longest)
        output = output.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.Resampling.LANCZOS,
        )
    return output

@dataclass
class EnhanceCfg:
    max_edge: int
    contrast: float
    sharpness: float
    autocontrast: bool = False

BASE_ENHANCE = EnhanceCfg(max_edge=900, contrast=1.25, sharpness=1.10, autocontrast=False)
STRONG_ENHANCE = EnhanceCfg(max_edge=1400, contrast=1.35, sharpness=1.25, autocontrast=True)

def enhance_for_vlm(img: Image.Image, cfg: EnhanceCfg) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"cannot enhance an empty image: {w}x{h}")
    m = max(w, h)
    if m > cfg.max_edge:
        scale = cfg.max_edge / float(m)
        # Extremely thin but valid crops can occur at noisy/overlapping layout
        # boundaries.  Pillow rejects a zero-sized dimension, so preserve at
        # least one pixel while the region selector supplies a better crop.
        resized_width = max(1, int(round(w * scale)))
        resized_height = max(1, int(round(h * scale)))
        img = img.resize(
            (resized_width, resized_height),
            Image.Resampling.BICUBIC,
        )

    if cfg.autocontrast:
        if blank_frac(img) < 0.92:
            img = ImageOps.autocontrast(img, cutoff=1)

    if cfg.contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(cfg.contrast)
    if cfg.sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(cfg.sharpness)
    return img

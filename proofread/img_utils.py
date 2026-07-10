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
    """Normalize page illumination and erase red ink before OCR/VLM calls.

    Background division removes slow illumination changes while preserving dark
    strokes. Red pixels are detected from both HSV hue and channel dominance;
    the mask is slightly expanded so JPEG-colored edges are removed as well.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    red_low = cv2.inRange(hsv, (0, 55, 40), (12, 255, 255))
    red_high = cv2.inRange(hsv, (168, 55, 40), (179, 255, 255))
    red_hue = (red_low > 0) | (red_high > 0)

    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    red_dominant = (red >= green + 18) & (red >= blue + 18)
    red_mask = (red_hue & red_dominant).astype(np.uint8) * 255
    red_mask = cv2.dilate(red_mask, np.ones((3, 3), np.uint8), iterations=1)

    cleaned = rgb.copy()
    cleaned[red_mask > 0] = 255

    blur_sigma = max(8.0, min(50.0, min(width, height) * 0.02))
    background = cv2.GaussianBlur(
        cleaned,
        (0, 0),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma,
        borderType=cv2.BORDER_REPLICATE,
    ).astype(np.float32)
    corrected_float = cleaned.astype(np.float32) * 255.0 / np.maximum(background, 1.0)
    # Restore stroke contrast after flattening the background illumination.
    corrected_float = 255.0 - (255.0 - corrected_float) * 1.35
    corrected = np.clip(corrected_float, 0, 255).astype(np.uint8)
    corrected[red_mask > 0] = 255

    red_pixels = int(np.count_nonzero(red_mask))
    total_pixels = max(1, width * height)
    metadata: Dict[str, Any] = {
        "method": "background_division_and_red_mask",
        "width": width,
        "height": height,
        "shadow_blur_sigma": round(blur_sigma, 2),
        "ink_contrast_gain": 1.35,
        "red_pixels_removed": red_pixels,
        "red_pixel_fraction": round(red_pixels / total_pixels, 8),
    }
    return Image.fromarray(corrected), metadata

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
    m = max(w, h)
    if m > cfg.max_edge:
        scale = cfg.max_edge / float(m)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    if cfg.autocontrast:
        if blank_frac(img) < 0.92:
            img = ImageOps.autocontrast(img, cutoff=1)

    if cfg.contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(cfg.contrast)
    if cfg.sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(cfg.sharpness)
    return img

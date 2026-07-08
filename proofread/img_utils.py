from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Optional, Tuple

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


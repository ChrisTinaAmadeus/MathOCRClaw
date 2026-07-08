from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


def _get_font(size: int = 18) -> ImageFont.ImageFont:
    for fp in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 2
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill="black")
    draw.text((x, y), text, font=font, fill=fill)


def _xyxy_to_int(b: Sequence[float]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in b]
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def _clamp_xyxy(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def draw_overlay(
    image: Image.Image,
    *,
    questions: List[Dict[str, Any]],
    figures: List[Dict[str, Any]],
    split_x: Optional[List[float]] = None,
    out_path: str,
    draw_edges: bool = False,
    edges: Optional[List[Tuple[int, int]]] = None,   # (det_index, fig_index)
    meta_lines: Optional[List[str]] = None,
    thickness: int = 3,
    font_size: int = 20,
    # NEW: split 容差带（和你 column 判定的 tol 对齐）
    boundary_tol_ratio: float = 0.02,
    draw_split_band: bool = True,
) -> str:
    im = image.copy()
    draw = ImageDraw.Draw(im)
    font = _get_font(font_size)

    w, h = im.size

    # meta_lines：用 label 形式画，保证可读性
    if meta_lines:
        y = 6
        for line in meta_lines:
            if not line:
                continue
            _draw_label(draw, (6, y), str(line), font, fill="yellow")
            y += font_size + 4

    # split_x：画线 + 可选容差带
    if split_x:
        tol = int(round(float(boundary_tol_ratio) * float(w)))
        for sx in split_x:
            x = int(round(float(sx)))
            if draw_split_band and tol > 0:
                # 画两条淡线模拟 band（PIL 纯 RGB 不好做半透明，这里用双线替代）
                draw.line([(x - tol, 0), (x - tol, h)], fill="cyan", width=1)
                draw.line([(x + tol, 0), (x + tol, h)], fill="cyan", width=1)
            draw.line([(x, 0), (x, h)], fill="cyan", width=max(1, thickness))

    fig_center: Dict[int, Tuple[float, float]] = {}
    for f in figures:
        bx = f.get("bbox_xyxy")
        if not bx:
            continue
        x1, y1, x2, y2 = _xyxy_to_int(bx)
        x1, y1, x2, y2 = _clamp_xyxy(x1, y1, x2, y2, w, h)

        fig_index = int(f.get("fig_index", -1))
        col_id = f.get("col_id", None)
        spanning = bool(f.get("is_spanning", False)) or (col_id == -1)

        # spanning 的框更醒目
        outline = "orange" if spanning else "red"
        width = thickness + 1 if spanning else thickness
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)

        label = f"F{fig_index}"
        if col_id is not None and col_id != -1:
            label += f" c{int(col_id)}"
        if spanning:
            label += " S"
        _draw_label(draw, (x1 + 2, max(0, y1 - font_size - 2)), label, font, fill=outline)

        fig_center[fig_index] = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    q_center: Dict[int, Tuple[float, float]] = {}
    for q in questions:
        bx = q.get("bbox_xyxy")
        if not bx:
            continue
        x1, y1, x2, y2 = _xyxy_to_int(bx)
        x1, y1, x2, y2 = _clamp_xyxy(x1, y1, x2, y2, w, h)

        det = int(q.get("det_index", -1))
        ridx = q.get("read_index", None)
        col = q.get("col_id", None)
        spanning = (col == -1)

        outline = "orange" if spanning else "lime"
        width = thickness + 1 if spanning else thickness
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)

        label = f"Q{det}"
        if ridx is not None:
            label += f" r{int(ridx)}"
        if col is not None and col != -1:
            label += f" c{int(col)}"
        if spanning:
            label += " S"
        _draw_label(draw, (x1 + 2, y1 + 2), label, font, fill=outline)

        q_center[det] = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    if draw_edges and edges:
        for det, fig_idx in edges:
            if det not in q_center or fig_idx not in fig_center:
                continue
            x1, y1 = q_center[det]
            x2, y2 = fig_center[fig_idx]
            draw.line([(x1, y1), (x2, y2)], fill="yellow", width=max(1, thickness - 1))

    im.save(out_path)
    return out_path


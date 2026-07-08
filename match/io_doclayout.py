from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


XYXY = Tuple[float, float, float, float]


@dataclass
class FigureBox:
    fig_index: int
    bbox_xyxy: XYXY
    label: str
    score: float
    cls_id: Optional[int] = None

    col_id: Optional[int] = None
    is_spanning: bool = False
    crop_path: Optional[str] = None
    bbox_xyxy_padded: Optional[Tuple[int, int, int, int]] = None

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_xyxy(v: Any) -> Optional[XYXY]:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(x) for x in v]
        return (x1, y1, x2, y2)
    except Exception:
        return None


def _unwrap_res(data: Dict[str, Any]) -> Dict[str, Any]:
    if "res" in data and isinstance(data["res"], dict):
        return data["res"]
    return data


def load_doclayout_json(
    json_path: str,
    *,
    image_stem: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    raw = json.loads(p.read_text(encoding="utf-8"))
    res = _unwrap_res(raw)

    stem = None
    if image_stem:
        stem = image_stem
    else:
        ip = res.get("input_path")
        if isinstance(ip, str) and ip:
            stem = Path(ip).stem
    if not stem:
        stem = p.stem
        stem = re.sub(r"(_res|_result)$", "", stem)

    return raw, stem


def extract_figures_from_doclayout(
    raw_data: Dict[str, Any],
    *,
    allowed_labels: Iterable[str] = ("image", "figure", "pic"),
    min_score: float = 0.0,
) -> List[FigureBox]:
    res = _unwrap_res(raw_data)
    boxes = res.get("boxes")
    if not isinstance(boxes, list):
        return []

    allow = {str(x).lower().strip() for x in allowed_labels if x and str(x).strip()}
    out: List[FigureBox] = []
    fig_idx = 1
    for b in boxes:
        if not isinstance(b, dict):
            continue
        label = str(b.get("label") or b.get("type") or "").lower().strip()
        if label not in allow:
            continue
        score = float(b.get("score", 1.0))
        if score < min_score:
            continue

        bbox = _safe_xyxy(b.get("coordinate")) or _safe_xyxy(b.get("bbox") or b.get("box"))
        if bbox is None:
            continue

        out.append(
            FigureBox(
                fig_index=fig_idx,
                bbox_xyxy=bbox,
                label=label,
                score=score,
                cls_id=int(b["cls_id"]) if "cls_id" in b else None,
            )
        )
        fig_idx += 1

    return out


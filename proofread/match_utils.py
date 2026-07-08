from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .img_utils import safe_open_image

def load_match_questions(match_json: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    j = json.loads(match_json.read_text(encoding="utf-8"))
    qs = j.get("questions", []) or []

    def keyf(q: Dict[str, Any]) -> Tuple[int, int]:
        ri = q.get("read_index")
        di = q.get("det_index")
        try:
            ri = int(ri) if ri is not None else 10**9
        except Exception:
            ri = 10**9
        try:
            di = int(di) if di is not None else 10**9
        except Exception:
            di = 10**9
        return (ri, di)

    qs_sorted = sorted(qs, key=keyf)
    return j, qs_sorted

def is_partial_question(q: Dict[str, Any]) -> bool:
    return (q.get("class_name") or "").strip().lower() == "partial_question"

def q_crop_hfrac(q: Dict[str, Any], page_h: Optional[float]) -> float:
    if not page_h:
        return 0.0
    b = q.get("bbox_xyxy") or q.get("bbox") or q.get("xyxy") or q.get("box")
    if not b or len(b) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = map(float, b)
        return max(0.0, y2 - y1) / float(page_h)
    except Exception:
        return 0.0

def get_crop_path(q: Dict[str, Any]) -> Optional[str]:
    for k in ("crop_path", "question_crop", "question_path", "q_crop_path", "path"):
        cp = (q.get(k) or "")
        if isinstance(cp, str) and cp.strip():
            return cp.strip()
    for k in ("question_dir", "q_dir", "dir"):
        d = (q.get(k) or "")
        if isinstance(d, str) and d.strip():
            dd = d.strip().rstrip("/\\")
            return f"{dd}/question.png"
    return None

def load_crop_image(page_dir: Path, q: Dict[str, Any]):
    cp = get_crop_path(q)
    if not cp:
        return None
    return safe_open_image(page_dir / cp)

def find_question_dir(page_dir: Path, q: Dict[str, Any]) -> Optional[Path]:
    cp = get_crop_path(q)
    if not cp:
        return None
    p = page_dir / cp
    if p.exists():
        return p.parent
    if p.parent.exists():
        return p.parent
    return None


def find_figure_candidates(page_dir: Path, q: Dict[str, Any]) -> List[Path]:
    cands: List[Path] = []
    qdir = find_question_dir(page_dir, q)
    if qdir and qdir.exists():
        for ext in ("png", "jpg", "jpeg", "webp"):
            cands += sorted(qdir.glob(f"figure_*.{ext}"))
    matches = q.get("matches")
    if isinstance(matches, list):
        for m in matches:
            if not isinstance(m, dict):
                continue
            fp = (m.get("fig_path") or m.get("crop_path") or m.get("path") or "").strip()
            if not fp:
                continue
            pp = page_dir / fp
            if pp.exists():
                cands.append(pp)

    seen = set()
    uniq: List[Path] = []
    for p in cands:
        key = p.as_posix()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


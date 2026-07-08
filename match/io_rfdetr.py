from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


XYXY = Tuple[float, float, float, float]


@dataclass
class QuestionBox:
    det_index: int
    bbox_xyxy: XYXY
    bbox_xyxy_padded: Optional[XYXY]
    class_id: int
    class_name: str
    score: float
    crop_path: Optional[str] = None

    col_id: Optional[int] = None
    read_index: Optional[int] = None


@dataclass
class RfDetrPage:
    image_stem: str
    image_path: str
    file_name: str
    width: int
    height: int
    overlay_path: Optional[str]
    questions: List[QuestionBox]

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_xyxy(v: Any) -> Optional[XYXY]:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    x1, y1, x2, y2 = v
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) <= 1e-6 or (y2 - y1) <= 1e-6:
        return None
    return (x1, y1, x2, y2)



def load_rfdetr_jsonl(
    jsonl_path: str,
    *,
    allowed_class_names: Optional[Iterable[str]] = None,
    min_score: float = 0.0,
) -> Dict[str, RfDetrPage]:
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    allow = None
    if allowed_class_names:
        allow = {s.strip() for s in allowed_class_names if s and s.strip()}

    pages: Dict[str, RfDetrPage] = {}
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                raise ValueError(f"RF-DETR jsonl 第 {line_no} 行 JSON 解析失败：{e}") from e

            file_name = str(rec.get("file_name") or "")
            image_path = str(rec.get("image_path") or "")
            stem = Path(file_name).stem if file_name else Path(image_path).stem
            if not stem:
                stem = f"page_{line_no:06d}"

            width = int(rec.get("width", 1))
            height = int(rec.get("height", 1))
            overlay_path = rec.get("overlay_path")

            dets = rec.get("detections") or []
            questions: List[QuestionBox] = []
            max_seen_index = 0
            for det in dets:
                if not isinstance(det, dict):
                    continue
                score = float(det.get("score", 0.0))
                if score < min_score:
                    continue

                cname = str(det.get("class_name", ""))
                if allow is not None and cname not in allow:
                    continue

                bbox = _safe_xyxy(det.get("bbox_xyxy")) or _safe_xyxy(det.get("bbox"))
                if bbox is None:
                    continue

                padded = _safe_xyxy(det.get("bbox_xyxy_padded"))

                det_index_raw = det.get("index")
                if det_index_raw is not None:
                    det_index = int(det_index_raw)
                    if det_index > max_seen_index:
                        max_seen_index = det_index
                else:
                    det_index = max_seen_index + 1
                    max_seen_index = det_index

                questions.append(
                    QuestionBox(
                        det_index=det_index,
                        bbox_xyxy=bbox,
                        bbox_xyxy_padded=padded,
                        class_id=int(det.get("class_id", -1)),
                        class_name=cname,
                        score=score,
                        crop_path=det.get("crop_path"),
                    )
                )

            pages[stem] = RfDetrPage(
                image_stem=stem,
                image_path=image_path,
                file_name=file_name or Path(image_path).name,
                width=width,
                height=height,
                overlay_path=overlay_path,
                questions=questions,
            )

    return pages


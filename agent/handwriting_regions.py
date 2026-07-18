from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


BBox = Tuple[int, int, int, int]


def _bbox(question: Dict[str, Any], width: int, height: int) -> BBox:
    raw = question.get("bbox_xyxy_padded") or question.get("bbox_xyxy") or []
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("question does not contain a valid bbox")
    x0, y0, x1, y1 = (int(round(float(value))) for value in raw)
    x0, x1 = max(0, min(x0, width - 1)), max(1, min(x1, width))
    y0, y1 = max(0, min(y0, height - 1)), max(1, min(y1, height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("question bbox is empty")
    return x0, y0, x1, y1


def _horizontal_overlap(a: BBox, b: BBox) -> float:
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    return overlap / max(1, min(a[2] - a[0], b[2] - b[0]))


def _main_question(question: Dict[str, Any]) -> bool:
    return "partial" not in str(question.get("class_name") or "").lower()


def classify_question_type(
    question: Dict[str, Any],
    question_text: str = "",
) -> Dict[str, Any]:
    """Classify a question using detector labels first, then text structure."""
    class_name = str(question.get("class_name") or "").strip().lower()
    if class_name in {"multiple_choice_question", "fill_blank_question"}:
        question_type = "choice" if class_name == "multiple_choice_question" else "fill_blank"
        return {
            "type": question_type,
            "confidence": 0.99,
            "source": "detector_class",
            "class_name": class_name,
        }

    text = str(question_text or "")
    option_markers = re.findall(
        r"(?:^|[\s\n])(?:[A-D][\.．、:：\)]|\([A-D]\)|（[A-D]）)",
        text,
        flags=re.I,
    )
    fill_signal = re.search(r"_{3,}|\\underline|填空题|填在.{0,8}(?:横线|空格)|横线上|空白处", text)
    if class_name == "problem_solving_question" and len(option_markers) < 2 and not fill_signal:
        return {
            "type": "short_answer",
            "confidence": 0.98,
            "source": "detector_class",
            "class_name": class_name,
        }
    if len(option_markers) >= 2 or re.search(r"选择题|单选|多选", text):
        return {
            "type": "choice",
            "confidence": 0.91 if class_name == "problem_solving_question" else (0.88 if len(option_markers) >= 2 else 0.78),
            "source": "text_override_detector" if class_name == "problem_solving_question" else "question_text",
            "class_name": class_name,
        }
    if fill_signal:
        return {
            "type": "fill_blank",
            "confidence": 0.90 if class_name == "problem_solving_question" else 0.84,
            "source": "text_override_detector" if class_name == "problem_solving_question" else "question_text",
            "class_name": class_name,
        }
    if re.search(r"求证|证明|解答|计算|求.{0,16}(?:值|范围|函数|概率|面积|长度)", text):
        return {
            "type": "short_answer",
            "confidence": 0.78,
            "source": "question_text",
            "class_name": class_name,
        }
    return {
        "type": "unknown",
        "confidence": 0.35,
        "source": "fallback",
        "class_name": class_name,
    }


def _column_components(boxes: Sequence[BBox]) -> List[List[int]]:
    """Group question detections into columns using transitive x-overlap."""
    remaining = set(range(len(boxes)))
    groups: List[List[int]] = []
    while remaining:
        seed = remaining.pop()
        group = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            linked = {
                other
                for other in remaining
                if _horizontal_overlap(boxes[current], boxes[other]) >= 0.28
            }
            remaining.difference_update(linked)
            group.update(linked)
            frontier.extend(linked)
        groups.append(sorted(group))
    return groups


def score_and_divide_question_frame(
    page_size: Tuple[int, int],
    questions: Sequence[Dict[str, Any]],
    target_index: int,
    question_text: str = "",
) -> Dict[str, Any]:
    """Build a scored handwriting frame using question-type-specific rules."""
    width, height = page_size
    if target_index < 0 or target_index >= len(questions):
        raise IndexError(target_index)

    target = questions[target_index]
    target_box = _bbox(target, width, height)
    target_is_spanning = bool(target.get("is_spanning"))
    main_indices = [
        index
        for index, question in enumerate(questions)
        if _main_question(question)
        and not question.get("is_spanning")
        and not target_is_spanning
    ]
    if target_index not in main_indices:
        main_indices.append(target_index)
    main_boxes = [_bbox(questions[index], width, height) for index in main_indices]
    groups = _column_components(main_boxes)
    local_target = main_indices.index(target_index)
    local_group = next(group for group in groups if local_target in group)
    column_indices = [main_indices[index] for index in local_group]
    column_boxes = [_bbox(questions[index], width, height) for index in column_indices]

    column_left = min(box[0] for box in column_boxes)
    column_right = max(box[2] for box in column_boxes)
    other_extents = []
    for group in groups:
        if group == local_group:
            continue
        group_boxes = [main_boxes[index] for index in group]
        other_extents.append((min(box[0] for box in group_boxes), max(box[2] for box in group_boxes)))

    x_pad = max(12, int(round(width * 0.025)))
    frame_left = max(0, column_left - x_pad)
    frame_right = min(width, column_right + x_pad)
    for other_left, other_right in other_extents:
        if other_right <= column_left:
            frame_left = max(frame_left, int(round((other_right + column_left) / 2)))
        elif other_left >= column_right:
            frame_right = min(frame_right, int(round((column_right + other_left) / 2)))

    ordered = sorted(
        ((_bbox(questions[index], width, height), index) for index in column_indices),
        key=lambda pair: (pair[0][1], pair[0][0]),
    )
    target_order_index = next(
        position for position, (_, index) in enumerate(ordered) if index == target_index
    )
    previous_box = ordered[target_order_index - 1][0] if target_order_index > 0 else None
    next_box = next(
        (box for box, index in ordered if box[1] > target_box[1] and index != target_index),
        None,
    )
    type_info = classify_question_type(target, question_text)
    question_type = type_info["type"]
    y_pad = max(6, int(round(height * 0.008)))
    available_top = max(0, target_box[1] - y_pad)
    if next_box is None:
        available_bottom = height
        available_boundary_kind = "page_bottom"
        available_boundary_confidence = 0.78
    else:
        available_bottom = max(target_box[3], next_box[1] - y_pad)
        available_boundary_kind = "next_question"
        available_boundary_confidence = 0.96

    target_height = target_box[3] - target_box[1]
    local_x_pad = max(10, int(round(width * 0.015)))
    adaptive_details: Dict[str, Any] = {}
    if question_type == "choice":
        compact_text = re.sub(r"\s+", "", str(question_text or ""))
        text_chars = len(compact_text)
        text_lines = max(1, len([line for line in str(question_text or "").splitlines() if line.strip()]))
        option_count = len(
            re.findall(r"(?:^|[\s\n])(?:[A-D][\.．、:：\)]|\([A-D]\)|（[A-D]）)", str(question_text or ""), flags=re.I)
        )
        length_factor = min(1.0, max(0.0, (text_chars - 35) / 260.0))
        line_factor = min(1.0, max(0.0, (text_lines - 2) / 5.0))
        structure_factor = min(1.0, max(length_factor, line_factor, option_count / 6.0))
        adaptive_x_pad = max(
            local_x_pad,
            int(round(width * (0.022 + 0.018 * structure_factor))),
        )
        adaptive_y_pad = max(
            y_pad * 2,
            int(round(height * (0.014 + 0.018 * structure_factor))),
            int(round(target_height * (0.12 + 0.10 * structure_factor))),
        )
        frame_left = max(frame_left, target_box[0] - adaptive_x_pad)
        frame_right = min(frame_right, target_box[2] + adaptive_x_pad)
        choice_top_limit = min(target_box[1], previous_box[3]) if previous_box else 0
        choice_bottom_limit = max(target_box[3], next_box[1]) if next_box else height
        choice_frame_top = max(choice_top_limit, target_box[1] - adaptive_y_pad)
        frame_bottom = min(choice_bottom_limit, target_box[3] + adaptive_y_pad)
        boundary_kind = "choice_adaptive_margin"
        boundary_confidence = 0.94
        strategy = "adaptive_question_box_for_option_marks_and_corrections"
        adaptive_details = {
            "text_chars": text_chars,
            "text_lines": text_lines,
            "option_count": option_count,
            "length_factor": round(length_factor, 4),
            "structure_factor": round(structure_factor, 4),
            "horizontal_margin_px": adaptive_x_pad,
            "vertical_margin_px": adaptive_y_pad,
            "previous_question_limit": choice_top_limit,
            "next_question_limit": choice_bottom_limit,
        }
    elif question_type == "fill_blank":
        frame_left = max(frame_left, target_box[0] - local_x_pad)
        frame_right = min(frame_right, target_box[2] + local_x_pad)
        extra_bottom = max(int(round(height * 0.045)), int(round(target_height * 0.35)))
        frame_bottom = min(available_bottom, target_box[3] + extra_bottom)
        boundary_kind = "fill_blank_margin"
        boundary_confidence = 0.90
        strategy = "question_box_plus_blank_margin"
    else:
        frame_bottom = available_bottom
        boundary_kind = available_boundary_kind
        boundary_confidence = available_boundary_confidence
        strategy = (
            "extend_to_next_question_for_working_area"
            if question_type == "short_answer"
            else "conservative_column_interval"
        )

    frame_bottom = max(target_box[3], frame_bottom)
    if question_type == "short_answer":
        frame_top = min(target_box[3], height - 1)
        contains_stem = False
        answer_start_rule = "after_stem_bottom"
    elif question_type == "choice":
        frame_top = choice_frame_top
        contains_stem = True
        answer_start_rule = "adaptive_include_option_marks_and_nearby_corrections"
    else:
        frame_top = available_top
        contains_stem = True
        answer_start_rule = "include_in_question_answer_marks"
    if frame_bottom <= frame_top:
        frame_bottom = min(height, frame_top + 1)

    frame = (frame_left, frame_top, frame_right, frame_bottom)
    detector_score = max(0.0, min(1.0, float(target.get("score") or 0.0)))
    class_score = 1.0 if _main_question(target) else 0.55
    frame_height_ratio = (frame_bottom - frame_top) / max(1, height)
    if 0.12 <= frame_height_ratio <= 0.62:
        geometry_score = 1.0
    elif 0.07 <= frame_height_ratio <= 0.78:
        geometry_score = 0.75
    else:
        geometry_score = 0.45
    selection_score = round(
        0.46 * detector_score
        + 0.16 * class_score
        + 0.16 * boundary_confidence
        + 0.10 * geometry_score
        + 0.12 * float(type_info["confidence"]),
        4,
    )
    return {
        "question_type": type_info,
        "strategy": strategy,
        "contains_stem": contains_stem,
        "answer_start_rule": answer_start_rule,
        "adaptive_details": adaptive_details,
        "source_bbox_xyxy": list(target_box),
        "frame_bbox_xyxy": list(frame),
        "column_question_indices": column_indices,
        "boundary_kind": boundary_kind,
        "score": selection_score,
        "score_breakdown": {
            "detector": round(detector_score, 4),
            "class": class_score,
            "boundary": boundary_confidence,
            "geometry": geometry_score,
            "question_type": round(float(type_info["confidence"]), 4),
        },
    }


def _vertical_tiles(frame: BBox, count: int = 2, overlap_ratio: float = 0.16) -> List[BBox]:
    x0, y0, x1, y1 = frame
    frame_height = y1 - y0
    if count <= 1 or frame_height < 180:
        return [frame]
    tile_height = int(round(frame_height / count * (1.0 + overlap_ratio)))
    max_start = max(y0, y1 - tile_height)
    starts = [int(round(y0 + (max_start - y0) * index / (count - 1))) for index in range(count)]
    return [(x0, start, x1, min(y1, start + tile_height)) for start in starts]


def save_handwriting_views(
    page_image: Image.Image,
    region: Dict[str, Any],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Save a full question frame plus overlapping detail views for the VLM."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = tuple(int(value) for value in region["frame_bbox_xyxy"])
    view_boxes = [("full", frame)]
    view_boxes.extend(
        (f"detail_{index:02d}", tile)
        for index, tile in enumerate(_vertical_tiles(frame), start=1)
    )

    views: List[Dict[str, Any]] = []
    for kind, bbox in view_boxes:
        path = output_dir / f"{kind}.png"
        page_image.crop(bbox).save(path, format="PNG", optimize=True)
        views.append({"kind": kind, "bbox_xyxy": list(bbox), "path": str(path)})
    return views


def _overlay_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_overlay_label(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    color: Tuple[int, int, int],
) -> None:
    x, y = xy
    text_bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(
        [text_bbox[0] - 3, text_bbox[1] - 2, text_bbox[2] + 3, text_bbox[3] + 2],
        fill=(15, 15, 15),
    )
    draw.text((x, y), text, font=font, fill=color)


def draw_handwriting_overlay(
    page_image: Image.Image,
    regions: Sequence[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Draw printed-stem and handwriting frames together on the source page."""
    canvas = page_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line_width = max(2, int(round(min(width, height) / 420)))
    font_size = max(13, int(round(min(width, height) / 100)))
    font = _overlay_font(font_size)
    stem_color = (0, 220, 90)
    answer_color = (255, 55, 200)

    _draw_overlay_label(draw, (8, 8), "GREEN=STEM  MAGENTA=HANDWRITING REGION", font, (255, 230, 0))
    for item in regions:
        region = item.get("region") if isinstance(item.get("region"), dict) else item
        stem = region.get("source_bbox_xyxy") or []
        answer = region.get("frame_bbox_xyxy") or []
        if len(stem) != 4 or len(answer) != 4:
            continue
        stem_box = tuple(int(round(float(value))) for value in stem)
        answer_box = tuple(int(round(float(value))) for value in answer)
        qno = item.get("qno")
        qlabel = str(qno) if qno is not None else "?"
        type_info = region.get("question_type") or {}
        qtype = str(type_info.get("type") or "unknown")
        type_confidence = float(type_info.get("confidence") or 0.0)
        score = float(region.get("score") or 0.0)

        draw.rectangle(stem_box, outline=stem_color, width=line_width)
        draw.rectangle(answer_box, outline=answer_color, width=line_width)
        stem_y = min(height - font_size - 7, stem_box[1] + 5)
        answer_y = max(8, answer_box[3] - font_size - 7)
        _draw_overlay_label(
            draw,
            (max(3, stem_box[0] + 3), stem_y),
            f"Q{qlabel} S {qtype} {type_confidence:.2f}",
            font,
            stem_color,
        )
        _draw_overlay_label(
            draw,
            (max(3, answer_box[0] + 3), answer_y),
            f"Q{qlabel} A {score:.3f}",
            font,
            answer_color,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return output_path

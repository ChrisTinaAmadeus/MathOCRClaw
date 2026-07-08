from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .cache import JsonCache
from .common import parse_verdict_letter, sha1_text
from .img_utils import BASE_ENHANCE, enhance_for_vlm, img_to_data_url, safe_open_image, blank_frac

FIG_CLS_PROMPT = '''
You are a background-figure classifier for exam paper illustrations. Output only one label: printed_figure / student_annotation / noise (no explanation).

Priority:
1) Never classify a purely hand-drawn sketch as printed_figure
2) A printed base figure with handwritten annotations / red-pen overlays is still printed_figure
3) If uncertain, output noise (do not casually predict student_annotation)

[printed_figure (must satisfy: at least one "printed anchor" is visible)]
Printed anchors (if any one of the following appears, classify as printed_figure):
- Regular dashed lines: dash lengths / spacing are highly consistent (common for hidden edges)
- Printed labels: at least 2 letters / numbers inside the figure look like printed font (consistent size and shape, unlike handwriting)
- Standard structures such as coordinate axes / grids / ticks / tables / statistical chart frameworks / circuit symbols
- Uniform gray filled regions (clean boundaries, even fill, e.g. shaded regions in set diagrams)
- Clear "two layers": faint base figure lines + darker / thicker handwritten overlays (the overlays do not invalidate the base figure)
- Faint gray or thin black geometric framework: faint gray / thin black straight lines forming a complete geometric figure (closed polygon / 3D frame), with relatively straight and consistent lines; even with red-pen overlays, it is still printed_figure

[student_annotation (pure hand-drawing)]
Requirements: no printed anchor is visible + the content is mainly freehand strokes (wobbly lines, varying thickness, repeated tracing, handwritten letters / numbers).

Special notes:
- If a faint gray / thin black geometric framework is visible, do not classify as student_annotation
- Do not classify as printed_figure just because the geometry looks neat / cube-like / circle-like
- A single circle / ellipse plus a few straight / curved lines (without dashed lines / grids / or >=2 printed labels) should be treated as hand-drawn by default

[noise]
Blank area / shadow / stains / body text only / too blurry or too small to determine whether any printed anchor exists -> noise

Output rules:
- First look for printed anchors: if present -> printed_figure
- If no printed anchor is visible but it is clearly hand-drawn -> student_annotation
- Otherwise -> noise
'''
FIG_REL_PROMPT = '''
You are a relevance verifier for exam questions and associated figures. Output only a single letter: Y / N / U (no extra characters / spaces / line breaks).

Overall strategy: it is better to give fewer Ys than to mistakenly include hand-drawn drafts as associated figures; however, when the question truly seems to require a figure and the image may be a printed figure heavily covered by handwriting, prefer U over N.

1) [Question requirement verification]
- If the question explicitly or strongly implicitly requires a figure (e.g. "as shown in the figure", "in the figure below", "in the diagram", "schematic", "function graph", "coordinate system", "statistical chart", "in the table", "circuit diagram", "point A in the figure", etc.) -> go to 2
- Otherwise -> output N

2) [Nature of the candidate figure (ignore handwriting)]
- If the candidate is clearly a pure hand-drawn draft / doodle / scratch work, with no structured printed skeleton -> output N
- If a printed structured skeleton can be confirmed (coordinate axes / grid / table frame / statistical chart / circuit symbols / regular geometric figure / printed labels inside the figure) -> go to 3
- If the question requires a figure, but the candidate is heavily damaged or covered by handwriting and it is impossible to confirm whether a printed skeleton exists -> output U (do not directly output N)

3) [Relevance verification (spatial + semantic)]
- If the figure lies within the region from this question number to the end of the question (or is adjacent to it and clearly referenced by expressions such as "as shown in the figure" / "in the figure"), and the figure type matches the question text -> output Y
- If it obviously belongs to another question / the position clearly does not correspond -> output U
- If the figure type clearly does not match the question text -> output N

Mandatory rules:
- Printed figure + handwritten annotations: as long as the skeleton is confirmable and matches, output Y
- If the question requires a figure but it is uncertain whether the image contains a printed skeleton, output U (not directly N)
'''

@dataclass
class FigureFilterCfg:
    min_edge: int = 28
    max_aspect: float = 8.0
    max_blank_frac: float = 0.97
    do_vlm_cls: bool = True
    do_vlm_rel: bool = True

def rule_filter_image(img: Image.Image, cfg: FigureFilterCfg) -> Tuple[bool, str]:
    w, h = img.size
    if min(w, h) < cfg.min_edge:
        return False, "too_small"
    asp = max(w, h) / float(max(1, min(w, h)))
    if asp > cfg.max_aspect:
        return False, "too_aspect"
    if blank_frac(img) > cfg.max_blank_frac:
        return False, "too_blank"
    return True, "ok"

def fig_cls_once(vlm, fig_img: Image.Image, *, cache: Optional[JsonCache]) -> str:
    fig2 = enhance_for_vlm(fig_img, BASE_ENHANCE)
    data_url, img_hash = img_to_data_url(fig2)
    ck = f"{vlm.cache_tag}::{img_hash}::fig_cls"
    if cache:
        hit = cache.get("fig_cls", ck)
        if isinstance(hit, str) and hit:
            return hit
    raw = vlm.invoke(
        [
            {"type": "text", "text": FIG_CLS_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
        #temperature=0.0,
        max_tokens=16,
    ).strip()
    raw_l = raw.lower()
    if "printed_figure" in raw_l:
        lab = "printed_figure"
    elif "student_annotation" in raw_l:
        lab = "student_annotation"
    elif "noise" in raw_l:
        lab = "noise"
    else:
        # fallback: be conservative -> not a printed figure
        lab = "noise"
    if cache:
        cache.set("fig_cls", ck, lab)
    return lab

def fig_rel_once(vlm, q_img: Image.Image, fig_img: Image.Image, *, cache: Optional[JsonCache]) -> str:
    q2 = enhance_for_vlm(q_img, BASE_ENHANCE)
    f2 = enhance_for_vlm(fig_img, BASE_ENHANCE)
    q_url, q_hash = img_to_data_url(q2)
    f_url, f_hash = img_to_data_url(f2)
    ck = f"{vlm.cache_tag}::{q_hash}::{f_hash}::fig_rel"
    if cache:
        hit = cache.get("fig_rel", ck)
        if isinstance(hit, str) and hit in ("Y", "N", "U"):
            return hit
    raw = vlm.invoke(
        [
            {"type": "text", "text": FIG_REL_PROMPT},
            {"type": "image_url", "image_url": {"url": q_url}},
            {"type": "image_url", "image_url": {"url": f_url}},
        ],
        #temperature=0.0,
        max_tokens=8,
    )
    v = parse_verdict_letter(raw)
    if cache:
        cache.set("fig_rel", ck, v)
    return v

def select_figures_for_question(
    q_img: Image.Image,
    fig_paths: List[Path],
    fig_vlm,
    *,
    cache: Optional[JsonCache],
    cfg: FigureFilterCfg,
) -> Tuple[List[Path], List[Dict[str, Any]]]:
    kept: List[Path] = []
    debug: List[Dict[str, Any]] = []
    for p in fig_paths:
        img = safe_open_image(p)
        if img is None:
            debug.append({"path": str(p), "keep": False, "reason": "open_fail"})
            continue
        ok, reason = rule_filter_image(img, cfg)
        if not ok:
            debug.append({"path": str(p), "keep": False, "reason": reason})
            continue
        lab = None
        if cfg.do_vlm_cls and fig_vlm is not None:
            lab = fig_cls_once(fig_vlm, img, cache=cache)
            if lab != "printed_figure":
                debug.append({"path": str(p), "keep": False, "reason": f"cls={lab}"})
                continue
        rel = None
        if cfg.do_vlm_rel and fig_vlm is not None:
            rel = fig_rel_once(fig_vlm, q_img, img, cache=cache)
            if rel != "Y":
                debug.append({"path": str(p), "keep": False, "reason": f"rel={rel}", "cls": lab})
                continue
        kept.append(p)
        debug.append({"path": str(p), "keep": True, "reason": "ok", "cls": lab, "rel": rel})
    return kept, debug


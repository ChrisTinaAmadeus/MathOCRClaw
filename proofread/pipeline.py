from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .align import assign_crops_to_md_blocks
from .cache import JsonCache
from .common import normalize_ws, extract_qno_from_text_block
from .figures import FigureFilterCfg, select_figures_for_question
from .match_utils import find_figure_candidates, load_crop_image, load_match_questions
from .md_utils import PageBlock, split_page_into_blocks
from .repair import RepairConfig, repair_question_v6_6
from .verify import verify_question_strict, verify_question_lenient, verify_evidence, verify_has_options
from .img_utils import safe_open_image


def process_one_page(
    page_dir: Path,
    page_md_path: Path,
    *,
    out_dir: Path,
    ver_vlm,
    gen_vlm=None,
    fig_vlm=None,
    cache: Optional[JsonCache] = None,
    skip_partial: bool = True,
    partial_main_min_hfrac: float = 0.0,
    use_offset_search: bool = True,
    use_crop_qno: bool = True,
    qno_vlm=None,
    max_qchars: int = 900,
    mask_u_token: str = "【无法识别】",
    mask_n_token: str = "【疑似幻觉】",
    fig_cfg: Optional[FigureFilterCfg] = None,
    weak_when_no_crop_qno: bool = True,  
    weak_mask_on_weak: bool = True,     
    weak_mask_on_u: bool = False,        
    mask_no_crop: bool = False,           
    weak_keep_figures: bool = True,       
    fixed_enable: bool = True,
    fixed_token: str = "【幻觉已修正】",
    fixed_after_allow_u: bool = True,     
    fixed_max_del_ratio: float = 0.20,    
    fixed_min_keep_options: int = 2,
    fixed_marker_mode: str = "append_line", 
    ablation_no_patcher: bool = False,
    verdict_comment: bool = False,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if (not ablation_no_patcher) and (gen_vlm is None):
        raise ValueError("gen_vlm must be provided unless --no-patcher ablation is enabled")

    if ablation_no_patcher:
        fixed_enable = False

    md = page_md_path.read_text(encoding="utf-8", errors="ignore")
    blocks: List[PageBlock] = split_page_into_blocks(md)

    match_json = page_dir / "match.json"
    report: Dict[str, Any] = {
        "page_dir": str(page_dir),
        "page_md": str(page_md_path),
        "n_blocks": len(blocks),
        "align": None,
        "align_safety": None,
        "items": [],
    }

    if not match_json.exists():
        out_md = "\n\n".join([b.text for b in blocks if b.text.strip()])
        (out_dir / f"{page_dir.name}_proofread.md").write_text(out_md, encoding="utf-8")
        report["error"] = "match.json not found"
        (out_dir / f"{page_dir.name}_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report

    match_meta, qs_sorted = load_match_questions(match_json)

    page_h = None
    m = {}

    if isinstance(match_meta, dict):
        for k in ("page_h", "height", "H", "img_h"):
            v = match_meta.get(k)
            if v is not None:
                page_h = v
                break
        if isinstance(match_meta.get("meta"), dict):
            m = match_meta.get("meta") or {}

        if page_h is None and m:
            page_h = m.get("page_h") or m.get("height") or m.get("H") or m.get("img_h")

        page = m.get("page") if isinstance(m, dict) else None
        if page_h is None and isinstance(page, dict):
            page_h = page.get("H") or page.get("height") or page.get("img_h") or page.get("page_h")

    if page_h is None:
        y2s = []
        for q in qs_sorted:
            b = (
                q.get("bbox_xyxy_padded")
                or q.get("bbox_xyxy")
                or q.get("bbox")
                or q.get("xyxy")
                or q.get("box")
            )
            if isinstance(b, (list, tuple)) and len(b) == 4:
                try:
                    y2s.append(float(b[3]))
                except Exception:
                    pass
        if y2s:
            page_h = max(y2s)

    try:
        page_h = float(page_h) if page_h is not None else None
    except Exception:
        page_h = None

    align_res = assign_crops_to_md_blocks(
        blocks,
        qs_sorted,
        page_dir=page_dir,
        page_h=page_h,
        skip_partial=skip_partial,
        partial_main_min_hfrac=partial_main_min_hfrac,
        use_offset_search=use_offset_search,
        use_crop_qno=use_crop_qno,
        qno_vlm=qno_vlm,
        cache=cache,
    )
    report["align"] = align_res.debug

    known_crop_qnos: set[int] = set()
    try:
        _cqs = (align_res.debug or {}).get("crop_qnos") or []
        for x in _cqs:
            if isinstance(x, int) and x > 0:
                known_crop_qnos.add(x)
    except Exception:
        known_crop_qnos = set()

    pair_by_block: Dict[int, Dict[str, Any]] = {}
    pairs = None
    if isinstance(getattr(align_res, "debug", None), dict):
        pairs = align_res.debug.get("pairs")
    if isinstance(pairs, list):
        for p in pairs:
            try:
                bi = int(p.get("md_block"))
            except Exception:
                continue
            if bi not in pair_by_block:
                pair_by_block[bi] = dict(p)

    def _first_mismatch_from_pairs(pairs_list: Any) -> Optional[int]:
        if not isinstance(pairs_list, list):
            return None
        for p in pairs_list:
            if not isinstance(p, dict):
                continue
            a = p.get("qno")
            b = p.get("crop_qno")
            if a is None or b is None:
                continue
            try:
                if int(a) != int(b):
                    return int(a)
            except Exception:
                continue
        return None
    def _bbox_area_from_qrec(q: Dict[str, Any]) -> float:
        b = (
            q.get("bbox_xyxy_padded")
            or q.get("bbox_xyxy")
            or q.get("bbox")
            or q.get("xyxy")
            or q.get("box")
        )
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            return 0.0
        try:
            x1, y1, x2, y2 = map(float, b)
        except Exception:
            return 0.0
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


    def _median_fallback(xs: List[float]) -> Optional[float]:
        """Robust median with no external dependency."""
        if not xs:
            return None
        xs2 = sorted(float(x) for x in xs)
        n = len(xs2)
        mid = n // 2
        if n % 2 == 1:
            return xs2[mid]
        return 0.5 * (xs2[mid - 1] + xs2[mid])

    non_unknown_areas: List[float] = []
    for q in qs_sorted:
        try:
            cp0 = q.get("crop_path") or q.get("crop") or q.get("path")
        except Exception:
            cp0 = None

        is_unk0 = False
        if cp0 and "q999999" in str(cp0):
            is_unk0 = True
        elif q.get("read_index", None) is None:
            is_unk0 = True
        if is_unk0:
            continue

        a = _bbox_area_from_qrec(q)
        if a > 1e-6:
            non_unknown_areas.append(float(a))

    area_med = _median_fallback(non_unknown_areas)

    def _merge_suspect_qidx(q_idx: int) -> bool:
        if area_med is None:
            return False
        try:
            q_idx_i = int(q_idx)
        except Exception:
            return False
        if not (0 <= q_idx_i < len(qs_sorted)):
            return False
        a = _bbox_area_from_qrec(qs_sorted[q_idx_i])
        if a <= 1e-6:
            return False
        return a > float(area_med) * 1.6


    def _merge_suspect_override_ok(
        qrec: Any,
        *,
        min_score_v2: float = 2.8,
        min_inside: float = 0.95,
        min_in_window: float = 0.95,
        min_iof: float = 0.60,
        max_matches: int = 2,
    ) -> bool:
        if not isinstance(qrec, dict):
            return False
        ms = qrec.get("matches")
        if not isinstance(ms, list) or len(ms) == 0:
            return False
        if (max_matches is not None) and (len(ms) > int(max_matches)):
            return False

        best = None
        best_sv2 = -1.0
        for m in ms:
            if not isinstance(m, dict):
                continue
            try:
                sv2 = float(m.get("score_v2", m.get("score", 0.0)) or 0.0)
            except Exception:
                sv2 = 0.0
            if sv2 > best_sv2:
                best_sv2, best = sv2, m

        if not isinstance(best, dict):
            return False

        try:
            inside = float(best.get("inside", 0.0) or 0.0)
            inw = float(best.get("in_window", 0.0) or 0.0)
            iof = float(best.get("iof", 0.0) or 0.0)
        except Exception:
            return False
        return (best_sv2 >= float(min_score_v2)) and (inside >= float(min_inside)) and (inw >= float(min_in_window)) and (iof >= float(min_iof))

    def _qno_ok(md_qno: Any, crop_qno: Any, align_trust_high: bool) -> bool:
        if md_qno is None:
            return False
        try:
            if crop_qno is not None:
                return int(crop_qno) == int(md_qno)
        except Exception:
            return False
        return bool(align_trust_high)

    def _extract_final_verdict(dbg: Any) -> Optional[str]:
        if not isinstance(dbg, dict):
            return None

        for k in (
            "v_after_repair",
            "final_verdict",
            "verdict_after",
            "v_strict",
            "v_lenient",
            "v_soft",
            "verdict",
            "v_after",
            "v",
        ):
            v = dbg.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip().upper()

        vv = dbg.get("verifier")
        if isinstance(vv, dict):
            v = vv.get("verdict")
            if isinstance(v, str) and v.strip():
                return v.strip().upper()

        return None

    def _norm_match_class(x: Any) -> str:
        s = (x or "")
        return str(s).strip().lower()

    def _is_fill_blank(match_class: Any) -> bool:
        mc = _norm_match_class(match_class)
        return ("fill_blank" in mc) or (mc == "blank") or (mc == "fill")

    TINY_CROP_MIN_H_DEFAULT = 120
    TINY_CROP_MIN_H_FILL_BLANK = 55 

    FILL_RECROP_TRIGGER_H = 120     
    FILL_Y_EXPAND_RATIO = 0.90
    FILL_Y_EXPAND_MIN = 80
    FILL_Y_EXPAND_MAX = 600

    page_img_path: Optional[Path] = None
    if isinstance(match_meta, dict):
        ip = match_meta.get("image_path") or match_meta.get("img_path") or match_meta.get("page_image")
        if isinstance(ip, str) and ip.strip():
            page_img_path = Path(ip.strip())

    meta_w = None
    meta_h = None
    if isinstance(match_meta, dict):
        for k in ("page_w", "width", "W", "img_w"):
            if match_meta.get(k) is not None:
                meta_w = match_meta.get(k)
                break
        for k in ("page_h", "height", "H", "img_h"):
            if match_meta.get(k) is not None:
                meta_h = match_meta.get(k)
                break
        if (meta_w is None or meta_h is None) and isinstance(match_meta.get("meta"), dict):
            mm = match_meta.get("meta") or {}
            meta_w = meta_w or mm.get("page_w") or mm.get("width") or mm.get("W") or mm.get("img_w")
            meta_h = meta_h or mm.get("page_h") or mm.get("height") or mm.get("H") or mm.get("img_h")

    try:
        meta_w = int(meta_w) if meta_w is not None else None
    except Exception:
        meta_w = None
    try:
        meta_h = int(meta_h) if meta_h is not None else None
    except Exception:
        meta_h = None

    _page_img_cache = {"img": None}

    def _get_page_img():
        if _page_img_cache["img"] is not None:
            return _page_img_cache["img"]
        if page_img_path is None:
            return None
        img = safe_open_image(page_img_path)
        _page_img_cache["img"] = img
        return img

    def _get_bbox_xyxy(q: Dict[str, Any]) -> Optional[tuple]:
        b = (
            q.get("bbox_xyxy_padded")
            or q.get("bbox_xyxy")
            or q.get("bbox")
            or q.get("xyxy")
            or q.get("box")
        )
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            return None
        try:
            x1, y1, x2, y2 = map(float, b)
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _tiny_h_threshold(match_class: Any) -> int:
        return TINY_CROP_MIN_H_FILL_BLANK if _is_fill_blank(match_class) else TINY_CROP_MIN_H_DEFAULT

    def _is_tiny_crop(img, match_class: Any) -> bool:
        if img is None:
            return True
        try:
            w, h = img.size
        except Exception:
            return True
        return h < _tiny_h_threshold(match_class)

    def _recrop_fill_blank_from_page(
        qrec: Dict[str, Any],
        match_class: Any,
        *,
        y_expand_ratio: float = FILL_Y_EXPAND_RATIO,
        y_expand_min: int = FILL_Y_EXPAND_MIN,
        y_expand_max: int = FILL_Y_EXPAND_MAX,
    ):
        if not _is_fill_blank(match_class):
            return None, None

        page_img = _get_page_img()
        if page_img is None:
            return None, None

        bbox = _get_bbox_xyxy(qrec)
        if bbox is None:
            return None, None

        img_w, img_h = page_img.size
        x1, y1, x2, y2 = bbox

        if meta_w and meta_h and meta_w > 0 and meta_h > 0:
            sx = img_w / float(meta_w)
            sy = img_h / float(meta_h)
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy

        bh = max(1.0, (y2 - y1))
        pad = max(int(bh * y_expand_ratio), int(y_expand_min))
        pad = min(pad, int(y_expand_max))

        yy1 = max(0, int(y1 - pad))
        yy2 = min(img_h, int(y2 + pad))
        xx1 = max(0, int(x1))
        xx2 = min(img_w, int(x2))

        if (xx2 - xx1) <= 2 or (yy2 - yy1) <= 2:
            return None, None

        try:
            recrop = page_img.crop((xx1, yy1, xx2, yy2))
        except Exception:
            return None, None

        dbg = {
            "recrop_fill_blank": True,
            "src_page": str(page_img_path) if page_img_path else None,
            "bbox_xyxy_used": [xx1, yy1, xx2, yy2],
            "pad": pad,
            "meta_wh": [meta_w, meta_h],
            "page_wh": [img_w, img_h],
        }
        return recrop, dbg



    def _recrop_generic_from_page(
        qrec: Dict[str, Any],
        *,
        pad_x_ratio: float = 0.06,
        pad_y_ratio: float = 0.08,
        min_pad_x: int = 30,
        min_pad_y: int = 40,
    ):
        bbox = _get_bbox_xyxy(qrec)
        if bbox is None:
            return None, {"recrop": "generic_no_bbox"}

        x1, y1, x2, y2 = bbox

        page_img = _get_page_img()
        used_img_path = str(page_img_path) if page_img_path is not None else None
        used_meta_w = meta_w
        used_meta_h = meta_h

        if page_img is None:
            img_path = qrec.get("image_path") or qrec.get("page_image") or qrec.get("page_img")
            if img_path:
                page_img = safe_open_image(img_path)
                used_img_path = str(img_path)

        if page_img is None:
            return None, {"recrop": "generic_open_page_fail", "img_path": used_img_path}

        if used_meta_w is None:
            used_meta_w = qrec.get("meta_w") or qrec.get("page_w") or qrec.get("width")
        if used_meta_h is None:
            used_meta_h = qrec.get("meta_h") or qrec.get("page_h") or qrec.get("height")

        W, H = page_img.size

        sx = 1.0
        sy = 1.0
        if used_meta_w and used_meta_h:
            try:
                sx = W / float(used_meta_w)
                sy = H / float(used_meta_h)
            except Exception:
                sx = 1.0
                sy = 1.0

        gx1, gy1, gx2, gy2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
        bw = max(1.0, gx2 - gx1)
        bh = max(1.0, gy2 - gy1)

        padx = max(float(min_pad_x), bw * float(pad_x_ratio))
        pady = max(float(min_pad_y), bh * float(pad_y_ratio))

        rx1 = max(0, int(gx1 - padx))
        ry1 = max(0, int(gy1 - pady))
        rx2 = min(W, int(gx2 + padx))
        ry2 = min(H, int(gy2 + pady))

        if rx2 <= rx1 or ry2 <= ry1:
            return None, {"recrop": "generic_empty"}

        return page_img.crop((rx1, ry1, rx2, ry2)), {
            "recrop": "generic_ok",
            "src_page": used_img_path,
            "bbox_meta": [x1, y1, x2, y2],
            "bbox_pix": [rx1, ry1, rx2, ry2],
            "scale": [sx, sy],
            "meta_wh": [used_meta_w, used_meta_h],
            "page_wh": [W, H],
        }

    def _normalize_verdict(v: Optional[str]) -> str:
        v = (v or "").strip().upper()
        return v if v in ("Y", "N", "U") else "U"

    def _safe_weak_verdict(v: Optional[str], weak_reasons: List[str]) -> str:
        vv = _normalize_verdict(v)
        if vv == "N" and any(r in weak_reasons for r in ("qno_mismatch_soft", "merge_suspect", "no_crop_qno", "unknown")):
            return "U"
        return vv

    def _weak_output(md_text: str, verdict: Optional[str], *, weak_mask: bool, qno: Optional[int] = None) -> str:
        t = normalize_ws(md_text)
        v = (verdict or "").upper().strip()

        if not weak_mask or v == "Y":
            return t

        token = mask_n_token if v == "N" else mask_u_token  # U/None -> 无法识别
        if qno is None:
            qno = extract_qno_from_text_block(t)

        return f"{qno}. {token}" if qno is not None else token


    def _confirm_mask_before_apply(
        *,
        crop_img,
        text: str,
        target: str,
        cache: Optional[JsonCache],
        max_chars: int,
    ) -> Dict[str, Any]:
        try:
            from PIL import Image
        except Exception:
            Image = None  # type: ignore

        t = normalize_ws(text or "")
        tgt = _normalize_verdict(target)

        probe = ""
        for ln in t.splitlines():
            ln2 = ln.strip()
            if ln2:
                probe = ln2[:120]
                break
        if not probe:
            probe = t[:120]

        dbg: Dict[str, Any] = {"target": tgt, "probe": probe}

        if crop_img is not None and probe:
            try:
                ev = verify_evidence(ver_vlm, crop_img, probe, cache=cache)
                dbg["evi_probe"] = ev.verdict
                if _normalize_verdict(ev.verdict) == "Y":
                    dbg["mask"] = False
                    dbg["reason"] = "probe_found_keep"
                    return dbg
            except Exception as e:
                dbg["evi_probe_err"] = str(e)

        img_variants = []
        if crop_img is not None:
            img_variants.append(("orig", crop_img))
            try:
                from .img_utils import enhance_for_vlm, STRONG_ENHANCE
                enh = enhance_for_vlm(crop_img, STRONG_ENHANCE)
                img_variants.append(("enh", enh))
            except Exception as e:
                dbg["enh_err"] = str(e)

            try:
                if Image is not None:
                    w, h = crop_img.size
                    if min(w, h) < 220:
                        up = crop_img.resize((int(w * 1.8), int(h * 1.8)), Image.BICUBIC)
                        img_variants.append(("up", up))
            except Exception as e:
                dbg["up_err"] = str(e)

        votes = []
        cand = t[:max_chars]

        for tag, im in img_variants[:3]:
            try:
                vv = verify_question_lenient(ver_vlm, im, cand, cache=cache).verdict
                vv2 = _normalize_verdict(vv)
                votes.append(vv2)
                dbg.setdefault("votes", []).append({"img": tag, "v": vv})
            except Exception as e:
                dbg.setdefault("vote_errs", []).append({"img": tag, "err": str(e)})

        nY = sum(1 for v in votes if v == "Y")
        nN = sum(1 for v in votes if v == "N")
        nU = sum(1 for v in votes if v == "U")
        dbg["vote_counts"] = {"Y": nY, "N": nN, "U": nU}

        if nY > 0:
            dbg["mask"] = False
            dbg["reason"] = "has_Y_keep"
            return dbg

        if tgt == "N":
            dbg["mask"] = bool(nN >= 2)
            dbg["reason"] = "N_confirm" if dbg["mask"] else "N_not_confirmed"
            return dbg

        if tgt == "U":
            dbg["mask"] = bool((nU >= 2) or (nN >= 2))
            dbg["reason"] = "U_confirm" if dbg["mask"] else "U_not_confirmed"
            return dbg

        dbg["mask"] = False
        dbg["reason"] = "unknown_target_keep"
        return dbg

    def _rel_md_img(fp: Path) -> str:
        try:
            rel = fp.relative_to(page_dir).as_posix()
        except Exception:
            rel = fp.as_posix()
        return f"![]({rel})"


    FIXED_WHITELIST = {
        "tail_prune",
        "drop_spurious_line",
        "drop_spurious_option",
        "dedup_repeat_span",
        "remove_spurious_figure_ref",
    }
    def _fixed_actions_from_repair_dbg(dbg: Any) -> List[Dict[str, Any]]:
        if not isinstance(dbg, dict):
            return []
        acts = dbg.get("fixed_actions")
        if isinstance(acts, list):
            return [a for a in acts if isinstance(a, dict) and a.get("action")]
        tp = dbg.get("tail_prune")
        if isinstance(tp, dict) and tp.get("did_prune") and tp.get("tail_evidence") == "N":
            return [{"action": "tail_prune", "removed_tail": tp.get("tail")}]
        return []

    def _combine_verdict(vs: Optional[str], vl: Optional[str]) -> str:
        sset = {vs, vl}
        sset = {x for x in sset if isinstance(x, str) and x.strip()}
        sset = {x.strip().upper() for x in sset}
        if "Y" in sset:
            return "Y"
        if ("N" in sset) and ("U" in sset):
            return "U"
        if sset == {"N"}:
            return "N"
        return "U"

    def _inject_fixed_marker(text: str, actions: Optional[List[Dict[str, Any]]] = None) -> str:

        t = normalize_ws(text)

        if fixed_token in t:
            return t
        names: List[str] = []
        for a in (actions or []):
            if isinstance(a, dict):
                nm = a.get("action")
            else:
                nm = str(a)
            if nm:
                names.append(str(nm))

        uniq: List[str] = []

        for nm in names:
            if nm not in uniq:
                uniq.append(nm)

        suffix = fixed_token if not uniq else f"{fixed_token} actions={','.join(uniq)}"

        if not t.strip():
            return suffix
        if fixed_marker_mode == "append_line":
            return t + "\n\n" + suffix
        lines = t.splitlines()
        if not lines:
            return suffix
        lines[0] = lines[0].rstrip() + " " + suffix
        return "\n".join(lines)
    _OPT_HEAD_RE = re.compile(r"^\s*[\(（]?\s*([A-D])\s*[\)）]?\s*[\.．、:：]\s*(.*)\s*$")

    def _split_option_blocks_simple(lines: List[str]):
        pre: List[str] = []
        opts: Dict[str, List[str]] = {}
        post: List[str] = []
        cur_L: Optional[str] = None
        cur_lines: List[str] = []
        seen_any = False
        in_post = False

        ANS_HEAD_RE = re.compile(r"^\s*(答案|解析|解[:：]|解答|【答案】|【解析】)")

        def flush():
            nonlocal cur_L, cur_lines
            if cur_L and cur_lines:
                opts[cur_L] = cur_lines[:]
            cur_L, cur_lines = None, []

        for ln in lines:
            if in_post:
                post.append(ln)
                continue

            if seen_any and ANS_HEAD_RE.search(ln or ""):
                flush()
                in_post = True
                post.append(ln)
                continue

            m = _OPT_HEAD_RE.match(ln)
            if m:
                seen_any = True
                flush()
                cur_L = m.group(1)
                cur_lines = [ln]
            else:
                if not seen_any:
                    pre.append(ln)
                else:
                    cur_lines.append(ln)

        if seen_any and (not in_post):
            flush()
        if not seen_any:
            pre = lines[:]

        return pre, opts, post

    def _calc_del_ratio(old_text: str, new_text: str) -> float:
        o = normalize_ws(old_text)
        n = normalize_ws(new_text)
        if not o:
            return 0.0
        return max(0.0, (len(o) - len(n))) / max(1.0, len(o))

    def _fixed_subtractive_attempt(
        crop_img,
        old_text: str,
        *,
        qno: Optional[int],
        weak_reasons_local: List[str],
    ):
        fixed_dbg: Dict[str, Any] = {
            "fixed_applied": False,
            "actions": [],
            "verdict_before_raw": None,
            "verdict_before_strong": None,
            "verdict_after_raw": None,
            "del_ratio": None,
        }

        old_norm = normalize_ws(old_text)
        if not old_norm.strip():
            return None, fixed_dbg

        vbs = verify_question_strict(ver_vlm, crop_img, old_norm, cache=cache, max_chars=max_qchars)
        vbl = None
        if vbs.verdict == "N":
            vbl = verify_question_lenient(ver_vlm, crop_img, old_norm, cache=cache, max_chars=max_qchars)
        verdict_before_raw = _combine_verdict(getattr(vbs, "verdict", None), getattr(vbl, "verdict", None) if vbl else None)
        verdict_before_strong = _safe_weak_verdict(verdict_before_raw, weak_reasons_local)

        fixed_dbg["verdict_before_raw"] = verdict_before_raw
        fixed_dbg["verdict_before_strong"] = verdict_before_strong

        if verdict_before_strong != "N":
            return None, fixed_dbg

        t = old_norm
        actions: List[Dict[str, Any]] = []

        fig_ref_pat = re.compile(r"(如图所示|如图|见图|见下图|如下图|图\s*\d+|图[一二三四五六七八九十]+)")
        for m in list(fig_ref_pat.finditer(t)):
            span = m.group(1)
            if not span or len(span) > 10:
                continue
            ev = verify_evidence(ver_vlm, crop_img, span, cache=cache)
            if ev.verdict == "N":
                t2 = t.replace(span, "")
                if t2 != t:
                    actions.append({"action": "remove_spurious_figure_ref", "removed": span})
                    t = normalize_ws(t2)

        lines = t.splitlines()
        did_dedup = False
        for k in range(1, min(6, len(lines) // 2) + 1):
            if lines[-k:] == lines[-2 * k:-k]:
                removed = lines[-k:]
                lines = lines[:-k]
                actions.append({"action": "dedup_repeat_span", "k_lines": k, "removed_lines": removed})
                t = normalize_ws("\n".join(lines))
                did_dedup = True
                break
        _ = did_dedup  

        has_opt = verify_has_options(ver_vlm, crop_img, cache=cache)
        fixed_dbg["has_options_probe"] = getattr(has_opt, "verdict", None)

        if getattr(has_opt, "verdict", None) == "Y":
            t_lines = t.splitlines()
            pre, opts, post = _split_option_blocks_simple(t_lines)
            if opts:
                removed_opts = []
                kept_lines = pre[:]
                for L, block_lines in opts.items():
                    blk = normalize_ws("\n".join(block_lines))
                    ev = verify_evidence(ver_vlm, crop_img, blk, cache=cache)
                    if ev.verdict == "N":
                        removed_opts.append({"letter": L, "text": blk})
                    else:
                        kept_lines.extend(block_lines)
                if removed_opts:
                    orig_cnt = len(opts)
                    kept_cnt = orig_cnt - len(removed_opts)
                    if (orig_cnt >= int(fixed_min_keep_options)) and (kept_cnt < int(fixed_min_keep_options)):
                        fixed_dbg.setdefault("notes", []).append(
                            f"drop_spurious_option_skipped_keep<{int(fixed_min_keep_options)}"
                        )
                    else:
                        actions.append({"action": "drop_spurious_option", "removed": removed_opts})
                        t = normalize_ws("\n".join(kept_lines + post))

        t_lines = t.splitlines()
        for _ in range(3):
            if not t_lines:
                break
            last = t_lines[-1].strip()
            if not last:
                t_lines = t_lines[:-1]
                continue
            if last.startswith("![]("):
                break
            if len(last) > 60:
                break
            ev = verify_evidence(ver_vlm, crop_img, last, cache=cache)
            if ev.verdict == "N":
                actions.append({"action": "drop_spurious_line", "removed_line": last})
                t_lines = t_lines[:-1]
                t = normalize_ws("\n".join(t_lines))
                continue
            break

        tail_pat = re.compile(r"([_]{4,}[^\n]{0,80})\s*$|([\.]{3,}[^\n]{0,80})\s*$")
        mt = tail_pat.search(t)
        if mt:
            tail = mt.group(1) or mt.group(2)
            if tail:
                ev = verify_evidence(ver_vlm, crop_img, tail.strip(), cache=cache)
                if ev.verdict == "N":
                    t2 = t[:mt.start()].rstrip()
                    if t2 != t:
                        actions.append({"action": "tail_prune", "removed_tail": tail.strip()})
                        t = normalize_ws(t2)

        if normalize_ws(t) == old_norm:
            fixed_dbg["actions"] = actions
            return None, fixed_dbg

        vas = verify_question_strict(ver_vlm, crop_img, t, cache=cache, max_chars=max_qchars)
        val = None
        if vas.verdict == "N":
            val = verify_question_lenient(ver_vlm, crop_img, t, cache=cache, max_chars=max_qchars)
        verdict_after_raw = _combine_verdict(getattr(vas, "verdict", None), getattr(val, "verdict", None) if val else None)
        fixed_dbg["verdict_after_raw"] = verdict_after_raw

        if verdict_after_raw == "N":
            fixed_dbg["actions"] = actions
            return None, fixed_dbg
        if (not fixed_after_allow_u) and (verdict_after_raw != "Y"):
            fixed_dbg["actions"] = actions
            return None, fixed_dbg

        del_ratio = _calc_del_ratio(old_norm, t)
        fixed_dbg["del_ratio"] = del_ratio
        if del_ratio > float(fixed_max_del_ratio):
            fixed_dbg["actions"] = actions
            fixed_dbg["reject_reason"] = f"del_ratio>{fixed_max_del_ratio}"
            return None, fixed_dbg

        # All actions must be whitelisted
        for a in actions:
            if a.get("action") not in FIXED_WHITELIST:
                fixed_dbg["actions"] = actions
                fixed_dbg["reject_reason"] = "non_whitelist_action"
                return None, fixed_dbg

        fixed_dbg["fixed_applied"] = True
        fixed_dbg["actions"] = actions
        return t, fixed_dbg

    used_unknown = 0
    used_merge_suspect = 0
    first_mismatch_qno: Optional[int] = None

    out_blocks: List[str] = []
    r_cfg = RepairConfig(
        max_qchars=max_qchars,
        mask_u_token=mask_u_token,
        mask_n_token=mask_n_token,
    )

    if fig_cfg is None:
        fig_cfg = FigureFilterCfg()

    have_md_qno = any((bb.kind == "q") and (bb.qno is not None) for bb in blocks)
    crop_qno_expected = bool(use_crop_qno and (qno_vlm is not None) and have_md_qno)

    align_dbg = align_res.debug if isinstance(getattr(align_res, "debug", None), dict) else {}
    winner_score = align_dbg.get("winner_score")

    tail_run = 0
    uncompared = 0
    mismatches = 0
    unknown_used = 0
    eq_ratio = None

    if isinstance(winner_score, (list, tuple)) and len(winner_score) >= 7:
        tail_run = int(winner_score[0])
        uncompared = int(winner_score[1])
        mismatches = int(winner_score[3])
        unknown_used = int(winner_score[4])
        try:
            eq = int(-winner_score[5])
            comparable = eq + mismatches
            eq_ratio = (eq / comparable) if comparable > 0 else None
        except Exception:
            eq_ratio = None

    page_first_mismatch_qno = _first_mismatch_from_pairs(pairs)
    no_crop_qno_should_be_weak = bool(
        (page_first_mismatch_qno is not None) or (tail_run > 0) or (unknown_used > 0)
    )

    align_trust_high = bool(
        (eq_ratio is not None)
        and (eq_ratio >= 0.75)
        and (tail_run == 0)
        and (uncompared <= 1)
        and (mismatches == 0)
        and (unknown_used == 0)
    )
    for bi, b in enumerate(blocks):
        if b.kind != "q":
            out_blocks.append(b.text)
            report["items"].append({"block": bi, "kind": b.kind, "kept": True})
            continue

        cand_pos = align_res.mapping.get(bi)
        q_idx = None
        if cand_pos is not None and 0 <= cand_pos < len(align_res.candidates):
            q_idx = align_res.candidates[cand_pos]

        if q_idx is None:
            if mask_no_crop:
                kept = f"{b.qno}. {mask_u_token}" if b.qno is not None else mask_u_token
                out_blocks.append(kept)
                status = "no_crop_mask_u"
            else:
                out_blocks.append(b.text)
                status = "no_crop_keep_md"

            report["items"].append({
                "block": bi,
                "kind": "q",
                "qno": b.qno,
                "md_len": len(b.text),
                "status": status,
                "mapped_match_qi": None,
                "weak_crop": True,
                "weak_reasons": ["no_crop"],
            })
            continue

        pair = pair_by_block.get(int(bi), {}) if pair_by_block else {}
        md_qno = b.qno
        crop_qno = pair.get("crop_qno")
        cp = pair.get("crop_path")

        if "is_unknown" in pair:
            is_unknown = bool(pair.get("is_unknown"))
        else:
            is_unknown = bool(cp and "q999999" in str(cp))

        if "merge_suspect" in pair:
            merge_suspect = bool(pair.get("merge_suspect"))
        else:
            merge_suspect = _merge_suspect_qidx(q_idx)

        if is_unknown:
            used_unknown += 1
        if merge_suspect:
            used_merge_suspect += 1

        mismatch = False
        mismatch_soft = False

        if (md_qno is not None) and (crop_qno is not None):
            try:
                mismatch = (int(md_qno) != int(crop_qno))
            except Exception:
                mismatch = False

        if mismatch:
            if first_mismatch_qno is None:
                try:
                    first_mismatch_qno = int(md_qno)
                except Exception:
                    first_mismatch_qno = None

            if merge_suspect or is_unknown or align_trust_high:
                mismatch_soft = True
            else:
                out_blocks.append(b.text)
                report["items"].append(
                    {
                        "block": bi,
                        "kind": "q",
                        "qno": md_qno,
                        "status": "align_mismatch_keep_md",
                        "mapped_match_qi": int(q_idx),
                        "crop_qno": int(crop_qno) if crop_qno is not None else None,
                        "crop_path": str(cp) if cp else None,
                        "is_unknown": bool(is_unknown),
                        "merge_suspect": bool(merge_suspect),
                    }
                )
                continue

        no_crop_qno_weak = bool(
            crop_qno_expected
            and weak_when_no_crop_qno
            and (crop_qno is None)
            and no_crop_qno_should_be_weak
        )
        mark_only = bool(mismatch_soft)

        weak_crop = bool(is_unknown or merge_suspect or mismatch_soft or no_crop_qno_weak)

        weak_reasons: List[str] = []
        if is_unknown:
            weak_reasons.append("unknown")
        if merge_suspect:
            weak_reasons.append("merge_suspect")
        if mismatch_soft:
            weak_reasons.append("qno_mismatch_soft")
        if no_crop_qno_weak:
            weak_reasons.append("no_crop_qno")
        if mark_only:
            weak_reasons.append("mark_only")

        if mark_only:
            qrec = qs_sorted[q_idx]
            match_class = pair.get("match_class") or qrec.get("class_name")

            crop_img = load_crop_image(page_dir, qrec)

            recrop_dbg = None
            try:
                if crop_img is None:
                    need_recrop = True
                else:
                    _, h0 = crop_img.size
                    need_recrop = (_is_fill_blank(match_class) and (h0 < FILL_RECROP_TRIGGER_H))
                if need_recrop:
                    recrop_img, recrop_dbg = _recrop_fill_blank_from_page(qrec, match_class)
                    if recrop_img is not None:
                        crop_img = recrop_img
            except Exception:
                recrop_dbg = None

            if crop_img is None:
                recrop2_img, recrop2_dbg = _recrop_generic_from_page(qrec)
                if recrop2_img is not None:
                    crop_img = recrop2_img
                if recrop2_dbg:
                    if recrop_dbg:
                        recrop_dbg = {"fill_blank": recrop_dbg, "generic": recrop2_dbg}
                    else:
                        recrop_dbg = {"generic": recrop2_dbg}

            if crop_img is None:
                if mask_no_crop:
                    kept = f"{b.qno}. {mask_u_token}" if b.qno is not None else mask_u_token
                    out_blocks.append(kept)
                    status = "crop_open_fail_mask_u"
                else:
                    out_blocks.append(b.text)
                    status = "crop_open_fail_keep_md"

                report["items"].append({
                    "block": bi,
                    "kind": "q",
                    "qno": md_qno,
                    "status": status,
                    "mapped_match_qi": int(q_idx),
                    "crop_path": str(cp) if cp else None,
                    "crop_qno": crop_qno,
                    "match_class": match_class,
                    "recrop": recrop_dbg,
                    "weak_reasons": weak_reasons,
                })
                continue



            tiny_crop = _is_tiny_crop(crop_img, match_class)

            if tiny_crop:

                recrop2_img, recrop2_dbg = _recrop_generic_from_page(qrec, pad_y_ratio=0.14, min_pad_y=60)
                if recrop2_img is not None:
                    tiny2 = _is_tiny_crop(recrop2_img, match_class)
                    if not tiny2:
                        crop_img = recrop2_img
                        tiny_crop = False
                    if recrop2_dbg:
                        if recrop_dbg:
                            recrop_dbg = {"fill_blank": recrop_dbg, "generic": recrop2_dbg}
                        else:
                            recrop_dbg = {"generic": recrop2_dbg}
            if crop_img is not None:
                w, h = crop_img.size
            else:
                w, h = None, None

            if recrop_dbg is not None:
                if isinstance(recrop_dbg, dict):
                    if ("fill_blank" in recrop_dbg) or bool(recrop_dbg.get("recrop_fill_blank")):
                        weak_reasons.append("recrop_fill_blank")
                    if ("generic" in recrop_dbg) or str(recrop_dbg.get("recrop", "")).startswith("generic_"):
                        weak_reasons.append("recrop_generic")
                    if not any(r.startswith("recrop_") for r in weak_reasons):
                        weak_reasons.append("recrop")
                else:
                    weak_reasons.append("recrop")

            if tiny_crop:
                weak_reasons.append(f"tiny_crop(h={h},w={w})")

            weak_crop2 = bool(weak_crop or tiny_crop)
            if is_unknown or tiny_crop:
                img_lines: List[str] = []
                figs_kept: List[Path] = []
                figs_dbg: List[Any] = []

                allow_fig = bool((not mismatch_soft) and (((not merge_suspect) or _merge_suspect_override_ok(qrec))) and _qno_ok(md_qno, crop_qno, align_trust_high) and ((not weak_crop2) or weak_keep_figures))
                if fig_vlm is not None and allow_fig:
                    fig_paths = find_figure_candidates(page_dir, qrec)
                    if fig_paths:
                        figs_kept, figs_dbg = select_figures_for_question(
                            crop_img, fig_paths, fig_vlm, cache=cache, cfg=fig_cfg
                        )
                        img_lines = [_rel_md_img(fp) for fp in (figs_kept or [])]

                kept_text = normalize_ws(b.text)
                if img_lines:
                    kept_text = kept_text + "\n\n" + "\n".join(img_lines)

                out_blocks.append(kept_text)
                report["items"].append(
                    {
                        "block": bi,
                        "kind": "q",
                        "qno": md_qno,
                        "status": "mark_only_unreliable_crop_keep_md",
                        "mapped_match_qi": int(q_idx),
                        "crop_path": str(cp) if cp else None,
                        "crop_qno": crop_qno,
                        "is_unknown": bool(is_unknown),
                        "merge_suspect": bool(merge_suspect),
                        "tiny_crop": bool(tiny_crop),
                        "weak_crop": True,
                        "weak_reasons": weak_reasons,
                        "mark_only": True,
                        "figures": figs_dbg,
                        "match_class": match_class,
                        "recrop": recrop_dbg,
                        "figures_kept": [x.as_posix() for x in figs_kept] if figs_kept else [],
                    }
                )
                continue

            v1 = verify_question_strict(ver_vlm, crop_img, b.text, cache=cache, max_chars=max_qchars)
            v2 = verify_question_lenient(ver_vlm, crop_img, b.text, cache=cache, max_chars=max_qchars)

            v_strict0 = _normalize_verdict(getattr(v1, "verdict", None))
            v_lenient0 = _normalize_verdict(getattr(v2, "verdict", None))

            s = {v_strict0, v_lenient0}
            s.discard("") 

            if "Y" in s:
                v_agg = "Y"
            elif ("N" in s) and ("U" in s):
                v_agg = "U"
            elif s == {"N"}:
                v_agg = "N"
            else:
                v_agg = "U"

            v_mask_raw = v_agg
            if (
                weak_crop2
                and weak_mask_on_weak
                and (md_qno not in known_crop_qnos)
                and (v_lenient0 == "N")
                and (v_strict0 != "Y")
            ):
                v_mask_raw = "N"
                if "lenient_N_missing_qno" not in weak_reasons:
                    weak_reasons.append("lenient_N_missing_qno")

            v_after = _safe_weak_verdict(v_mask_raw, weak_reasons)

            will_mask = bool(
                weak_crop2
                and weak_mask_on_weak
                and (v_mask_raw == "N" or (weak_mask_on_u and v_mask_raw == "U"))
            )

            mask_confirm = None
            if will_mask:
                try:
                    mask_confirm = _confirm_mask_before_apply(
                        crop_img=crop_img,
                        text=normalize_ws(b.text),
                        target=v_mask_raw,
                        cache=cache,
                        max_chars=max_qchars,
                    )
                    if not mask_confirm.get("mask", False):
                        will_mask = False
                except Exception as e:
                    mask_confirm = {"mask": False, "err": str(e)}

            if will_mask:
                kept_text = _weak_output(b.text, v_mask_raw, weak_mask=True, qno=b.qno)
                out_blocks.append(kept_text)
                report["items"].append(
                    {
                        "block": bi,
                        "kind": "q",
                        "qno": md_qno,
                        "status": "mark_only_mask_no_fig",
                        "mapped_match_qi": int(q_idx),
                        "crop_path": str(cp) if cp else None,
                        "crop_qno": crop_qno,
                        "is_unknown": bool(is_unknown),
                        "merge_suspect": bool(merge_suspect),
                        "weak_crop": True,
                        "weak_reasons": weak_reasons,
                        "no_crop_qno_weak": bool(no_crop_qno_weak),
                        "mismatch_soft": bool(mismatch_soft),
                        "mark_only": True,
                        "v_strict": v_strict0,
                        "v_lenient": v_lenient0,
                        "v_after_repair": v_after,
                        "v_after_raw": v_mask_raw,
                        "v_agg": v_agg,
                        "mask_confirm": mask_confirm,
                    }
                )
                continue
                out_blocks.append(kept_text)
                report["items"].append(
                    {
                        "block": bi,
                        "kind": "q",
                        "qno": md_qno,
                        "status": "mark_only_mask_no_fig",
                        "mapped_match_qi": int(q_idx),
                        "crop_path": str(cp) if cp else None,
                        "crop_qno": crop_qno,
                        "is_unknown": bool(is_unknown),
                        "merge_suspect": bool(merge_suspect),
                        "weak_crop": True,
                        "weak_reasons": weak_reasons,
                        "no_crop_qno_weak": bool(no_crop_qno_weak),
                        "mismatch_soft": bool(mismatch_soft),
                        "mark_only": True,
                        "v_strict": getattr(v1, "verdict", None),
                        "v_lenient": getattr(v2, "verdict", None),
                        "v_after_repair": v_after,
                        "v_after_raw": v_raw,
                    "mask_confirm": mask_confirm,
                    }
                )
                continue

            kept_text = normalize_ws(b.text)

            img_lines: List[str] = []
            figs_kept: List[Path] = []
            figs_dbg: List[Any] = []

            allow_fig = bool((not mismatch_soft) and (((not merge_suspect) or _merge_suspect_override_ok(qrec))) and _qno_ok(md_qno, crop_qno, align_trust_high) and ((not weak_crop2) or weak_keep_figures))
            if fig_vlm is not None and allow_fig:
                fig_paths = find_figure_candidates(page_dir, qrec)
                if fig_paths:
                    figs_kept, figs_dbg = select_figures_for_question(
                        crop_img, fig_paths, fig_vlm, cache=cache, cfg=fig_cfg
                    )
                    img_lines = [_rel_md_img(fp) for fp in (figs_kept or [])]

            if img_lines:
                kept_text = normalize_ws(kept_text) + "\n\n" + "\n".join(img_lines)

            out_blocks.append(kept_text)
            report["items"].append(
                {
                    "block": bi,
                    "kind": "q",
                    "qno": md_qno,
                    "status": "mark_only_keep",
                    "mapped_match_qi": int(q_idx),
                    "crop_path": str(cp) if cp else None,
                    "crop_qno": crop_qno,
                    "is_unknown": bool(is_unknown),
                    "merge_suspect": bool(merge_suspect),
                    "weak_crop": True,
                    "weak_reasons": weak_reasons,
                    "no_crop_qno_weak": bool(no_crop_qno_weak),
                    "mismatch_soft": bool(mismatch_soft),
                    "mark_only": True,
                    "v_strict": getattr(v1, "verdict", None),
                    "v_lenient": getattr(v2, "verdict", None),
                    "v_after_repair": v_after,
                    "figures": figs_dbg,
                    "figures_kept": [x.as_posix() for x in figs_kept] if figs_kept else [],
                }
            )
            continue

        qrec = qs_sorted[q_idx]
        match_class = pair.get("match_class") or qrec.get("class_name")
        
        crop_img = load_crop_image(page_dir, qrec)

        recrop_dbg = None
        try:
            if crop_img is None:
                need_recrop = True
            else:
                _, h0 = crop_img.size
                need_recrop = (_is_fill_blank(match_class) and (h0 < FILL_RECROP_TRIGGER_H))
            if need_recrop:
                recrop_img, recrop_dbg = _recrop_fill_blank_from_page(qrec, match_class)
                if recrop_img is not None:
                    crop_img = recrop_img
        except Exception:
            recrop_dbg = None

        if crop_img is None:
            if mask_no_crop:
                kept = f"{b.qno}. {mask_u_token}" if b.qno is not None else mask_u_token
                out_blocks.append(kept)
                status = "crop_open_fail_mask_u"
            else:
                out_blocks.append(b.text)
                status = "crop_open_fail_keep_md"

            report["items"].append({
                "block": bi,
                "kind": "q",
                "qno": md_qno,
                "status": status,
                "mapped_match_qi": int(q_idx),
                "crop_path": str(cp) if cp else None,
                "crop_qno": crop_qno,
                "match_class": match_class,
                "recrop": recrop_dbg,
                "weak_reasons": weak_reasons,
            })
            continue

        tiny_crop = _is_tiny_crop(crop_img, match_class)

        if tiny_crop:


            recrop2_img, recrop2_dbg = _recrop_generic_from_page(qrec, pad_y_ratio=0.14, min_pad_y=60)
            if recrop2_img is not None:
                tiny2 = _is_tiny_crop(recrop2_img, match_class)
                if not tiny2:
                    crop_img = recrop2_img
                    tiny_crop = False
                if recrop2_dbg:
                    if recrop_dbg:
                        recrop_dbg = {"fill_blank": recrop_dbg, "generic": recrop2_dbg}
                    else:
                        recrop_dbg = {"generic": recrop2_dbg}
        if crop_img is not None:
            w, h = crop_img.size
        else:
            w, h = None, None

        if recrop_dbg is not None:
            if isinstance(recrop_dbg, dict):
                if ("fill_blank" in recrop_dbg) or bool(recrop_dbg.get("recrop_fill_blank")):
                    weak_reasons.append("recrop_fill_blank")
                if ("generic" in recrop_dbg) or str(recrop_dbg.get("recrop", "")).startswith("generic_"):
                    weak_reasons.append("recrop_generic")
                if not any(r.startswith("recrop_") for r in weak_reasons):
                    weak_reasons.append("recrop")
            else:
                weak_reasons.append("recrop")

        if tiny_crop:
            weak_reasons.append(f"tiny_crop(h={h},w={w})")

        weak_crop2 = bool(weak_crop or tiny_crop)

        if is_unknown or tiny_crop:
            img_lines: List[str] = []
            figs_kept: List[Path] = []
            figs_dbg: List[Any] = []

            allow_fig = bool((not mismatch_soft) and (((not merge_suspect) or _merge_suspect_override_ok(qrec))) and _qno_ok(md_qno, crop_qno, align_trust_high) and ((not weak_crop2) or weak_keep_figures))
            if fig_vlm is not None and allow_fig:
                fig_paths = find_figure_candidates(page_dir, qrec)
                if fig_paths:
                    figs_kept, figs_dbg = select_figures_for_question(
                        crop_img, fig_paths, fig_vlm, cache=cache, cfg=fig_cfg
                    )
                    img_lines = [_rel_md_img(fp) for fp in (figs_kept or [])]

            kept_text = normalize_ws(b.text)
            if img_lines:
                kept_text = kept_text + "\n\n" + "\n".join(img_lines)

            out_blocks.append(kept_text)
            report["items"].append(
                {
                    "block": bi,
                    "kind": "q",
                    "qno": md_qno,
                    "status": "unreliable_crop_keep_md",
                    "mapped_match_qi": int(q_idx),
                    "crop_path": str(cp) if cp else None,
                    "crop_qno": crop_qno,
                    "is_unknown": bool(is_unknown),
                    "merge_suspect": bool(merge_suspect),
                    "tiny_crop": bool(tiny_crop),
                    "weak_crop": True,
                    "weak_reasons": weak_reasons,
                    "figures": figs_dbg,
                    "match_class": match_class,
                    "recrop": recrop_dbg,

                    "figures_kept": [x.as_posix() for x in figs_kept] if figs_kept else [],
                }
            )
            continue

        if ablation_no_patcher:
            v_strict = verify_question_strict(ver_vlm, crop_img, b.text, cache=cache, max_chars=max_qchars)
            v_lenient = None
            if getattr(v_strict, "verdict", None) == "N":
                v_lenient = verify_question_lenient(ver_vlm, crop_img, b.text, cache=cache, max_chars=max_qchars)

            v_raw = _combine_verdict(getattr(v_strict, "verdict", None), getattr(v_lenient, "verdict", None) if v_lenient else None)
            v_after = _safe_weak_verdict(v_raw, weak_reasons)

            will_mask = bool((not weak_crop2) and (v_raw != "Y")) or bool(weak_crop2 and weak_mask_on_weak and (v_raw == "N" or (weak_mask_on_u and v_raw == "U")))
            mask_confirm = None
            if weak_crop2 and will_mask:
                try:
                    mask_confirm = _confirm_mask_before_apply(
                        crop_img=crop_img,
                        text=normalize_ws(b.text),
                        target=v_raw,
                        cache=cache,
                        max_chars=max_qchars,
                    )
                    if not mask_confirm.get("mask", False):
                        will_mask = False
                        v_raw = "Y"
                        v_after = "Y"
                except Exception as e:
                    mask_confirm = {"mask": False, "err": str(e)}
                    will_mask = False

            if will_mask:
                out_text = _weak_output(b.text, v_raw, weak_mask=True, qno=b.qno)
                action = "mask_n" if v_raw == "N" else "mask_u"
            else:
                out_text = normalize_ws(b.text)
                action = "keep"
                if weak_crop2 and (v_after != "Y") and (not weak_mask_on_weak):
                    action = "keep_md_on_weak"

            img_lines: List[str] = []
            figs_kept: List[Path] = []
            figs_dbg: List[Any] = []

            if action.startswith("keep"):
                allow_fig = bool((not mismatch_soft) and (((not merge_suspect) or _merge_suspect_override_ok(qrec))) and _qno_ok(md_qno, crop_qno, align_trust_high) and ((not weak_crop2) or weak_keep_figures))
                if fig_vlm is not None and allow_fig:
                    fig_paths = find_figure_candidates(page_dir, qrec)
                    if fig_paths:
                        figs_kept, figs_dbg = select_figures_for_question(
                            crop_img, fig_paths, fig_vlm, cache=cache, cfg=fig_cfg
                        )
                        if figs_kept:
                            img_lines = [_rel_md_img(fp) for fp in figs_kept]
                if img_lines:
                    out_text = normalize_ws(out_text) + "\n\n" + "\n".join(img_lines)

            if verdict_comment:
                out_text = normalize_ws(out_text) + f"\n\n<!-- verdict={v_raw} action={action} -->"

            out_blocks.append(out_text)
            report["items"].append(
                {
                    "block": bi,
                    "kind": "q",
                    "qno": md_qno,
                    "status": "ablation_no_patcher",
                    "action": action,
                    "mapped_match_qi": int(q_idx),
                    "crop_path": str(cp) if cp else None,
                    "crop_qno": crop_qno,
                    "match_class": match_class,
                    "weak_crop": bool(weak_crop2),
                    "weak_reasons": weak_reasons,
                    "v_strict": getattr(v_strict, "verdict", None),
                    "v_lenient": getattr(v_lenient, "verdict", None) if v_lenient else None,
                    "v_after_raw": v_raw,
                    "v_after": v_after,
                    "mask_confirm": mask_confirm,
                    "figures": figs_dbg,
                    "figures_kept": [x.as_posix() for x in figs_kept] if figs_kept else [],
                    "recrop": recrop_dbg,
                    "mismatch_soft": bool(mismatch_soft),
                    "mark_only": bool(mark_only),
                }
            )
            continue

        if fixed_enable:
            fixed_text, fixed_dbg = _fixed_subtractive_attempt(
                crop_img,
                b.text,
                qno=b.qno,
                weak_reasons_local=weak_reasons,
            )
            if fixed_text is not None and isinstance(fixed_dbg, dict) and fixed_dbg.get("fixed_applied"):
                # In weak-crop mode, only accept FIXED if the post-verdict is strong Y; otherwise it would be masked.
                v_after_fixed = fixed_dbg.get("verdict_after_raw") or "U"
                if not (weak_crop2 and weak_mask_on_weak and (v_after_fixed != "Y")):
                    out_text = fixed_text

                    img_lines: List[str] = []
                    figs_kept: List[Path] = []
                    figs_dbg: List[Any] = []

                    allow_fig = bool((not mismatch_soft) and (((not merge_suspect) or _merge_suspect_override_ok(qrec))) and _qno_ok(md_qno, crop_qno, align_trust_high) and ((not weak_crop2) or weak_keep_figures))
                    if fig_vlm is not None and allow_fig:
                        fig_paths = find_figure_candidates(page_dir, qrec)
                        if fig_paths:
                            figs_kept, figs_dbg = select_figures_for_question(
                                crop_img, fig_paths, fig_vlm, cache=cache, cfg=fig_cfg
                            )
                            img_lines = [_rel_md_img(fp) for fp in (figs_kept or [])]

                    if img_lines:
                        out_text = normalize_ws(out_text) + "\n\n" + "\n".join(img_lines)

                    # FIXED marker should be appended AFTER figure lines are appended.
                    out_text = _inject_fixed_marker(
                        out_text,
                        (fixed_dbg or {}).get("actions") if isinstance(fixed_dbg, dict) else None,
                    )

                    out_blocks.append(out_text)
                    report["items"].append(
                        {
                            "block": bi,
                            "kind": "q",
                            "qno": md_qno,
                            "status": "fixed_subtractive",
                            "mapped_match_qi": int(q_idx),
                            "crop_path": str(cp) if cp else None,
                            "crop_qno": crop_qno,
                            "is_unknown": bool(is_unknown),
                            "merge_suspect": bool(merge_suspect),
                            "weak_crop": bool(weak_crop2),
                            "no_crop_qno_weak": bool(no_crop_qno_weak),
                            "fixed": fixed_dbg,
                            "figures": figs_dbg,
                            "figures_kept": [x.as_posix() for x in figs_kept] if figs_kept else [],
                        }
                    )
                    continue
        final_text, dbg = repair_question_v6_6(
            ver_vlm, gen_vlm, crop_img, b.text, cache=cache, cfg=r_cfg
        )

        v_raw = _extract_final_verdict(dbg) or "U"
        v_after = _safe_weak_verdict(v_raw, weak_reasons)
        will_mask = bool(weak_crop2 and weak_mask_on_weak and (v_raw == "N" or (weak_mask_on_u and v_raw == "U")))
        mask_confirm = None

        if will_mask:

            try:
                mask_confirm = _confirm_mask_before_apply(
                    crop_img=crop_img,
                    text=normalize_ws(b.text),
                    target=v_raw,   
                    cache=cache,
                    max_chars=max_qchars,
                )

                if not mask_confirm.get("mask", False):

                    will_mask = False

            except Exception as e:
                mask_confirm = {"mask": False, "err": str(e)}
        if will_mask:
            kept_text = _weak_output(b.text, v_raw, weak_mask=True, qno=b.qno)
            out_blocks.append(kept_text)
            report["items"].append(
                {
                    "block": bi,
                    "kind": "q",
                    "qno": md_qno,
                    "status": "weak_crop_mask_no_fig",
                    "mapped_match_qi": int(q_idx),
                    "crop_path": str(cp) if cp else None,
                    "crop_qno": crop_qno,
                    "is_unknown": bool(is_unknown),
                    "merge_suspect": bool(merge_suspect),
                    "weak_crop": True,
                    "weak_reasons": weak_reasons,
                    "no_crop_qno_weak": bool(no_crop_qno_weak),
                    "mismatch_soft": bool(mismatch_soft),
                    "mark_only": bool(mark_only),
                    "v_after_repair": v_after,
                    "v_after_raw": v_raw,
                    "mask_confirm": mask_confirm,
                    "repair": dbg,
                }
            )
            continue

        if weak_crop2 and (v_after != "Y") and (not weak_mask_on_weak):
            out_text = _weak_output(b.text, v_after, weak_mask=False, qno=b.qno)  # keep md
        else:
            out_text = final_text

        img_lines: List[str] = []
        figs_kept: List[Path] = []
        figs_dbg: List[Any] = []

        allow_fig = bool((not mismatch_soft) and (((not merge_suspect) or _merge_suspect_override_ok(qrec))) and _qno_ok(md_qno, crop_qno, align_trust_high) and ((not weak_crop2) or weak_keep_figures))
        if fig_vlm is not None and allow_fig:
            fig_paths = find_figure_candidates(page_dir, qrec)
            if fig_paths:
                figs_kept, figs_dbg = select_figures_for_question(
                    crop_img, fig_paths, fig_vlm, cache=cache, cfg=fig_cfg
                )
                if figs_kept:
                    img_lines = [_rel_md_img(fp) for fp in figs_kept]


        if img_lines:
            out_text = normalize_ws(out_text) + "\n\n" + "\n".join(img_lines)

        fixed_from_repair = None
        if fixed_enable and isinstance(dbg, dict):
            acts = _fixed_actions_from_repair_dbg(dbg)

            acts = [a for a in acts if a.get("action") in FIXED_WHITELIST]

            rewrite_acts = dbg.get("rewrite_actions", [])
            has_rewrite = isinstance(rewrite_acts, list) and len(rewrite_acts) > 0
            has_mask = (mask_u_token in (out_text or "")) or (mask_n_token in (out_text or ""))

            if acts and (not has_rewrite) and (not has_mask):
                del_ratio = _calc_del_ratio(b.text, out_text)
                ok_verdict = (v_after == "Y") or (fixed_after_allow_u and v_after == "U")
                if ok_verdict and (del_ratio <= float(fixed_max_del_ratio)):
                    out_text = _inject_fixed_marker(out_text, acts)
                    fixed_from_repair = {
                        "fixed_applied": True,
                        "source": "repair",
                        "actions": acts,
                        "del_ratio": del_ratio,
                        "verdict_after_raw": v_after,
                    }

        out_blocks.append(out_text)

        report["items"].append(
            {
                "block": bi,
                "kind": "q",
                "qno": md_qno,
                "status": dbg.get("stage"),
                "mapped_match_qi": int(q_idx),
                "repair": dbg,
                "figures": figs_dbg,
                "figures_kept": [x.as_posix() for x in figs_kept] if figs_kept else [],
                "crop_path": str(cp) if cp else None,
                "crop_qno": crop_qno,
                "is_unknown": bool(is_unknown),
                "merge_suspect": bool(merge_suspect),
                "weak_crop": bool(weak_crop2),
                "no_crop_qno_weak": bool(no_crop_qno_weak),
                "v_after_repair": v_after,
                "fixed": fixed_from_repair,

            }
        )
        continue

    if first_mismatch_qno is None:
        first_mismatch_qno = _first_mismatch_from_pairs(pairs)

    report["align_safety"] = {
        "used_unknown": int(used_unknown),
        "used_merge_suspect": int(used_merge_suspect),
        "first_mismatch_qno": first_mismatch_qno,
        "weak_when_no_crop_qno": bool(weak_when_no_crop_qno),
        "crop_qno_expected": bool(crop_qno_expected),
        "weak_mask_on_weak": bool(weak_mask_on_weak),
        "weak_keep_figures": bool(weak_keep_figures),
        "ablation_no_patcher": bool(ablation_no_patcher),
    }

    print(
        f"[pipeline] WARN page={page_dir.name} used_unknown={used_unknown} "
        f"used_merge_suspect={used_merge_suspect} first_mismatch_qno={first_mismatch_qno} "
        f"weak_when_no_crop_qno={weak_when_no_crop_qno}"
    )

    out_md = "\n\n".join([normalize_ws(x) for x in out_blocks if (x or "").strip()])
    (out_dir / f"{page_dir.name}_proofread.md").write_text(out_md, encoding="utf-8")
    (out_dir / f"{page_dir.name}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if cache:
        cache.save()

    return report



from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .cache import JsonCache
from .img_utils import BASE_ENHANCE, enhance_for_vlm, img_to_data_url, safe_open_image
from .match_utils import get_crop_path, is_partial_question, q_crop_hfrac

QNO_FROM_CROP_PROMPT = (
    "You are a question-number reader. Read only the earliest question number appearing in the image (Arabic numeral, 1-300).\n"
    "If no question number is visible, output null.\n"
    "Output strict JSON only: {\"qno\": <number|null>}. Do not output any explanation.\n"
)

@dataclass
class AlignResult:
    mapping: Dict[int, Optional[int]]  # md_block_idx -> cand_pos (index in candidates list)
    candidates: List[int]              # list of match_qs_sorted indices (qi) for each cand_pos
    debug: Dict[str, Any]


def extract_qno_from_crop(vlm, img: Image.Image, *, cache: Optional[JsonCache], allowed_qnos: Optional[set[int]] = None) -> Optional[int]:
    img2 = enhance_for_vlm(img, BASE_ENHANCE)
    data_url, img_hash = img_to_data_url(img2)
    ck = f"{img_hash}::{vlm.cache_tag}::qno"
    if cache:
        hit = cache.get("qno", ck)
        if isinstance(hit, int):
            return hit
        if hit is None and ck in (cache.data.get("qno") or {}):
            return None

    raw = vlm.invoke(
        [
            {"type": "text", "text": QNO_FROM_CROP_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
       # temperature=0.0,
        max_tokens=64,
    )
    qno: Optional[int] = None
    try:
        obj = json.loads(raw)
        v = obj.get("qno", None)
        qno = int(v) if v is not None else None
    except Exception:
        import re
        m = re.search(r"\b(\d{1,3})\b", raw or "")
        if m:
            qno = int(m.group(1))
    if qno is not None and not (1 <= qno <= 300):
        qno = None
    if qno is not None and allowed_qnos and (qno not in allowed_qnos):
        qno = None

    if cache:
        cache.set("qno", ck, qno)
    return qno


def _best_offset(md_n: int, cand_n: int) -> int:
    def score(off: int) -> Tuple[int, int, int]:
        in_range = 0
        for k in range(md_n):
            mi = k + off
            if 0 <= mi < cand_n:
                in_range += 1
        return (in_range, -abs(off), 1 if off == 0 else 0)

    max_shift = abs(md_n - cand_n) + 2
    best_off = 0
    best_sc = score(0)
    for off in range(-max_shift, max_shift + 1):
        sc = score(off)
        if sc > best_sc:
            best_sc = sc
            best_off = off
    return best_off

def assign_crops_to_md_blocks(
    blocks: List[Any],
    match_qs_sorted: List[Dict[str, Any]],
    *,
    page_dir: Path,
    page_h: Optional[float],
    skip_partial: bool = True,
    partial_main_min_hfrac: float = 0.0,
    use_offset_search: bool = True,
    use_crop_qno: bool = True,
    qno_vlm=None,
    cache: Optional[JsonCache] = None,
) -> AlignResult:

    from statistics import median
    from typing import Any, Dict, List, Optional, Tuple

    # -------------------------
    # Helper: stable unique
    # -------------------------
    def unique_preserve(xs: List[int]) -> List[int]:
        seen = set()
        out: List[int] = []
        for x in xs:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    # -------------------------
    # Helpers: unknown/reserve
    # -------------------------
    def _is_unknown_crop_path(cp: Optional[str]) -> bool:
        if not cp:
            return False
        s = str(cp)
        return ("/q999999_" in s) or s.startswith("questions/q999999_")

    def _is_unknown_qrec(q: Dict[str, Any]) -> bool:
        cp = get_crop_path(q)
        if _is_unknown_crop_path(cp):
            return True
        ri = q.get("read_index", None)
        return ri is None

    def _is_unknown_pos(cands: List[int], cand_pos: int) -> bool:
        """cand_pos -> qi -> qrec -> unknown?"""
        if cand_pos is None or not (0 <= int(cand_pos) < len(cands)):
            return False
        qi = cands[int(cand_pos)]
        if qi is None or not (0 <= int(qi) < len(match_qs_sorted)):
            return False
        return _is_unknown_qrec(match_qs_sorted[int(qi)])

    # -------------------------
    # md question blocks & qno universe
    # -------------------------
    q_block_idx = [i for i, b in enumerate(blocks) if getattr(b, "kind", None) == "q"]
    allowed_qnos = {
        int(getattr(blocks[i], "qno"))
        for i in q_block_idx
        if getattr(blocks[i], "qno", None) is not None
    }

    md_pos_by_qno: Dict[int, int] = {}
    for pos, bi in enumerate(q_block_idx):
        qno = getattr(blocks[bi], "qno", None)
        if qno is None:
            continue
        md_pos_by_qno[int(qno)] = int(pos)

    # -------------------------
    # 0) Build candidate lists: main vs reserve + partial skip/backfill
    # -------------------------
    main_candidates: List[int] = []
    reserve_candidates: List[int] = []
    skipped_partial: List[int] = []

    for qi, q in enumerate(match_qs_sorted):
        if _is_unknown_qrec(q):
            reserve_candidates.append(qi)
            continue

        if not is_partial_question(q):
            main_candidates.append(qi)
            continue

        if not skip_partial:
            main_candidates.append(qi)
            continue

        if partial_main_min_hfrac <= 0:
            skipped_partial.append(qi)
            continue

        hf = q_crop_hfrac(q, page_h) or 0.0
        if hf >= partial_main_min_hfrac:
            main_candidates.append(qi)
        else:
            skipped_partial.append(qi)

    candidates: List[int] = main_candidates[:]
    backfilled_partial: List[int] = []
    backfilled_reserve: List[int] = []

    need = len(q_block_idx) - len(candidates)
    if need > 0 and skipped_partial:
        sel = skipped_partial[:need]
        backfilled_partial = sel[:]
        candidates = unique_preserve(candidates + sel)

    need = len(q_block_idx) - len(candidates)
    if need > 0 and reserve_candidates:
        sel = reserve_candidates[:need]
        backfilled_reserve = sel[:]
        candidates = candidates + sel  # append only

    dbg: Dict[str, Any] = {
        "skip_partial": skip_partial,
        "partial_main_min_hfrac": partial_main_min_hfrac,
        "n_md_qblocks": len(q_block_idx),
        "n_match_total": len(match_qs_sorted),
        "n_main": len(main_candidates),
        "n_reserve": len(reserve_candidates),
        "n_candidates": len(candidates),
        "skipped_partial": skipped_partial,
        "backfilled_partial": backfilled_partial,
        "backfilled_reserve": backfilled_reserve,
        "backfill_order_stable": True,  
        "mode": None,
        "anchors_used": 0,
        "offset": 0,
        "crop_qnos": [],
        "anchor_duplicates": {},
        "delta_stats": {},
        "chosen_delta": None,
        "pairs": [],
        "candidates_match_qi": candidates[:],
        "attempts": {},
        "winner": None,
        "winner_score": None,
        "warning": None,
        # merge_suspect debug
        "merge_area_med": None,
        "merge_area_n_baseline": 0,
        "merge_factor": None,               
        "merge_factor_small_n": None,
        "merge_small_n_threshold": 3,
        "merge_baseline_kind": "non_unknown(main+skipped_partial)",
        "merge_disabled_no_baseline": False,
        # offset debug
        "offset_scan_span_last": None,
    }

    mapping_empty: Dict[int, Optional[int]] = {bi: None for bi in q_block_idx}
    if not q_block_idx or not candidates:
        dbg["mode"] = "empty"
        return AlignResult(mapping=mapping_empty, candidates=candidates, debug=dbg)

    # -------------------------
    # 1) Optional: read qno from crops (cache by match_q index)
    # -------------------------
    crop_qno_by_qi: Dict[int, Optional[int]] = {}
    qno_enabled = bool(use_crop_qno and (qno_vlm is not None) and allowed_qnos)

    if qno_enabled:

        qis_for_qno = unique_preserve(main_candidates + skipped_partial + reserve_candidates)
        for qi in qis_for_qno:
            q = match_qs_sorted[qi]
            cp = get_crop_path(q)
            if not cp:
                crop_qno_by_qi[qi] = None
                continue
            img = safe_open_image(page_dir / cp)
            if img is None:
                crop_qno_by_qi[qi] = None
                continue
            try:
                crop_qno_by_qi[qi] = extract_qno_from_crop(
                    qno_vlm, img, cache=cache, allowed_qnos=allowed_qnos
                )
            except Exception:
                crop_qno_by_qi[qi] = None

    def _crop_qnos_for(cands: List[int]) -> Dict[int, Optional[int]]:
        return {pos: crop_qno_by_qi.get(qi) for pos, qi in enumerate(cands)}

    # -------------------------
    # 2) Merge-suspect heuristic (baseline fix)
    # -------------------------
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

    baseline_qis = [
        qi for qi in (main_candidates + skipped_partial)
        if (0 <= int(qi) < len(match_qs_sorted)) and (not _is_unknown_qrec(match_qs_sorted[int(qi)]))
    ]
    baseline_areas = [_bbox_area_from_qrec(match_qs_sorted[int(qi)]) for qi in baseline_qis]
    baseline_areas = [a for a in baseline_areas if a > 1e-6]
    n_base = len(baseline_areas)

    SMALL_N = 3
    factor_default = 1.6
    factor_small_n = 2.8

    if n_base == 0:
        dbg["merge_area_med"] = 1.0
        dbg["merge_area_n_baseline"] = 0
        dbg["merge_factor"] = None
        dbg["merge_factor_small_n"] = float(factor_small_n)
        dbg["merge_disabled_no_baseline"] = True

        def _merge_suspect(qi: int) -> bool:
            return False
    else:
        area_med = float(median(baseline_areas))
        merge_factor = factor_default if n_base >= SMALL_N else factor_small_n

        dbg["merge_area_med"] = float(area_med)
        dbg["merge_area_n_baseline"] = int(n_base)
        dbg["merge_factor"] = float(merge_factor)
        dbg["merge_factor_small_n"] = float(factor_small_n)
        dbg["merge_disabled_no_baseline"] = False

        def _merge_suspect(qi: int) -> bool:
            qrec = match_qs_sorted[qi]
            a = _bbox_area_from_qrec(qrec)
            if a <= 1e-6:
                return False
            # Type-aware factor floors: avoid false positives on naturally large problems.
            f = float(merge_factor)
            cls = (qrec.get("class_name") or "").strip()
            if cls == "problem_solving_question":
                f = max(f, 2.3)
            elif cls == "fill_blank_question":
                f = max(f, 2.0)
            elif cls == "partial_question":
                f = max(f, 2.2)
            elif cls == "multiple_choice_question":
                f = max(f, 1.6)
            if bool(qrec.get("is_spanning", False)):
                f = max(f, 2.6)
            #return a > area_med * f
            big = a > area_med * f
            if big:
                ms = qrec.get("matches") or []
                if isinstance(ms, list) and ms and len(ms) <= 2:
                    best = None
                    best_sv2 = -1.0
                    for mm in ms:
                        if not isinstance(mm, dict):
                            continue
                        try:
                            sv2 = float(mm.get("score_v2", mm.get("score", 0.0)) or 0.0)
                        except Exception:
                            sv2 = 0.0
                        if sv2 > best_sv2:
                            best_sv2, best = sv2, mm
                    if isinstance(best, dict):
                        try:
                            inside = float(best.get("inside", 0.0) or 0.0)
                            inw = float(best.get("in_window", 0.0) or 0.0)
                            iof = float(best.get("iof", 0.0) or 0.0)
                        except Exception:
                            inside, inw, iof = 0.0, 0.0, 0.0
                        # If the figure match is very strong, this is likely a legitimate
                        # "question-with-figure" box rather than a merged multi-question crop.
                        if (best_sv2 >= 2.8) and (inside >= 0.95) and (inw >= 0.95) and (iof >= 0.60):
                            return False
            return big
    # -------------------------
    # Internal: build pairs
    # -------------------------
    def _build_pairs(
        q_blocks: List[int],
        cands: List[int],
        crop_qnos: Dict[int, Optional[int]],
        mapping_in: Dict[int, int],
    ) -> List[Dict[str, Any]]:
        pairs: List[Dict[str, Any]] = []
        for bi in q_blocks:
            cand_pos = mapping_in.get(bi)
            qi = cands[cand_pos] if (cand_pos is not None and 0 <= cand_pos < len(cands)) else None
            qrec = match_qs_sorted[qi] if qi is not None else None

            cp = None
            is_unk = False
            ms = False
            if qrec is not None:
                cp = get_crop_path(qrec)
                is_unk = _is_unknown_qrec(qrec)
                ms = _merge_suspect(int(qi))

            pairs.append(
                {
                    "md_block": int(bi),
                    "qno": getattr(blocks[bi], "qno", None),
                    "cand_pos": cand_pos,
                    "match_qi": qi,
                    "match_class": (qrec.get("class_name") if qrec is not None else None),
                    "crop_path": cp,
                    "crop_qno": crop_qnos.get(cand_pos) if cand_pos is not None else None,
                    "is_unknown": bool(is_unk),
                    "merge_suspect": bool(ms),
                }
            )
        return pairs

    # -------------------------
    # Internal: fill remaining mapping (offset_search / sequential)
    # -------------------------
    def _best_offset_by_qno(
        remain_md: List[int],
        remain_cands: List[int],
        crop_qnos: Dict[int, Optional[int]],
    ) -> int:
        if not remain_md or not remain_cands:
            return 0

        any_sig = any(crop_qnos.get(cp) is not None for cp in remain_cands)
        if not any_sig:
            return 0

        best_off = 0
        best_key = None

        span = min(len(remain_md), len(remain_cands))
        dbg["offset_scan_span_last"] = int(span)

        lo = -span
        hi = +span

        for off in range(lo, hi + 1):
            eq = 0
            none_cnt = 0
            overlap = 0

            for k, bi in enumerate(remain_md):
                j = k + off
                if j < 0 or j >= len(remain_cands):
                    continue

                overlap += 1

                cand_pos = remain_cands[j]
                md_qno = getattr(blocks[bi], "qno", None)
                cq = crop_qnos.get(cand_pos)

                if md_qno is None or cq is None:
                    none_cnt += 1
                    continue

                try:
                    if int(md_qno) == int(cq):
                        eq += 1
                except Exception:
                    none_cnt += 1

            key = (-eq, none_cnt, -overlap, abs(off))
            if best_key is None or key < best_key:
                best_key = key
                best_off = off

        return int(best_off)

    def _fill_with_offset(
        q_blocks: List[int],
        cand_positions: List[int],
        mapping_in: Dict[int, int],
        used: set[int],
        *,
        crop_qnos: Dict[int, Optional[int]],
    ) -> Tuple[int, str]:
        remain_md = [bi for bi in q_blocks if mapping_in.get(bi) is None]
        remain_cands = [cp for cp in cand_positions if cp not in used]
        if not remain_md or not remain_cands:
            return 0, "none"

        if use_offset_search:
            off = _best_offset_by_qno(remain_md, remain_cands, crop_qnos)
            for k, bi in enumerate(remain_md):
                j = k + off
                if 0 <= j < len(remain_cands):
                    mapping_in[bi] = remain_cands[j]
                    used.add(remain_cands[j])
            return int(off), "offset_search"
        else:
            for k, bi in enumerate(remain_md):
                if k < len(remain_cands):
                    mapping_in[bi] = remain_cands[k]
                    used.add(remain_cands[k])
            return 0, "sequential"

    # -------------------------
    # Internal: delta stats
    # -------------------------
    def _compute_delta_stats(
        cands: List[int],
        crop_qnos: Dict[int, Optional[int]],
        *,
        allow_reserve_anchors: bool = False,
    ) -> Tuple[Optional[int], Dict[str, int], Dict[str, List[int]]]:
        qno_to_positions: Dict[int, List[int]] = {}
        for cand_pos, qno in crop_qnos.items():
            if qno is None:
                continue
            qi = cands[int(cand_pos)]
            if (not allow_reserve_anchors) and _is_unknown_qrec(match_qs_sorted[qi]):
                continue
            qno_to_positions.setdefault(int(qno), []).append(int(cand_pos))

        anchor_duplicates: Dict[str, List[int]] = {}
        for qno, poses in qno_to_positions.items():
            if len(poses) > 1:
                anchor_duplicates[str(qno)] = poses[:]

        deltas: Dict[int, int] = {}
        for qno, poses in qno_to_positions.items():
            if qno not in md_pos_by_qno:
                continue
            md_pos = md_pos_by_qno[qno]
            for cand_pos in poses:
                d = int(cand_pos) - int(md_pos)
                deltas[d] = deltas.get(d, 0) + 1

        chosen_delta: Optional[int] = None
        if deltas:
            best_d, best_cnt = max(deltas.items(), key=lambda kv: kv[1])
            if best_cnt >= 2:
                chosen_delta = int(best_d)

        return chosen_delta, {str(k): int(v) for k, v in deltas.items()}, anchor_duplicates

    # -------------------------
    # Attempt A
    # -------------------------
    def _align_delta_offset(
        cands: List[int],
        tag: str,
        *,
        allow_reserve_anchors: bool = False,
        allow_reserve_fill: bool = False,
    ) -> Tuple[Dict[int, int], Dict[str, Any], List[Dict[str, Any]]]:
        crop_qnos = _crop_qnos_for(cands)
        chosen_delta, delta_stats, anchor_dups = _compute_delta_stats(
            cands, crop_qnos, allow_reserve_anchors=allow_reserve_anchors
        )

        mapping_local: Dict[int, int] = {}
        used_cand_pos: set[int] = set()

        mode = None
        anchors_used = 0
        off_used = 0

        if chosen_delta is not None:
            mode = "crop_qno_offset"
            for pos, bi in enumerate(q_block_idx):
                cand_pos = pos + chosen_delta
                if not (0 <= cand_pos < len(cands)):
                    continue
                if cand_pos in used_cand_pos:
                    continue
                if (not allow_reserve_fill) and _is_unknown_pos(cands, cand_pos):
                    continue
                mapping_local[bi] = cand_pos
                used_cand_pos.add(cand_pos)
            anchors_used = len(used_cand_pos)

        cand_positions = list(range(len(cands)))
        if not allow_reserve_fill:
            cand_positions = [p for p in cand_positions if not _is_unknown_pos(cands, p)]

        off, fill_mode = _fill_with_offset(
            q_block_idx,
            cand_positions,
            mapping_local,
            used_cand_pos,
            crop_qnos=crop_qnos,
        )
        off_used = off
        mode = mode or fill_mode

        pairs = _build_pairs(q_block_idx, cands, crop_qnos, mapping_local)
        dbg_local = {
            "tag": tag,
            "mode": mode,
            "anchors_used": int(anchors_used),
            "mapped_total": int(len(mapping_local)),
            "offset": int(off_used),
            "crop_qnos": [crop_qnos.get(i) for i in range(len(cands))],
            "delta_stats": delta_stats,
            "chosen_delta": chosen_delta,
            "anchor_duplicates": anchor_dups,
            "candidates_match_qi": cands[:],
            "allow_reserve_fill": bool(allow_reserve_fill),
        }
        return mapping_local, dbg_local, pairs

    # -------------------------
    # Attempt B
    # -------------------------
    def _align_qno_match_first(
        cands: List[int], tag: str, *, allow_reserve_fill: bool = False
    ) -> Tuple[Dict[int, int], Dict[str, Any], List[Dict[str, Any]]]:
        crop_qnos = _crop_qnos_for(cands)

        pos_by_qno: Dict[int, List[int]] = {}
        for cand_pos, qno in crop_qnos.items():
            if qno is None:
                continue
            pos_by_qno.setdefault(int(qno), []).append(int(cand_pos))
        for qno in pos_by_qno:
            pos_by_qno[qno].sort()

        mapping_local: Dict[int, int] = {}
        used: set[int] = set()

        for bi in q_block_idx:
            md_qno = getattr(blocks[bi], "qno", None)
            if md_qno is None:
                continue
            qno_i = int(md_qno)
            poses = pos_by_qno.get(qno_i)
            if not poses:
                continue

            md_pos = md_pos_by_qno.get(qno_i, None)

            best = None
            for p in poses:
                if p in used:
                    continue
                qi = cands[p]
                qrec = match_qs_sorted[qi]

                unk = _is_unknown_qrec(qrec)
                ms = _merge_suspect(qi)
                dist = abs(int(p) - int(md_pos)) if md_pos is not None else 0
                a = _bbox_area_from_qrec(qrec)
                hf = q_crop_hfrac(qrec, page_h) or 0.0

                key = (1 if ms else 0, 1 if unk else 0, dist, hf, a, p)
                if best is None or key < best[0]:
                    best = (key, p)

            if best is None:
                continue
            pick = int(best[1])
            mapping_local[bi] = pick
            used.add(pick)

        exact_hits = int(len(mapping_local))

        cand_positions = list(range(len(cands)))
        if not allow_reserve_fill:
            cand_positions = [p for p in cand_positions if not _is_unknown_pos(cands, p)]

        off, fill_mode = _fill_with_offset(
            q_block_idx,
            cand_positions,
            mapping_local,
            used,
            crop_qnos=crop_qnos,
        )
        pairs = _build_pairs(q_block_idx, cands, crop_qnos, mapping_local)

        dbg_local = {
            "tag": tag,
            "mode": "qno_match_first+" + fill_mode,
            "anchors_used": int(exact_hits),
            "mapped_total": int(len(mapping_local)),
            "offset": int(off),
            "crop_qnos": [crop_qnos.get(i) for i in range(len(cands))],
            "delta_stats": {},
            "chosen_delta": None,
            "anchor_duplicates": {},
            "candidates_match_qi": cands[:],
            "allow_reserve_fill": bool(allow_reserve_fill),
        }
        return mapping_local, dbg_local, pairs

    # -------------------------
    # Scoring
    # -------------------------
    def _tail_mismatch_run(pairs: List[Dict[str, Any]]) -> int:
        run = 0
        for p in reversed(pairs):
            mq = p.get("match_qi")
            a = p.get("qno")
            b = p.get("crop_qno")

            if mq is None:
                continue
            if b is None:
                run += 1
                continue
            if a is None:
                run += 1
                continue
            try:
                if int(a) == int(b):
                    break
            except Exception:
                run += 1
                continue
            run += 1
        return run

    def _attempt_score(pairs: List[Dict[str, Any]]) -> Tuple[int, int, int, int, int, int, int]:
        mapped = sum(1 for p in pairs if p.get("match_qi") is not None)

        comparable_pairs = [
            (p.get("qno"), p.get("crop_qno"))
            for p in pairs
            if (p.get("match_qi") is not None) and (p.get("qno") is not None) and (p.get("crop_qno") is not None)
        ]
        comparable = len(comparable_pairs)

        uncompared = mapped - comparable
        none_qno = sum(1 for p in pairs if (p.get("match_qi") is not None) and (p.get("crop_qno") is None))

        eq = 0
        for a, b in comparable_pairs:
            try:
                if int(a) == int(b):
                    eq += 1
            except Exception:
                pass
        mism = comparable - eq

        unk = sum(1 for p in pairs if p.get("is_unknown"))
        tail = _tail_mismatch_run(pairs)
        return (tail, int(uncompared), int(none_qno), int(mism), int(unk), -int(eq), -int(mapped))

    def _tail_drift_start(pairs: List[Dict[str, Any]]) -> Optional[int]:
        tail = _tail_mismatch_run(pairs)
        if tail <= 0:
            return None
        cnt = 0
        for p in reversed(pairs):
            if p.get("match_qi") is None:
                continue
            cnt += 1
            if cnt == tail:
                try:
                    return int(p.get("qno"))
                except Exception:
                    return None
        return None

    # -------------------------
    # Multi-attempt selector
    # -------------------------
    attempts: List[
        Tuple[
            Tuple[int, int, int, int, int, int, int],
            Dict[int, int],
            Dict[str, Any],
            List[Dict[str, Any]],
        ]
    ] = []

    def _consider(mapping_c: Dict[int, int], dbg_c: Dict[str, Any], pairs_c: List[Dict[str, Any]]):
        sc = _attempt_score(pairs_c)
        tail_run, uncompared, none_qno, mism, unk, neg_eq, neg_mapped = sc
        dbg["attempts"][dbg_c["tag"]] = {
            "score": sc,
            "tail_run": tail_run,
            "uncompared": uncompared,
            "none_qno": none_qno,
            "mismatches": mism,
            "unknown_used": unk,
            "qno_equal": -neg_eq,
            "mapped": -neg_mapped,
            **dbg_c,
        }
        attempts.append((sc, mapping_c, dbg_c, pairs_c))

    if not qno_enabled:
        m0, d0, p0 = _align_delta_offset(
            candidates,
            "primary_delta_offset_no_qno",
            allow_reserve_anchors=False,
            allow_reserve_fill=False,
        )
        dbg["mode"] = d0["mode"]
        dbg["anchors_used"] = d0["anchors_used"]
        dbg["offset"] = d0["offset"]
        dbg["crop_qnos"] = d0["crop_qnos"]
        dbg["delta_stats"] = d0["delta_stats"]
        dbg["chosen_delta"] = d0["chosen_delta"]
        dbg["anchor_duplicates"] = d0["anchor_duplicates"]
        dbg["pairs"] = p0
        dbg["candidates_match_qi"] = d0["candidates_match_qi"][:]
        dbg["winner"] = d0["tag"]
        dbg["winner_score"] = _attempt_score(p0)

        drift_start = _tail_drift_start(p0)
        dbg["warning"] = (
            f"[align] WARN winner={d0['tag']} mode={d0['mode']} "
            f"score={dbg['winner_score']} drift_start_qno={drift_start}"
        )
        print(dbg["warning"])
        return AlignResult(mapping=m0, candidates=d0["candidates_match_qi"][:], debug=dbg)

    mA, dA, pA = _align_delta_offset(
        candidates,
        "A_delta_offset_primary",
        allow_reserve_anchors=False,
        allow_reserve_fill=False,
    )
    _consider(mA, dA, pA)

    full_cands = unique_preserve(main_candidates + skipped_partial + reserve_candidates)
    mB, dB, pB = _align_qno_match_first(
        full_cands,
        "B_qno_match_first_full",
        allow_reserve_fill=False,
    )
    _consider(mB, dB, pB)

    if main_candidates:
        mC, dC, pC = _align_delta_offset(
            main_candidates,
            "C_delta_offset_main_only",
            allow_reserve_anchors=False,
            allow_reserve_fill=False,
        )
        _consider(mC, dC, pC)

    attempts.sort(key=lambda x: x[0])
    best_sc, best_mapping, best_dbg, best_pairs = attempts[0]

    dbg["winner"] = best_dbg["tag"]
    dbg["winner_score"] = best_sc
    dbg["mode"] = best_dbg["mode"]
    dbg["anchors_used"] = best_dbg.get("anchors_used", 0)
    dbg["offset"] = best_dbg.get("offset", 0)
    dbg["crop_qnos"] = best_dbg.get("crop_qnos", [])
    dbg["delta_stats"] = best_dbg.get("delta_stats", {})
    dbg["chosen_delta"] = best_dbg.get("chosen_delta", None)
    dbg["anchor_duplicates"] = best_dbg.get("anchor_duplicates", {})
    dbg["pairs"] = best_pairs
    dbg["candidates_match_qi"] = best_dbg["candidates_match_qi"][:]
    if "mapped_total" in best_dbg:
        dbg["mapped_total"] = best_dbg.get("mapped_total")

    drift_start = _tail_drift_start(best_pairs)
    dbg["warning"] = (
        f"[align] WARN winner={best_dbg['tag']} mode={best_dbg['mode']} "
        f"score={best_sc} drift_start_qno={drift_start}"
    )
    print(dbg["warning"])

    return AlignResult(mapping=best_mapping, candidates=best_dbg["candidates_match_qi"][:], debug=dbg)

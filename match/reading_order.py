from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

XYXY = Tuple[float, float, float, float]


# -------------------------
# Basic geometry
# -------------------------
def bbox_w(b: XYXY) -> float:
    return max(0.0, float(b[2] - b[0]))


def bbox_h(b: XYXY) -> float:
    return max(0.0, float(b[3] - b[1]))


def bbox_area(b: XYXY) -> float:
    return bbox_w(b) * bbox_h(b)


def inter_area(a: XYXY, b: XYXY) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a: XYXY, b: XYXY) -> float:
    inter = inter_area(a, b)
    if inter <= 0.0:
        return 0.0
    ua = bbox_area(a) + bbox_area(b) - inter
    return float(inter / max(1e-9, ua))


def containment_ratio(a: XYXY, b: XYXY) -> float:
    """Inter / min(area(a), area(b)) ∈ [0,1]. High means one box mostly contains the other."""
    inter = inter_area(a, b)
    if inter <= 0.0:
        return 0.0
    denom = min(bbox_area(a), bbox_area(b))
    return float(inter / max(1e-9, denom))


def _overlaps(a: XYXY, b: XYXY, *, iou_thr: float, contain_thr: float) -> bool:
    ia = inter_area(a, b)
    if ia <= 0.0:
        return False
    if iou_xyxy(a, b) >= float(iou_thr):
        return True
    if containment_ratio(a, b) >= float(contain_thr):
        return True
    return False


def filter_items_by_overlap(
    items: List[Dict[str, Any]],
    *,
    mode: str = "keep",
    iou_thr: float = 0.92,
    contain_thr: float = 0.95,
) -> List[Dict[str, Any]]:
    """Filter items by overlap/containment (deterministic).

    mode:
      - keep: keep all
      - keep_large / drop_small: keep only the largest boxes within an overlap cluster
      - keep_small: keep only the smallest boxes within an overlap cluster
      - drop_all: drop any box that overlaps another above thresholds

    Notes:
      - Filtering is applied on item['bbox'] (the current working bbox).
      - Stable: for equal area, original order is preserved.
    """
    mode = (mode or "keep").lower()
    if mode in ("keep", "all", "none"):
        return items
    if len(items) <= 1:
        return items

    if mode == "drop_all":
        bad = set()
        for i in range(len(items)):
            bi = items[i].get("bbox")
            if not bi:
                continue
            for j in range(i + 1, len(items)):
                bj = items[j].get("bbox")
                if not bj:
                    continue
                if _overlaps(bi, bj, iou_thr=iou_thr, contain_thr=contain_thr):
                    bad.add(i)
                    bad.add(j)
        return [it for k, it in enumerate(items) if k not in bad]

    keep_small = mode in ("keep_small", "small")
    keyed = [(bbox_area(it["bbox"]), idx, it) for idx, it in enumerate(items) if it.get("bbox")]
    keyed.sort(key=lambda t: (t[0], -t[1]) if keep_small else (-t[0], t[1]))

    kept: List[Dict[str, Any]] = []
    kept_b: List[XYXY] = []
    for _, _, it in keyed:
        b = it["bbox"]
        conflict = False
        for kb in kept_b:
            if _overlaps(b, kb, iou_thr=iou_thr, contain_thr=contain_thr):
                conflict = True
                break
        if not conflict:
            kept.append(it)
            kept_b.append(b)

    keep_set = {id(it) for it in kept}
    return [it for it in items if id(it) in keep_set]

def x_center(b: XYXY) -> float:
    return (float(b[0]) + float(b[2])) * 0.5


def y_center(b: XYXY) -> float:
    return (float(b[1]) + float(b[3])) * 0.5


def clip_bbox(b: XYXY, W: float, H: float) -> XYXY:
    x1, y1, x2, y2 = map(float, b)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = min(max(x1, 0.0), W)
    x2 = min(max(x2, 0.0), W)
    y1 = min(max(y1, 0.0), H)
    y2 = min(max(y2, 0.0), H)
    return (x1, y1, x2, y2)


def shrink_bbox(b: XYXY, frac: float = 0.008) -> XYXY:
    """轻度内缩，减弱“右侧拖宽/粘连边界”对判列与中缝的影响。"""
    x1, y1, x2, y2 = map(float, b)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    dx = w * frac
    dy = h * frac
    return (x1 + dx, y1 + dy, x2 - dx, y2 - dy)


# -------------------------
# Spread detection by gutter "coverage dip"
# -------------------------
@dataclass
class GutterDebug:
    accepted: bool
    split_x: Optional[float]
    min_cov: float
    mid_cov: float
    n_eligible: int
    left_count: int
    right_count: int
    near_gutter_frac: float
    reason: str


def gutter_coverage_profile(
    boxes: List[XYXY],
    W: float,
    H: float,
    *,
    n_bins: int = 80,
    min_box_h_frac: float = 0.035,
    min_box_w_frac: float = 0.08,
    max_box_w_frac: float = 0.75,
) -> Tuple[List[float], List[float], List[XYXY]]:
    """沿 x 方向做覆盖度 profile，寻找“中缝”dip。"""
    eligible: List[XYXY] = []
    for b in boxes:
        bw = bbox_w(b)
        bh = bbox_h(b)
        if H > 1e-6 and (bh / H) < min_box_h_frac:
            continue
        if W > 1e-6 and (bw / W) < min_box_w_frac:
            continue
        if W > 1e-6 and (bw / W) > max_box_w_frac:
            continue
        eligible.append(b)

    if not eligible:
        return [0.0] * n_bins, [((i + 0.5) / n_bins) * W for i in range(n_bins)], []

    cov = [0.0] * n_bins
    xs = [((i + 0.5) / n_bins) * W for i in range(n_bins)]
    for b in eligible:
        x1, _, x2, _ = b
        lo = int(max(0, math.floor((x1 / max(W, 1e-9)) * n_bins)))
        hi = int(min(n_bins - 1, math.floor((x2 / max(W, 1e-9)) * n_bins)))
        for i in range(lo, hi + 1):
            cov[i] += 1.0

    denom = float(len(eligible))
    cov = [c / denom for c in cov]
    return cov, xs, eligible


def find_gutter_split_x(
    boxes: List[XYXY],
    W: float,
    H: float,
    *,
    mid_range: Tuple[float, float] = (0.35, 0.65),
    dip_ratio: float = 0.72,
    min_side_items: int = 2,
    near_gutter_px_frac: float = 0.03,
    max_near_gutter_frac: float = 0.55,
) -> Tuple[Optional[float], GutterDebug]:
    cov, xs, eligible = gutter_coverage_profile(boxes, W, H)
    if not eligible:
        return None, GutterDebug(False, None, 0.0, 0.0, 0, 0, 0, 0.0, "no eligible boxes")

    lo = int(mid_range[0] * len(xs))
    hi = int(mid_range[1] * len(xs))
    lo = max(0, min(lo, len(xs) - 1))
    hi = max(lo + 1, min(hi, len(xs)))

    mid_cov = sum(cov[lo:hi]) / max(1, (hi - lo))
    min_i = min(range(lo, hi), key=lambda i: cov[i])
    min_cov = cov[min_i]
    split_x = xs[min_i]

    if mid_cov <= 1e-9:
        return None, GutterDebug(False, None, min_cov, mid_cov, len(eligible), 0, 0, 0.0, "mid_cov too small")
    if min_cov > dip_ratio * mid_cov:
        return None, GutterDebug(False, None, min_cov, mid_cov, len(eligible), 0, 0, 0.0, "no clear dip")

    near_px = near_gutter_px_frac * W
    left = right = near = 0
    for b in boxes:
        xc = x_center(b)
        if abs(xc - split_x) <= near_px:
            near += 1
        if xc < split_x:
            left += 1
        else:
            right += 1

    total = max(1, len(boxes))
    near_frac = near / total

    if left < min_side_items or right < min_side_items:
        return None, GutterDebug(False, split_x, min_cov, mid_cov, len(eligible), left, right, near_frac, "too few items on one side")
    if near_frac > max_near_gutter_frac:
        return None, GutterDebug(False, split_x, min_cov, mid_cov, len(eligible), left, right, near_frac, "too many near gutter")

    return split_x, GutterDebug(True, split_x, min_cov, mid_cov, len(eligible), left, right, near_frac, "accepted")


# -------------------------
# Column analysis (x1 + elbow + merge)
# -------------------------
@dataclass
class ColumnDebug:
    k: int
    centers: List[float]
    splits: List[float]
    carrier_n: int
    single_col_forced: bool
    reason: str


def _kmeans_1d(points: List[float], k: int, iters: int = 50) -> Tuple[List[float], List[int], float]:
    """Deterministic 1D k-means with quantile init."""
    if k <= 1 or len(points) == 0:
        c = [sum(points) / len(points)] if points else [0.0]
        labels = [0] * len(points)
        sse = sum((p - c[0]) ** 2 for p in points) if points else 0.0
        return c, labels, sse

    pts = sorted(points)
    centers = []
    for i in range(k):
        idx = int(round(i * (len(pts) - 1) / max(1, (k - 1))))
        centers.append(float(pts[idx]))

    labels = [0] * len(points)
    for _ in range(iters):
        changed = False
        for i, p in enumerate(points):
            best = min(range(k), key=lambda j: abs(p - centers[j]))
            if labels[i] != best:
                labels[i] = best
                changed = True

        new_centers = centers[:]
        for j in range(k):
            group = [points[i] for i in range(len(points)) if labels[i] == j]
            if group:
                new_centers[j] = sum(group) / len(group)

        if all(abs(new_centers[j] - centers[j]) < 1e-6 for j in range(k)):
            centers = new_centers
            break
        centers = new_centers
        if not changed:
            break

    sse = 0.0
    for i, p in enumerate(points):
        sse += (p - centers[labels[i]]) ** 2
    return centers, labels, float(sse)


def choose_k_elbow(points: List[float], *, max_k: int = 3, improve_thresh: float = 0.20) -> Tuple[int, List[float]]:
    if len(points) < 2:
        return 1, [0.0]
    max_k = max(1, min(int(max_k), len(points)))
    sse_list = []
    centers_list = []
    for k in range(1, max_k + 1):
        centers, labels, sse = _kmeans_1d(points, k)
        sse_list.append(float(sse))
        centers_list.append(centers)

    chosen = 1
    for k in range(2, max_k + 1):
        prev = sse_list[k - 2]
        cur = sse_list[k - 1]
        rel = (prev - cur) / max(prev, 1e-9)
        if rel >= improve_thresh:
            chosen = k
        else:
            break
    return chosen, sorted(map(float, centers_list[chosen - 1]))


def analyze_columns_x1(
    items: List[Dict[str, Any]],
    W: float,
    H: float,
    *,
    max_cols: int = 3,
    carrier_min_w_frac: float = 0.07,
    carrier_max_w_frac: float = 0.85,
    carrier_min_h_frac: float = 0.04,
    elbow_improve_thresh: float = 0.20,
    single_col_force_ratio: float = 0.80,
    min_center_sep_frac: float = 0.12,
) -> Tuple[int, List[float], List[float], ColumnDebug]:
    carriers = []
    x1s: List[float] = []
    for it in items:
        b = it["bbox"]
        bw = bbox_w(b)
        bh = bbox_h(b)
        if W > 1e-6:
            if (bw / W) < carrier_min_w_frac:
                continue
            if (bw / W) > carrier_max_w_frac:
                continue
        if H > 1e-6 and (bh / H) < carrier_min_h_frac:
            continue
        carriers.append(it)
        x1s.append(float(b[0]))

    if len(x1s) < 2:
        dbg = ColumnDebug(1, [0.0], [], len(x1s), False, "not enough carriers")
        return 1, [0.0], [], dbg

    k0, _ = choose_k_elbow(x1s, max_k=max_cols, improve_thresh=elbow_improve_thresh)

    raw_centers, labels, _ = _kmeans_1d(x1s, k0)
    sizes = [0] * k0
    for lb in labels:
        sizes[lb] += 1

    cs = sorted([(float(raw_centers[j]), int(sizes[j])) for j in range(k0)], key=lambda t: t[0])

    # merge close centers
    min_sep = float(min_center_sep_frac) * float(W)
    merged = True
    while merged and len(cs) > 1:
        merged = False
        new_cs = []
        i = 0
        while i < len(cs):
            if i < len(cs) - 1 and abs(cs[i + 1][0] - cs[i][0]) < min_sep:
                c1, s1 = cs[i]
                c2, s2 = cs[i + 1]
                s = s1 + s2
                c = (c1 * s1 + c2 * s2) / max(1, s)
                new_cs.append((float(c), int(s)))
                i += 2
                merged = True
            else:
                new_cs.append(cs[i])
                i += 1
        cs = new_cs

    centers = [c for c, _ in cs]
    k = len(centers)

    forced = False
    reason = "elbow+merge"

    if k > 1:
        max_frac = max(s for _, s in cs) / max(1, len(x1s))
        min_j = min(range(k), key=lambda j: cs[j][1])
        min_center = float(cs[min_j][0])
        min_size = int(cs[min_j][1])

        edge_frac = 0.80
        minority_is_edge_strip = (min_center >= edge_frac * W) or (min_center <= (1.0 - edge_frac) * W)
        minority_too_small = (min_size < 2)

        if max_frac >= single_col_force_ratio and (minority_too_small or (not minority_is_edge_strip)):
            k = 1
            centers = [sum(x1s) / len(x1s)]
            forced = True
            reason = f"single_col_forced max_frac={max_frac:.2f} min_center={min_center:.1f} sizes={[s for _, s in cs]}"

    splits: List[float] = []
    if k > 1:
        centers = sorted(map(float, centers))
        for i in range(k - 1):
            splits.append((centers[i] + centers[i + 1]) * 0.5)

    dbg = ColumnDebug(int(k), list(map(float, centers)), list(map(float, splits)), len(x1s), forced, reason)
    return int(k), list(map(float, centers)), list(map(float, splits)), dbg


# -------------------------
# Span-based zoning (no-drop)
# -------------------------
def is_span_box(b: XYXY, W: float, H: float, *, span_w_frac: float = 0.88, span_min_h_frac: float = 0.035) -> bool:
    if W <= 1e-6 or H <= 1e-6:
        return False
    return (bbox_w(b) / W) >= float(span_w_frac) and (bbox_h(b) / H) >= float(span_min_h_frac)


def partition_by_spans_no_drop(
    items: List[Dict[str, Any]],
    W: float,
    H: float,
    *,
    span_w_frac: float = 0.88,
    span_min_h_frac: float = 0.035,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    spans = []
    nons = []
    for it in items:
        if it.get("force_span") or is_span_box(it["bbox_raw"], W, H, span_w_frac=span_w_frac, span_min_h_frac=span_min_h_frac):
            spans.append(it)
        else:
            nons.append(it)

    spans = sorted(spans, key=lambda it: it["bbox_raw"][1])

    used = set()
    segs: List[Tuple[str, List[Dict[str, Any]]]] = []
    for sp in spans:
        y_cut = float(sp["bbox_raw"][1])
        above = []
        for idx, it in enumerate(nons):
            if idx in used:
                continue
            if y_center(it["bbox_raw"]) < y_cut:
                above.append((idx, it))
        if above:
            above_sorted = [it for _, it in sorted(above, key=lambda p: (p[1]["bbox_raw"][1], p[1]["bbox_raw"][0]))]
            for idx, _ in above:
                used.add(idx)
            segs.append(("zone", above_sorted))
        segs.append(("span", [sp]))

    tail = [it for idx, it in enumerate(nons) if idx not in used]
    if tail:
        tail = sorted(tail, key=lambda it: (it["bbox_raw"][1], it["bbox_raw"][0]))
        segs.append(("zone", tail))

    if not segs:
        segs = [("zone", items)]
    return segs


# -------------------------
# Ordering within a (sub)page
# -------------------------
@dataclass
class PageDebug:
    W: float
    H: float
    segments: List[Dict[str, Any]]


def order_zone(items: List[Dict[str, Any]], W: float, H: float, *, max_cols: int = 3, **col_kwargs: Any):
    k, centers, splits, cdbg = analyze_columns_x1(items, W, H, max_cols=max_cols, **col_kwargs)

    cols: List[List[Dict[str, Any]]] = [[] for _ in range(max(1, k))]
    for it in items:
        x1 = float(it["bbox"][0])
        cid = 0
        for s in splits:
            if x1 >= s:
                cid += 1
        cid = min(max(cid, 0), max(1, k) - 1)
        cols[cid].append(it)

    for cid in range(max(1, k)):
        cols[cid].sort(key=lambda it: (it["bbox"][1], it["bbox"][0], it["bbox"][3], it["bbox"][2]))

    ordered: List[Dict[str, Any]] = []
    for cid in range(max(1, k)):
        ordered.extend(cols[cid])

    return ordered, {"col": asdict(cdbg)}


def order_page(
    items: List[Dict[str, Any]],
    W: float,
    H: float,
    *,
    max_cols: int = 3,
    span_w_frac: float = 0.86,
    span_min_h_frac: float = 0.030,
    **col_kwargs: Any,
) -> Tuple[List[Dict[str, Any]], PageDebug]:
    segs = partition_by_spans_no_drop(items, W, H, span_w_frac=span_w_frac, span_min_h_frac=span_min_h_frac)

    out: List[Dict[str, Any]] = []
    seg_dbg: List[Dict[str, Any]] = []
    for typ, seg_items in segs:
        if typ == "span":
            seg_items = sorted(seg_items, key=lambda it: (it["bbox"][1], it["bbox"][0]))
            out.extend(seg_items)
            seg_dbg.append({"type": "span", "n": len(seg_items)})
        else:
            ordered, dbg = order_zone(seg_items, W, H, max_cols=max_cols, **col_kwargs)
            out.extend(ordered)
            seg_dbg.append({"type": "zone", "n": len(seg_items), **dbg})

    return out, PageDebug(float(W), float(H), seg_dbg)


# -------------------------
# Public API
# -------------------------
@dataclass
class ReadIndexV92Cfg:
    # main
    max_cols: int = 3
    read_index_base: int = 1

    # span
    span_w_frac: float = 0.86
    span_min_h_frac: float = 0.030
    shrink_frac: float = 0.008


    # overlap/containment suppression (optional)
    overlap_mode: str = "keep"  # keep | keep_large | keep_small | drop_all
    overlap_iou_thr: float = 0.92
    overlap_contain_thr: float = 0.95
    # spread / gutter
    min_side_items: int = 2
    near_gutter_px_frac: float = 0.03
    max_near_gutter_frac: float = 0.55
    dip_ratio: float = 0.72


def assign_read_index_v9_2(
    questions: List[object],
    *,
    page_w: float,
    page_h: float,
    cfg: Optional[ReadIndexV92Cfg] = None,
) -> Dict[str, Any]:
    """
    给 QuestionBox 列表就地写入 read_index，并返回 ro_debug（可写入输出 JSON）。

    questions 需要字段：
      - bbox_xyxy: (x1,y1,x2,y2)
    """
    if cfg is None:
        cfg = ReadIndexV92Cfg()

    W = float(max(1.0, page_w))
    H = float(max(1.0, page_h))

    items: List[Dict[str, Any]] = []
    for i, q in enumerate(questions):
        b = getattr(q, "bbox_xyxy", None)
        if not b:
            continue
        b_raw = clip_bbox(tuple(map(float, b)), W, H)
        force_span = is_span_box(b_raw, W, H, span_w_frac=cfg.span_w_frac, span_min_h_frac=cfg.span_min_h_frac)
        b_work = shrink_bbox(b_raw, frac=cfg.shrink_frac)
        items.append({"_qi": i, "bbox": b_work, "bbox_raw": b_raw, "force_span": force_span})


    # optional overlap/containment suppression before doing spread/column logic
    if getattr(cfg, "overlap_mode", "keep") != "keep":
        items = filter_items_by_overlap(
            items,
            mode=str(cfg.overlap_mode),
            iou_thr=float(getattr(cfg, "overlap_iou_thr", 0.92)),
            contain_thr=float(getattr(cfg, "overlap_contain_thr", 0.95)),
        )

    if not items:
        return {"mode": "empty", "reading_order": [], "gutter": {"accepted": False, "reason": "no valid bboxes"}, "page": {}}

    split_x, gdbg = find_gutter_split_x(
        [it["bbox"] for it in items],
        W,
        H,
        min_side_items=cfg.min_side_items,
        near_gutter_px_frac=cfg.near_gutter_px_frac,
        max_near_gutter_frac=cfg.max_near_gutter_frac,
        dip_ratio=cfg.dip_ratio,
    )

    if split_x is None or (not gdbg.accepted):
        ordered_items, pdbg = order_page(
            items,
            W,
            H,
            max_cols=cfg.max_cols,
            span_w_frac=cfg.span_w_frac,
            span_min_h_frac=cfg.span_min_h_frac,
        )
        ro = [it["_qi"] for it in ordered_items]
        idx_to_rank = {qi: (rank + int(cfg.read_index_base)) for rank, qi in enumerate(ro)}
        for qi, q in enumerate(questions):
            setattr(q, "read_index", idx_to_rank.get(qi))
        return {"mode": "single", "reading_order": ro, "gutter": asdict(gdbg), "page": asdict(pdbg), "read_index_base": int(cfg.read_index_base)}

    # spread ordering: STRICT left -> right
    left_w = float(split_x)
    right_w = float(W - split_x)

    left_items: List[Dict[str, Any]] = []
    right_items: List[Dict[str, Any]] = []
    for it in items:
        x1, y1, x2, y2 = it["bbox"]
        if x_center(it["bbox"]) < split_x:
            b2 = (x1, y1, min(x2, split_x), y2)
            left_items.append({**it, "bbox": b2})
        else:
            b2 = (max(0.0, x1 - split_x), y1, max(0.0, x2 - split_x), y2)
            right_items.append({**it, "bbox": b2})

    left_ordered, ldbg = order_page(
        left_items, left_w, H, max_cols=cfg.max_cols, span_w_frac=cfg.span_w_frac, span_min_h_frac=cfg.span_min_h_frac
    )
    right_ordered, rdbg = order_page(
        right_items, right_w, H, max_cols=cfg.max_cols, span_w_frac=cfg.span_w_frac, span_min_h_frac=cfg.span_min_h_frac
    )

    ordered_items = left_ordered + right_ordered
    ro = [it["_qi"] for it in ordered_items]
    idx_to_rank = {qi: (rank + int(cfg.read_index_base)) for rank, qi in enumerate(ro)}
    for qi, q in enumerate(questions):
        setattr(q, "read_index", idx_to_rank.get(qi))

    return {
        "mode": "spread",
        "split_x": float(split_x),
        "reading_order": ro,
        "gutter": asdict(gdbg),
        "left": asdict(ldbg),
        "right": asdict(rdbg),
        "counts": {"left": len(left_items), "right": len(right_items), "total": len(items)},
        "read_index_base": int(cfg.read_index_base),
    }

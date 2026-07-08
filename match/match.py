from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import shutil
import time
import csv
from statistics import mean

# NOTE: 允许两种启动方式：
#  1) python -m exam_match.match        (推荐)
#  2) python exam_match/match.py        (为了方便调试；需要补上包根目录到 sys.path)
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    __package__ = "exam_match"

from PIL import Image

from .io_rfdetr import load_rfdetr_jsonl, QuestionBox
from .io_doclayout import load_doclayout_json, extract_figures_from_doclayout, FigureBox
from .columns import analyze_columns, assign_col_id_bbox, x_center, bbox_w, bbox_h
from .reading_order import assign_read_index_v9_2, ReadIndexV92Cfg
from .viz import draw_overlay


XYXY = Tuple[float, float, float, float]


def bbox_area(b: XYXY) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter_area(a: XYXY, b: XYXY) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a: XYXY, b: XYXY) -> float:
    inter = inter_area(a, b)
    if inter <= 0.0:
        return 0.0
    ua = bbox_area(a) + bbox_area(b) - inter
    return float(inter / max(1e-9, ua))


def containment_ratio(a: XYXY, b: XYXY) -> float:
    inter = inter_area(a, b)
    if inter <= 0.0:
        return 0.0
    denom = min(bbox_area(a), bbox_area(b))
    return float(inter / max(1e-9, denom))


def _overlaps(a: XYXY, b: XYXY, *, iou_thr: float, contain_thr: float) -> bool:
    if inter_area(a, b) <= 0.0:
        return False
    return (iou_xyxy(a, b) >= float(iou_thr)) or (containment_ratio(a, b) >= float(contain_thr))


def filter_objects_by_overlap(
    objs: List[Any],
    get_bbox,
    *,
    mode: str = "keep",
    iou_thr: float = 0.90,
    contain_thr: float = 0.97,
) -> List[Any]:
    """Subset selection for overlap/containment removal. Deterministic.

    - objs are kept as original object references (no copy).
    - used for matching-stage candidate reduction (does not modify output list unless you assign it).
    """
    mode = (mode or "keep").lower()
    if mode in ("keep", "all", "none") or len(objs) <= 1:
        return objs

    # drop_all
    if mode == "drop_all":
        bad = set()
        bxs = [get_bbox(o) for o in objs]
        for i in range(len(objs)):
            bi = bxs[i]
            for j in range(i + 1, len(objs)):
                bj = bxs[j]
                if _overlaps(bi, bj, iou_thr=iou_thr, contain_thr=contain_thr):
                    bad.add(i)
                    bad.add(j)
        return [o for k, o in enumerate(objs) if k not in bad]

    keep_small = mode in ("keep_small", "small")
    keyed = [(bbox_area(get_bbox(o)), idx, o) for idx, o in enumerate(objs)]
    keyed.sort(key=lambda t: (t[0], -t[1]) if keep_small else (-t[0], t[1]))

    kept = []
    kept_b = []
    for _, _, o in keyed:
        b = get_bbox(o)
        conflict = False
        for kb in kept_b:
            if _overlaps(b, kb, iou_thr=iou_thr, contain_thr=contain_thr):
                conflict = True
                break
        if not conflict:
            kept.append(o)
            kept_b.append(b)

    keep_set = {id(o) for o in kept}
    return [o for o in objs if id(o) in keep_set]


def gap_x(a: XYXY, b: XYXY) -> float:
    if b[2] < a[0]:
        return a[0] - b[2]
    if a[2] < b[0]:
        return b[0] - a[2]
    return 0.0


def gap_y(a: XYXY, b: XYXY) -> float:
    if b[3] < a[1]:
        return a[1] - b[3]
    if a[3] < b[1]:
        return b[1] - a[3]
    return 0.0


def center_dist_norm(a: XYXY, b: XYXY, page_w: float, page_h: float) -> float:
    ax, ay = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
    bx, by = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
    dx = ax - bx
    dy = ay - by
    diag = math.hypot(max(1.0, page_w), max(1.0, page_h))
    return math.hypot(dx, dy) / diag


def clip_xyxy(b: XYXY, w: int, h: int) -> Optional[XYXY]:
    x1, y1, x2, y2 = b
    x1 = max(0.0, min(x1, w - 1))
    y1 = max(0.0, min(y1, h - 1))
    x2 = max(1.0, min(x2, w))
    y2 = max(1.0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def expand_and_clip_int(b: XYXY, w: int, h: int, pad_ratio: float):
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    px, py = bw * pad_ratio, bh * pad_ratio
    nx1 = int(math.floor(x1 - px))
    ny1 = int(math.floor(y1 - py))
    nx2 = int(math.ceil(x2 + px))
    ny2 = int(math.ceil(y2 + py))
    nx1 = max(0, min(nx1, w - 1))
    ny1 = max(0, min(ny1, h - 1))
    nx2 = max(1, min(nx2, w))
    ny2 = max(1, min(ny2, h))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return (nx1, ny1, nx2, ny2)


def xyxy_to_int_clip(b: XYXY, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = b
    nx1 = int(math.floor(x1))
    ny1 = int(math.floor(y1))
    nx2 = int(math.ceil(x2))
    ny2 = int(math.ceil(y2))
    nx1 = max(0, min(nx1, w - 1))
    ny1 = max(0, min(ny1, h - 1))
    nx2 = max(1, min(nx2, w))
    ny2 = max(1, min(ny2, h))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return (nx1, ny1, nx2, ny2)


def save_crop_from_bounds(img: Image.Image, bounds: Tuple[int, int, int, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop = img.crop(bounds)
    crop.save(str(out_path))


def safe_tag(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)


# -------------------------
# NEW: 强制从 pages_root 纠偏原图路径，避免 jsonl 内 image_path 写错导致大量跳页
# -------------------------
def _first_existing_path(cands: List[Path]) -> Optional[str]:
    for p in cands:
        try:
            if p is not None and p.exists() and p.is_file():
                return str(p)
        except Exception:
            continue
    return None


def resolve_page_image_path(page, img_stem: str, pages_root: Optional[Path]) -> Optional[str]:
    """
    返回一个“可用的原图路径”。

    优先级（从高到低）：
      1) 如果指定了 pages_root：用 pages_root + (file_name / image_path 的后缀 / 常见后缀) 去找
      2) 否则：使用 page.image_path（rfdetr_jsonl 里带的）
    """
    # 先走 pages_root（强制纠偏）
    if pages_root is not None:
        pages_root = Path(pages_root)

        # a) 尝试 page.file_name（取 basename，避免它是全路径）
        fn = getattr(page, "file_name", None)
        if isinstance(fn, str) and fn.strip():
            bn = Path(fn).name
            p = pages_root / bn
            if p.exists():
                return str(p)

        # b) 尝试从 page.image_path 推断后缀
        ip = getattr(page, "image_path", None)
        ext = ""
        if isinstance(ip, str) and ip.strip():
            ext = Path(ip).suffix.lower()
        if ext:
            p = pages_root / f"{img_stem}{ext}"
            if p.exists():
                return str(p)

        # c) 常见后缀兜底
        exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]
        cands = [pages_root / f"{img_stem}{e}" for e in exts]
        got = _first_existing_path(cands)
        if got:
            return got

        # d) 最后兜底：glob 同 stem（非递归，避免巨慢）
        try:
            hits = sorted(pages_root.glob(f"{img_stem}.*"))
            got = _first_existing_path(hits)
            if got:
                return got
        except Exception:
            pass

        return None

    # 不指定 pages_root：使用 jsonl 自带 image_path
    ip = getattr(page, "image_path", None)
    if isinstance(ip, str) and ip.strip() and os.path.exists(ip):
        return ip
    return None


def filter_figures(
    figs: List[FigureBox],
    page_w: int,
    page_h: int,
    *,
    area_min_ratio: float,
    aspect_max: float,
) -> List[FigureBox]:
    page_area = max(1.0, float(page_w) * float(page_h))
    out: List[FigureBox] = []
    for f in figs:
        bc = clip_xyxy(f.bbox_xyxy, page_w, page_h)
        if bc is None:
            continue
        f.bbox_xyxy = bc
        a = bbox_area(bc) / page_area
        if a < area_min_ratio:
            continue
        w = bbox_w(bc)
        h = max(0.0, bc[3] - bc[1])
        if w <= 1 or h <= 1:
            continue
        aspect = max(w / h, h / w)
        if aspect > aspect_max:
            continue
        out.append(f)
    return out


def assign_columns(
    questions: List[QuestionBox],
    figures: List[FigureBox],
    page_w: int,
    page_h: int,
    *,
    max_cols: int,
    improve_min: float,
    min_sep_ratio: float,
    min_cluster_frac: float,
    min_cluster_size_abs: int,
    # NEW: 小样本列判定（max-gap）相关
    min_samples_for_cluster: int,
    gap_min_ratio: float,
    edge_margin_ratio: float,
    min_samples_for_gap: int,
    gap_min_cluster_size: int,
    gap_over_median_min: float,
    # NEW: 防止单栏误判
    wide_exclude_w_frac: float,
    boundary_tol_ratio: float,
    min_side_conf_frac: float,
    max_amb_frac: float,
    # NEW: figure 跨列判定
    fig_span_w_frac: float,
    fig_span_margin_ratio: float,
):
    q_bboxes = [q.bbox_xyxy for q in questions]
    colinfo = analyze_columns(
        q_bboxes,
        page_w=float(page_w),
        page_h=float(page_h),
        max_cols=max_cols,
        feature="x1",
        improve_min=improve_min,
        min_sep_ratio=min_sep_ratio,
        min_cluster_frac=min_cluster_frac,
        min_cluster_size_abs=min_cluster_size_abs,
        min_samples_for_cluster=min_samples_for_cluster,
        gap_min_ratio=gap_min_ratio,
        edge_margin_ratio=edge_margin_ratio,
        min_samples_for_gap=min_samples_for_gap,
        gap_min_cluster_size=gap_min_cluster_size,
        gap_over_median_min=gap_over_median_min,
        wide_exclude_w_frac=wide_exclude_w_frac,
        boundary_tol_ratio=boundary_tol_ratio,
        min_side_conf_frac=min_side_conf_frac,
        max_amb_frac=max_amb_frac,
    )
    split_x = colinfo.split_x
    num_cols = colinfo.num_cols

    for q in questions:
        q.col_id = assign_col_id_bbox(q.bbox_xyxy, split_x, page_w=float(page_w), boundary_tol_ratio=boundary_tol_ratio)
        q.is_spanning = (q.col_id == -1)

    margin = float(fig_span_margin_ratio) * float(page_w)
    for f in figures:
        f.col_id = assign_col_id_bbox(f.bbox_xyxy, split_x, page_w=float(page_w), boundary_tol_ratio=boundary_tol_ratio) if num_cols > 1 else 0
        w_frac = bbox_w(f.bbox_xyxy) / max(1.0, float(page_w))

        cross_split = False
        if num_cols > 1 and split_x:
            for sx in split_x:
                if f.bbox_xyxy[0] < (sx - margin) and f.bbox_xyxy[2] > (sx + margin):
                    cross_split = True
                    break

        f.is_spanning = bool((f.col_id == -1) or (w_frac >= float(fig_span_w_frac)) or cross_split)

    return num_cols, split_x, colinfo.to_jsonable()


def score_pair(q: QuestionBox, f: FigureBox, *, page_w: int, page_h: int) -> Dict[str, float]:
    ia = inter_area(q.bbox_xyxy, f.bbox_xyxy)
    af = bbox_area(f.bbox_xyxy)
    aq = bbox_area(q.bbox_xyxy)
    iof = ia / max(1e-6, af)
    ioq = ia / max(1e-6, aq)
    gx = gap_x(q.bbox_xyxy, f.bbox_xyxy) / max(1.0, page_w)
    gy = gap_y(q.bbox_xyxy, f.bbox_xyxy) / max(1.0, page_h)
    cd = center_dist_norm(q.bbox_xyxy, f.bbox_xyxy, page_w, page_h)

    fx, fy = x_center(f.bbox_xyxy), (f.bbox_xyxy[1] + f.bbox_xyxy[3]) * 0.5
    inside = 1.0 if (q.bbox_xyxy[0] <= fx <= q.bbox_xyxy[2] and q.bbox_xyxy[1] <= fy <= q.bbox_xyxy[3]) else 0.0

    score = (
        2.6 * iof +
        0.7 * ioq +
        0.3 * inside -
        1.2 * gx -
        0.9 * gy -
        0.3 * cd
    )
    if iof >= 0.55:
        score += 0.6

    return {"score": float(score), "iof": float(iof), "ioq": float(ioq), "gx": float(gx), "gy": float(gy), "cd": float(cd), "inside": float(inside)}


def match_figures_for_questions(
    questions: List[QuestionBox],
    figures: List[FigureBox],
    *,
    page_w: int,
    page_h: int,
    num_cols: int,
    mode: str = "unique",
    min_pair_score: float = 0.30,
    max_fig_per_question: int = 2,
    max_q_per_fig: int = 1,
    max_q_per_spanning_fig: int = 2,
    x_gap_max: float = 0.12,
    y_gap_max: float = 0.14,
    share_overlap_min: float = 0.20,
) -> Dict[int, List[Dict[str, Any]]]:
    mode = (mode or "unique").lower().strip()
    if mode not in ("unique", "share"):
        raise ValueError(f"mode 必须是 unique 或 share，但得到：{mode}")

    pairs: List[Tuple[float, int, int, Dict[str, float]]] = []
    for qi, q in enumerate(questions):
        for fi, f in enumerate(figures):
            if num_cols > 1 and (not f.is_spanning):
                qc = q.col_id
                fc = f.col_id
                if (qc is not None and int(qc) != -1) and (fc is not None and int(fc) != -1):
                    if int(qc) != int(fc):
                        continue

            comp = score_pair(q, f, page_w=page_w, page_h=page_h)
            s = comp["score"]
            if s < min_pair_score:
                continue

            if comp["iof"] < 0.10:
                if comp["gx"] > x_gap_max or comp["gy"] > y_gap_max:
                    continue

            pairs.append((s, qi, fi, comp))

    if not pairs:
        return {}

    if mode == "share":
        out: Dict[int, List[Dict[str, Any]]] = {}
        per_q: Dict[int, List[Tuple[float, int, Dict[str, float]]]] = {}
        for s, qi, fi, comp in pairs:
            per_q.setdefault(qi, []).append((s, fi, comp))
        for qi, lst in per_q.items():
            lst.sort(key=lambda t: t[0], reverse=True)
            if max_fig_per_question > 0:
                lst = lst[:max_fig_per_question]
            q_det = questions[qi].det_index
            out[q_det] = [{"figure_index": figures[fi].fig_index, "score": float(s), **comp} for s, fi, comp in lst]
        return out

    pairs.sort(key=lambda t: t[0], reverse=True)
    fig_assigned_cnt = [0] * len(figures)
    q_assigned_cnt = [0] * len(questions)

    out: Dict[int, List[Dict[str, Any]]] = {}
    for s, qi, fi, comp in pairs:
        if max_fig_per_question > 0 and q_assigned_cnt[qi] >= max_fig_per_question:
            continue

        f = figures[fi]
        q = questions[qi]
        limit = max_q_per_spanning_fig if f.is_spanning else max_q_per_fig
        if fig_assigned_cnt[fi] >= limit:
            continue

        if fig_assigned_cnt[fi] > 0:
            if (not f.is_spanning) and comp["iof"] < share_overlap_min:
                continue

        q_det = q.det_index
        out.setdefault(q_det, []).append({"figure_index": f.fig_index, "score": float(s), **comp})
        fig_assigned_cnt[fi] += 1
        q_assigned_cnt[qi] += 1

    for det, lst in list(out.items()):
        lst.sort(key=lambda d: d["score"], reverse=True)
        if max_fig_per_question > 0:
            out[det] = lst[:max_fig_per_question]
    return out


# -------------------------
# Better Q↔Fig matching (v2): RO-window + spread/segment aware + min-cost flow
# -------------------------
class _MinCostMaxFlow:
    """Successive Shortest Augmenting Path (SSAP) min-cost max-flow with potentials.

    Small graphs only (our use-case: per-page dozens of nodes), pure Python, no deps.
    Costs are integers (can be negative); we add a Bellman-Ford init for potentials.
    """
    class _Edge:
        __slots__ = ("to", "rev", "cap", "cost")
        def __init__(self, to: int, rev: int, cap: int, cost: int):
            self.to = to
            self.rev = rev
            self.cap = cap
            self.cost = cost

    def __init__(self, n: int):
        self.n = int(n)
        self.g: List[List[_MinCostMaxFlow._Edge]] = [[] for _ in range(self.n)]

    def add_edge(self, fr: int, to: int, cap: int, cost: int) -> None:
        fr = int(fr); to = int(to); cap = int(cap); cost = int(cost)
        fwd = _MinCostMaxFlow._Edge(to, len(self.g[to]), cap, cost)
        rev = _MinCostMaxFlow._Edge(fr, len(self.g[fr]), 0, -cost)
        self.g[fr].append(fwd)
        self.g[to].append(rev)

    def min_cost_flow(self, s: int, t: int, maxf: int) -> Tuple[int, int]:
        import heapq

        n = self.n
        s = int(s); t = int(t); maxf = int(maxf)

        pot = [0] * n
        INF = 10**18
        dist = [INF] * n
        dist[s] = 0
        for _ in range(n - 1):
            updated = False
            for u in range(n):
                if dist[u] >= INF:
                    continue
                du = dist[u]
                for e in self.g[u]:
                    if e.cap <= 0:
                        continue
                    nd = du + e.cost
                    if nd < dist[e.to]:
                        dist[e.to] = nd
                        updated = True
            if not updated:
                break
        for i in range(n):
            if dist[i] < INF:
                pot[i] = dist[i]

        flow = 0
        cost = 0
        prevv = [0] * n
        preve = [0] * n

        while flow < maxf:
            dist = [INF] * n
            dist[s] = 0
            pq = [(0, s)]
            while pq:
                d, v = heapq.heappop(pq)
                if d != dist[v]:
                    continue
                for i, e in enumerate(self.g[v]):
                    if e.cap <= 0:
                        continue
                    nd = d + e.cost + pot[v] - pot[e.to]
                    if nd < dist[e.to]:
                        dist[e.to] = nd
                        prevv[e.to] = v
                        preve[e.to] = i
                        heapq.heappush(pq, (nd, e.to))

            if dist[t] >= INF:
                break

            for v in range(n):
                if dist[v] < INF:
                    pot[v] += dist[v]

            addf = maxf - flow
            v = t
            while v != s:
                u = prevv[v]
                e = self.g[u][preve[v]]
                if e.cap < addf:
                    addf = e.cap
                v = u

            v = t
            while v != s:
                u = prevv[v]
                e = self.g[u][preve[v]]
                e.cap -= addf
                self.g[v][e.rev].cap += addf
                cost += addf * e.cost
                v = u

            flow += addf

        return flow, cost


def _is_span_like(b: XYXY, page_w: float, page_h: float, *, span_w_frac: float, span_min_h_frac: float) -> bool:
    return (bbox_w(b) / max(1.0, float(page_w))) >= float(span_w_frac) and (bbox_h(b) / max(1.0, float(page_h))) >= float(span_min_h_frac)


def _side_id(b: XYXY, split_x: Optional[float], *, is_spanning: bool, margin: float = 0.0) -> int:
    """0: left/single, 1: right, -1: cross/unknown."""
    if split_x is None:
        return 0
    if is_spanning:
        return -1
    xc = x_center(b)
    if abs(xc - float(split_x)) <= margin:
        return -1
    return 0 if xc < float(split_x) else 1


def _seg_id(yc: float, cuts: List[float]) -> int:
    lo, hi = 0, len(cuts)
    while lo < hi:
        mid = (lo + hi) // 2
        if yc < cuts[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def match_figures_for_questions_v2(
    questions: List[QuestionBox],
    figures: List[FigureBox],
    *,
    page_w: int,
    page_h: int,
    num_cols: int,
    ro_debug: Optional[Dict[str, Any]] = None,
    span_w_frac: float = 0.86,
    span_min_h_frac: float = 0.030,
    mode: str = "unique",
    algo: str = "flow",
    min_pair_score: float = 0.25,
    max_fig_per_question: int = 2,
    max_q_per_fig: int = 1,
    max_q_per_spanning_fig: int = 2,
    pre_margin_y: float = 0.015,
    post_margin_y: float = 0.020,
    seg_mismatch_penalty: float = 0.35,
    out_window_penalty: float = 0.45,
    in_window_bonus: float = 0.40,
) -> Dict[int, List[Dict[str, Any]]]:
    mode = (mode or "unique").lower().strip()
    algo = (algo or "flow").lower().strip()
    if mode not in ("unique", "share"):
        raise ValueError(f"mode 必须是 unique 或 share，但得到：{mode}")
    if algo not in ("flow", "greedy"):
        raise ValueError(f"algo 必须是 flow 或 greedy，但得到：{algo}")

    split_x = None
    if isinstance(ro_debug, dict) and ro_debug.get("mode") == "spread" and ro_debug.get("split_x") is not None:
        try:
            split_x = float(ro_debug["split_x"])
        except Exception:
            split_x = None

    H = float(max(1, page_h))
    W = float(max(1, page_w))
    side_margin = 0.02 * W

    for q in questions:
        qb = q.bbox_xyxy
        q._xc = x_center(qb)
        q._yc = (qb[1] + qb[3]) * 0.5
        q._span_like = _is_span_like(qb, W, H, span_w_frac=span_w_frac, span_min_h_frac=span_min_h_frac)
        q._side = _side_id(qb, split_x, is_spanning=q._span_like or (getattr(q, "col_id", None) == -1), margin=side_margin)

    cuts_by_side: Dict[int, List[float]] = {0: [], 1: []}
    for q in questions:
        if not getattr(q, "_span_like", False):
            continue
        if q._side in (0, 1):
            cuts_by_side[q._side].append(float(q.bbox_xyxy[1]))
        else:
            cuts_by_side[0].append(float(q.bbox_xyxy[1]))
            cuts_by_side[1].append(float(q.bbox_xyxy[1]))
    for k in (0, 1):
        cuts = sorted(set(cuts_by_side[k]))
        filtered = []
        for y in cuts:
            if not filtered or abs(y - filtered[-1]) > 6.0:
                filtered.append(y)
        cuts_by_side[k] = filtered

    for q in questions:
        side = q._side if q._side in (0, 1) else 0
        q._seg = _seg_id(float(q._yc), cuts_by_side[side])

    order = list(range(len(questions)))
    if all(getattr(questions[i], "read_index", None) is not None for i in order):
        order.sort(key=lambda i: int(getattr(questions[i], "read_index")))
    else:
        order.sort(key=lambda i: (questions[i].bbox_xyxy[1], questions[i].bbox_xyxy[0]))

    next_y1 = [float(page_h)] * len(questions)
    for pos, qi in enumerate(order):
        q = questions[qi]
        side = q._side if q._side in (0, 1) else 0
        y1 = float(q.bbox_xyxy[1])

        seg_end = float(page_h)
        for cut in cuts_by_side[side]:
            if cut > y1 + 1e-3:
                seg_end = cut
                break

        ny = float(page_h)
        for pj in order[pos + 1:]:
            q2 = questions[pj]
            if split_x is not None and (q._side in (0, 1)) and (q2._side in (0, 1)) and (q2._side != q._side):
                continue
            ny = float(q2.bbox_xyxy[1])
            break

        next_y1[qi] = min(ny, seg_end)

    for f in figures:
        fb = f.bbox_xyxy
        f._xc = x_center(fb)
        f._yc = (fb[1] + fb[3]) * 0.5
        f._side = _side_id(fb, split_x, is_spanning=bool(f.is_spanning), margin=side_margin)
        if f._side in (0, 1):
            f._seg = _seg_id(float(f._yc), cuts_by_side[f._side])
        else:
            f._seg = -1

    def _pair_score(qi: int, fi: int) -> Optional[Tuple[float, Dict[str, float]]]:
        q = questions[qi]
        f = figures[fi]

        if split_x is not None and (not f.is_spanning):
            if (q._side in (0, 1)) and (f._side in (0, 1)) and (q._side != f._side):
                return None

        if num_cols > 1 and (not f.is_spanning):
            qc = getattr(q, "col_id", None)
            fc = getattr(f, "col_id", None)
            if (qc is not None and int(qc) != -1) and (fc is not None and int(fc) != -1) and int(qc) != int(fc):
                return None

        comp = score_pair(q, f, page_w=page_w, page_h=page_h)
        s = float(comp["score"])

        if (not f.is_spanning) and (q._side in (0, 1)) and (f._side in (0, 1)):
            if int(getattr(q, "_seg", 0)) != int(getattr(f, "_seg", 0)):
                s -= float(seg_mismatch_penalty)
                comp["seg_mismatch"] = 1.0
            else:
                comp["seg_mismatch"] = 0.0

        win_top = float(q.bbox_xyxy[1]) - float(pre_margin_y) * H
        win_bot = float(next_y1[qi]) + float(post_margin_y) * H
        fyc = float(getattr(f, "_yc", (f.bbox_xyxy[1] + f.bbox_xyxy[3]) * 0.5))
        in_window = 1.0 if (win_top <= fyc <= win_bot) else 0.0
        if comp["iof"] >= 0.05 or comp["ioq"] >= 0.05:
            in_window = 1.0

        comp["in_window"] = float(in_window)
        comp["win_top"] = float(win_top / H)
        comp["win_bot"] = float(win_bot / H)

        if in_window >= 0.5:
            s += float(in_window_bonus)
        else:
            s -= float(out_window_penalty)

        comp["score_v2"] = float(s)
        return s, comp

    pairs: List[Tuple[float, int, int, Dict[str, float]]] = []
    for qi in range(len(questions)):
        for fi in range(len(figures)):
            got = _pair_score(qi, fi)
            if got is None:
                continue
            s, comp = got
            if s < float(min_pair_score):
                continue
            pairs.append((float(s), qi, fi, comp))

    if not pairs:
        return {}

    if mode == "share":
        per_q: Dict[int, List[Tuple[float, int, Dict[str, float]]]] = {}
        for s, qi, fi, comp in pairs:
            per_q.setdefault(qi, []).append((s, fi, comp))
        out: Dict[int, List[Dict[str, Any]]] = {}
        for qi, lst in per_q.items():
            lst.sort(key=lambda t: t[0], reverse=True)
            if max_fig_per_question > 0:
                lst = lst[:max_fig_per_question]
            q_det = questions[qi].det_index
            out[q_det] = [{"figure_index": figures[fi].fig_index, "score": float(s), **comp} for s, fi, comp in lst]
        return out

    if algo == "greedy":
        pairs.sort(key=lambda t: t[0], reverse=True)
        fig_cnt = [0] * len(figures)
        q_cnt = [0] * len(questions)
        out: Dict[int, List[Dict[str, Any]]] = {}
        for s, qi, fi, comp in pairs:
            if max_fig_per_question > 0 and q_cnt[qi] >= max_fig_per_question:
                continue
            limit = max_q_per_spanning_fig if figures[fi].is_spanning else max_q_per_fig
            if fig_cnt[fi] >= limit:
                continue
            q_det = questions[qi].det_index
            out.setdefault(q_det, []).append({"figure_index": figures[fi].fig_index, "score": float(s), **comp})
            fig_cnt[fi] += 1
            q_cnt[qi] += 1
        for det, lst in list(out.items()):
            lst.sort(key=lambda d: d["score"], reverse=True)
            if max_fig_per_question > 0:
                out[det] = lst[:max_fig_per_question]
        return out

    F = len(figures)
    Q = len(questions)
    S = 0
    T = 1
    base_f = 2
    base_q = base_f + F
    n_nodes = base_q + Q
    mcmf = _MinCostMaxFlow(n_nodes)

    total_supply = 0
    fig_limit = []
    for fi, f in enumerate(figures):
        lim = int(max_q_per_spanning_fig if f.is_spanning else max_q_per_fig)
        lim = max(0, lim)
        fig_limit.append(lim)
        if lim > 0:
            mcmf.add_edge(S, base_f + fi, lim, 0)
            mcmf.add_edge(base_f + fi, T, lim, 0)
            total_supply += lim

    for qi in range(Q):
        cap = int(max_fig_per_question)
        cap = max(0, cap)
        if cap > 0:
            mcmf.add_edge(base_q + qi, T, cap, 0)

    SCALE = 1000
    pair_map: Dict[Tuple[int, int], Dict[str, float]] = {}
    for s, qi, fi, comp in pairs:
        if fig_limit[fi] <= 0:
            continue
        cost = int(round(-float(s) * SCALE))
        mcmf.add_edge(base_f + fi, base_q + qi, 1, cost)
        pair_map[(fi, qi)] = comp

    mcmf.min_cost_flow(S, T, total_supply)

    out: Dict[int, List[Dict[str, Any]]] = {}
    for fi in range(F):
        u = base_f + fi
        for e in mcmf.g[u]:
            if not (base_q <= e.to < base_q + Q):
                continue
            qi = e.to - base_q
            rev = mcmf.g[e.to][e.rev]
            flowed = rev.cap
            if flowed <= 0:
                continue
            q_det = questions[qi].det_index
            comp = pair_map.get((fi, qi), {})
            out.setdefault(q_det, []).append({"figure_index": figures[fi].fig_index, "score": float(-e.cost / SCALE), **comp})

    for det, lst in list(out.items()):
        lst.sort(key=lambda d: d.get("score", -1), reverse=True)
        if max_fig_per_question > 0:
            out[det] = lst[:max_fig_per_question]
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("RF-DETR + DocLayout match (columns + reading order + optional matching)")
    ap.add_argument("--rfdetr-jsonl", type=str, required=True)
    ap.add_argument("--doclayout-json-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)

    # NEW
    ap.add_argument(
        "--pages-root",
        type=str,
        default="",
        help="原图根目录（例如 /.../workflow/Rangle_image5）。指定后将优先从该目录拼出原图路径，避免 rfdetr_jsonl 里的 image_path 写错导致大量跳页。",
    )

    ap.add_argument("--question-classes", type=str, default="multiple_choice_question,fill_blank_question,problem_solving_question,partial_question")
    ap.add_argument("--min-question-score", type=float, default=0.35)

    ap.add_argument("--image-labels", type=str, default="image,figure,pic")
    ap.add_argument("--min-figure-score", type=float, default=0.0)

    ap.add_argument("--area-min-ratio", type=float, default=0.004)
    ap.add_argument("--aspect-max", type=float, default=10.0)

    ap.add_argument("--fig-pad-ratio", type=float, default=0.02)

    ap.add_argument("--fig-span-w-frac", type=float, default=0.55)
    ap.add_argument("--fig-span-margin-ratio", type=float, default=0.02)

    ap.add_argument("--max-cols", type=int, default=3)
    ap.add_argument("--col-improve-min", type=float, default=0.22)
    ap.add_argument("--col-min-sep-ratio", type=float, default=0.16)
    ap.add_argument("--col-min-cluster-frac", type=float, default=0.12)
    ap.add_argument("--col-min-cluster-size-abs", type=int, default=2)

    ap.add_argument("--col-min-samples-cluster", type=int, default=8)
    ap.add_argument("--col-min-samples-gap", type=int, default=4)
    ap.add_argument("--col-gap-min-ratio", type=float, default=0.18)
    ap.add_argument("--col-edge-margin-ratio", type=float, default=0.08)
    ap.add_argument("--col-gap-min-cluster-size", type=int, default=2)
    ap.add_argument("--col-gap-over-median-min", type=float, default=2.2)

    ap.add_argument("--col-wide-exclude-frac", type=float, default=0.78)
    ap.add_argument("--col-boundary-tol-ratio", type=float, default=0.02)
    ap.add_argument("--col-min-side-conf-frac", type=float, default=0.60)
    ap.add_argument("--col-max-amb-frac", type=float, default=0.45)

    ap.add_argument("--ro-overlap-mode", type=str, default="keep", choices=["keep", "keep_large", "keep_small", "drop_all"])
    ap.add_argument("--ro-overlap-iou", type=float, default=0.92)
    ap.add_argument("--ro-overlap-contain", type=float, default=0.95)

    ap.add_argument("--match-overlap-mode", type=str, default="keep", choices=["keep", "keep_large", "keep_small", "drop_all"])
    ap.add_argument("--match-overlap-iou", type=float, default=0.90)
    ap.add_argument("--match-overlap-contain", type=float, default=0.97)

    ap.add_argument("--match-mode", type=str, default="unique", choices=["unique", "share"])
    ap.add_argument("--match-algo", type=str, default="v2", choices=["v1", "v2"], help="Q↔Fig 匹配算法：v1=旧版局部贪心；v2=RO-window+spread/segment+全局分配")
    ap.add_argument("--match-backend", type=str, default="flow", choices=["flow", "greedy"], help="v2 的后端：flow=最小费用流；greedy=贪心(快)")
    ap.add_argument("--match-pre-margin-y", type=float, default=0.015, help="v2: 窗口上边界预留(占H比例)")
    ap.add_argument("--match-post-margin-y", type=float, default=0.020, help="v2: 窗口下边界预留(占H比例)")
    ap.add_argument("--match-seg-mismatch-penalty", type=float, default=0.35, help="v2: 跨span段匹配惩罚")
    ap.add_argument("--match-out-window-penalty", type=float, default=0.45, help="v2: 不在窗口内惩罚")
    ap.add_argument("--match-in-window-bonus", type=float, default=0.40, help="v2: 在窗口内奖励")
    ap.add_argument("--min-pair-score", type=float, default=0.30)
    ap.add_argument("--max-fig-per-question", type=int, default=2)
    ap.add_argument("--max-q-per-fig", type=int, default=1)
    ap.add_argument("--max-q-per-spanning-fig", type=int, default=2)
    ap.add_argument("--x-gap-max", type=float, default=0.12)
    ap.add_argument("--y-gap-max", type=float, default=0.14)
    ap.add_argument("--share-overlap-min", type=float, default=0.20)

    ap.add_argument("--save-viz", action="store_true")
    ap.add_argument("--draw-edges", action="store_true")
    ap.add_argument("--max-edges-per-question", type=int, default=1)

    ap.add_argument("--q-pad-ratio", type=float, default=0.02, help="题框crop的pad比例")
    ap.add_argument("--measure-latency", action="store_true",
                    help="统计每页各阶段耗时(毫秒)并在最后输出汇总；若设置 --latency-json/--latency-csv 也会自动启用")
    ap.add_argument("--latency-json", type=str, default="",
                    help="可选：把 per-page 耗时写成 JSON（相对路径默认写到 output_dir 下）")
    ap.add_argument("--latency-csv", type=str, default="",
                    help="可选：把 per-page 耗时写成 CSV（相对路径默认写到 output_dir 下）")

    return ap.parse_args()



def _resolve_out_path(base_dir: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base_dir / pp)


def _percentile(sorted_vals: List[float], q: float) -> float:
    """q in [0,1]. Linear interpolation between closest ranks."""
    if not sorted_vals:
        return 0.0
    q = float(q)
    if q <= 0.0:
        return float(sorted_vals[0])
    if q >= 1.0:
        return float(sorted_vals[-1])
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    w = pos - lo
    return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)

def main() -> None:
    args = parse_args()
    do_latency = bool(getattr(args, 'measure_latency', False) or getattr(args, 'latency_json', '') or getattr(args, 'latency_csv', ''))
    latency_rows: List[Dict[str, Any]] = []

    pages_root: Optional[Path] = None
    if isinstance(args.pages_root, str) and args.pages_root.strip():
        pages_root = Path(args.pages_root).expanduser()
        if not pages_root.exists() or not pages_root.is_dir():
            raise FileNotFoundError(f"--pages-root 不存在或不是目录：{pages_root}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_classes = [s.strip() for s in args.question_classes.split(",") if s.strip()]
    img_labels = [s.strip() for s in args.image_labels.split(",") if s.strip()]

    pages = load_rfdetr_jsonl(args.rfdetr_jsonl, allowed_class_names=q_classes, min_score=args.min_question_score)

    doc_dir = Path(args.doclayout_json_dir)
    json_files = sorted(doc_dir.glob("*.json"))
    if not json_files:
        json_subdir = doc_dir / "json"
        if json_subdir.exists() and json_subdir.is_dir():
            json_files = sorted(json_subdir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"在 {doc_dir} 及其 json/ 子目录下都没有找到任何 .json")

    for jpath in json_files:
        # latency profile (per page)
        _lat: Dict[str, Any] = {}
        _t_page0 = time.perf_counter() if do_latency else 0.0
        raw, img_stem = load_doclayout_json(str(jpath))
        if do_latency:
            _lat['image_stem'] = img_stem
        page = pages.get(img_stem)
        if page is None:
            print(f"[WARN] RF-DETR jsonl 中找不到 {img_stem}，跳过。", file=sys.stderr)
            continue

        # NEW: 用 pages_root 纠偏
        img_path = resolve_page_image_path(page, img_stem, pages_root)
        if not img_path or not os.path.exists(img_path):
            print(
                f"[WARN] 原图不存在：resolved={img_path} (jsonl_image_path={getattr(page, 'image_path', None)})，跳过该页。",
                file=sys.stderr
            )
            continue

        page.image_path = img_path

        _t_img0 = time.perf_counter() if do_latency else 0.0
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] 打开原图失败：{img_path} ({e})，跳过该页。", file=sys.stderr)
            continue

        if do_latency:
            _lat['open_image_ms'] = (time.perf_counter() - _t_img0) * 1000.0

        w, h = img.size
        page.width, page.height = w, h

        _t0 = time.perf_counter() if do_latency else 0.0
        figs_all = extract_figures_from_doclayout(raw, allowed_labels=img_labels, min_score=args.min_figure_score)
        figs = filter_figures(figs_all, page_w=w, page_h=h, area_min_ratio=args.area_min_ratio, aspect_max=args.aspect_max)
        if do_latency:
            _lat['extract_figures_ms'] = (time.perf_counter() - _t0) * 1000.0

        _t0 = time.perf_counter() if do_latency else 0.0
        num_cols, split_x, col_debug = assign_columns(
            page.questions,
            figs,
            page_w=w,
            page_h=h,
            max_cols=args.max_cols,
            improve_min=args.col_improve_min,
            min_sep_ratio=args.col_min_sep_ratio,
            min_cluster_frac=args.col_min_cluster_frac,
            min_cluster_size_abs=args.col_min_cluster_size_abs,
            min_samples_for_cluster=args.col_min_samples_cluster,
            gap_min_ratio=args.col_gap_min_ratio,
            edge_margin_ratio=args.col_edge_margin_ratio,
            min_samples_for_gap=args.col_min_samples_gap,
            gap_min_cluster_size=args.col_gap_min_cluster_size,
            gap_over_median_min=args.col_gap_over_median_min,
            wide_exclude_w_frac=args.col_wide_exclude_frac,
            boundary_tol_ratio=args.col_boundary_tol_ratio,
            min_side_conf_frac=args.col_min_side_conf_frac,
            max_amb_frac=args.col_max_amb_frac,
            fig_span_w_frac=args.fig_span_w_frac,
            fig_span_margin_ratio=args.fig_span_margin_ratio,
        )
        if do_latency:
            _lat['assign_columns_ms'] = (time.perf_counter() - _t0) * 1000.0

        ro_cfg = ReadIndexV92Cfg(
            max_cols=args.max_cols,
            read_index_base=1,
            overlap_mode=args.ro_overlap_mode,
            overlap_iou_thr=args.ro_overlap_iou,
            overlap_contain_thr=args.ro_overlap_contain,
        )
        _t0 = time.perf_counter() if do_latency else 0.0
        ro_debug = assign_read_index_v9_2(page.questions, page_w=w, page_h=h, cfg=ro_cfg)
        if do_latency:
            _lat['reading_order_ms'] = (time.perf_counter() - _t0) * 1000.0

        _t0 = time.perf_counter() if do_latency else 0.0
        q_for_match = filter_objects_by_overlap(
            page.questions,
            lambda q: q.bbox_xyxy,
            mode=args.match_overlap_mode,
            iou_thr=args.match_overlap_iou,
            contain_thr=args.match_overlap_contain,
        )
        f_for_match = filter_objects_by_overlap(
            figs,
            lambda f: f.bbox_xyxy,
            mode=args.match_overlap_mode,
            iou_thr=args.match_overlap_iou,
            contain_thr=args.match_overlap_contain,
        )
        if do_latency:
            _lat['match_prep_ms'] = (time.perf_counter() - _t0) * 1000.0

        _t0 = time.perf_counter() if do_latency else 0.0
        if args.match_algo == "v1":
            q2m = match_figures_for_questions(
                q_for_match,
                f_for_match,
                page_w=w,
                page_h=h,
                num_cols=num_cols,
                mode=args.match_mode,
                min_pair_score=args.min_pair_score,
                max_fig_per_question=args.max_fig_per_question,
                max_q_per_fig=args.max_q_per_fig,
                max_q_per_spanning_fig=args.max_q_per_spanning_fig,
                x_gap_max=args.x_gap_max,
                y_gap_max=args.y_gap_max,
                share_overlap_min=args.share_overlap_min,
            )
        else:
            q2m = match_figures_for_questions_v2(
                q_for_match,
                f_for_match,
                page_w=w,
                page_h=h,
                num_cols=num_cols,
                ro_debug=ro_debug if isinstance(ro_debug, dict) else None,
                span_w_frac=ro_cfg.span_w_frac,
                span_min_h_frac=ro_cfg.span_min_h_frac,
                mode=args.match_mode,
                algo=args.match_backend,
                min_pair_score=args.min_pair_score,
                max_fig_per_question=args.max_fig_per_question,
                max_q_per_fig=args.max_q_per_fig,
                max_q_per_spanning_fig=args.max_q_per_spanning_fig,
                pre_margin_y=args.match_pre_margin_y,
                post_margin_y=args.match_post_margin_y,
                seg_mismatch_penalty=args.match_seg_mismatch_penalty,
                out_window_penalty=args.match_out_window_penalty,
                in_window_bonus=args.match_in_window_bonus,
            )

        if do_latency:
            _lat['match_ms'] = (time.perf_counter() - _t0) * 1000.0
            _lat['num_questions'] = int(len(getattr(page, 'questions', []) or []))
            _lat['num_figures'] = int(len(figs))
            try:
                _lat['num_matches'] = int(sum(len(v or []) for v in (q2m or {}).values()))
            except Exception:
                _lat['num_matches'] = 0

        _t0 = time.perf_counter() if do_latency else 0.0
        page_dir = out_dir / img_stem
        if page_dir.exists():
            shutil.rmtree(page_dir)
        page_dir.mkdir(parents=True, exist_ok=True)

        questions_root = page_dir / "questions"
        questions_root.mkdir(parents=True, exist_ok=True)

        viz_root = page_dir / "viz"
        if args.save_viz:
            viz_root.mkdir(parents=True, exist_ok=True)

        fig_by_index = {f.fig_index: f for f in figs}
        if do_latency:
            _lat['io_prep_ms'] = (time.perf_counter() - _t0) * 1000.0


        def _q_sort_key(q: QuestionBox):
            ri = q.read_index if q.read_index is not None else 10**9
            return (int(ri), float(q.bbox_xyxy[1]), float(q.bbox_xyxy[0]), int(q.det_index))

        _t0 = time.perf_counter() if do_latency else 0.0
        sorted_questions = sorted(page.questions, key=_q_sort_key)

        q_crop_path_by_det: Dict[int, Optional[str]] = {}
        q_matches_aug: Dict[int, List[Dict[str, Any]]] = {}

        for q in sorted_questions:
            ri = q.read_index if q.read_index is not None else 999999
            q_dir = questions_root / f"q{int(ri):04d}_det{int(q.det_index):03d}"
            q_dir.mkdir(parents=True, exist_ok=True)

            qb = None
            if q.bbox_xyxy_padded is not None:
                qb = xyxy_to_int_clip(q.bbox_xyxy_padded, w, h)
            if qb is None:
                qb = expand_and_clip_int(q.bbox_xyxy, w=w, h=h, pad_ratio=float(getattr(args, "q_pad_ratio", 0.02)))

            if qb is not None:
                q_out = q_dir / "question.png"
                save_crop_from_bounds(img, qb, q_out)
                q_crop_path_by_det[q.det_index] = str(q_out.relative_to(page_dir))
            else:
                q_crop_path_by_det[q.det_index] = None

            matches = q2m.get(q.det_index, []) or []
            aug_list: List[Dict[str, Any]] = []

            for m in sorted(matches, key=lambda d: d.get("score", -1), reverse=True):
                fig_idx = int(m.get("figure_index"))
                f = fig_by_index.get(fig_idx)
                if f is None:
                    continue

                fb = expand_and_clip_int(f.bbox_xyxy, w=w, h=h, pad_ratio=float(args.fig_pad_ratio))
                if fb is None:
                    continue

                score = float(m.get("score", f.score))
                name = f"figure_{fig_idx:02d}_{safe_tag(f.label)}_s{score:.2f}.png"
                outp = q_dir / name
                save_crop_from_bounds(img, fb, outp)

                mm = dict(m)
                mm["crop_path"] = str(outp.relative_to(page_dir))
                aug_list.append(mm)

            q_matches_aug[q.det_index] = aug_list

        if do_latency:
            _lat['save_crops_ms'] = (time.perf_counter() - _t0) * 1000.0

        _t0 = time.perf_counter() if do_latency else 0.0
        out_questions: List[Dict[str, Any]] = []
        for q in page.questions:
            out_questions.append(
                {
                    "det_index": q.det_index,
                    "read_index": q.read_index,
                    "col_id": q.col_id,
                    "is_spanning": getattr(q, "is_spanning", (q.col_id == -1)),
                    "bbox_xyxy": list(q.bbox_xyxy),
                    "bbox_xyxy_padded": list(q.bbox_xyxy_padded) if q.bbox_xyxy_padded else None,
                    "class_id": q.class_id,
                    "class_name": q.class_name,
                    "score": q.score,
                    "crop_path": q_crop_path_by_det.get(q.det_index),
                    "matches": q_matches_aug.get(q.det_index, []),
                }
            )

        out_figures: List[Dict[str, Any]] = []
        for f in figs:
            out_figures.append(
                {
                    "fig_index": f.fig_index,
                    "col_id": f.col_id,
                    "is_spanning": f.is_spanning,
                    "label": f.label,
                    "score": f.score,
                    "bbox_xyxy": list(f.bbox_xyxy),
                    "bbox_xyxy_padded": list(f.bbox_xyxy_padded) if f.bbox_xyxy_padded else None,
                    "crop_path": getattr(f, "crop_path", None),
                }
            )

        out_rec = {
            "image_stem": img_stem,
            "image_path": page.image_path,
            "file_name": page.file_name,
            "width": w,
            "height": h,
            "meta": {
                "num_cols": num_cols,
                "split_x": split_x,
                "columns_debug": col_debug,
                "ro_debug": ro_debug,
            },
            "figures": out_figures,
            "questions": out_questions,
        }

        out_path = page_dir / "match.json"
        out_path.write_text(json.dumps(out_rec, ensure_ascii=False, indent=2), encoding="utf-8")
        if do_latency:
            _lat['write_json_ms'] = (time.perf_counter() - _t0) * 1000.0

        reason = None
        if isinstance(col_debug, dict):
            reason = col_debug.get("reason")
        reason_str = f", reason={reason}" if reason else ""
        print(f"[OK] {img_stem}: wrote {out_path.name} (Q={len(out_questions)}, F={len(out_figures)}, cols={num_cols}{reason_str})")

        split_x_viz = list(split_x) if split_x else []
        if isinstance(ro_debug, dict) and ro_debug.get("mode") == "spread" and ro_debug.get("split_x") is not None:
            gx = float(ro_debug["split_x"])
            if all(abs(gx - sx) > 5.0 for sx in split_x_viz):
                split_x_viz.append(gx)

        meta_lines = [
            f"cols={num_cols}",
            f"ro={ro_debug.get('mode') if isinstance(ro_debug, dict) else 'unknown'}",
        ]
        if isinstance(ro_debug, dict) and ro_debug.get("mode") == "spread" and ro_debug.get("split_x") is not None:
            meta_lines.append(f"gutter_x={float(ro_debug['split_x']):.1f}")

        if args.save_viz:
            _t_viz0 = time.perf_counter() if do_latency else 0.0
            reading_out = viz_root / f"{img_stem}_reading_overlay.png"
            draw_overlay(
                img,
                questions=out_questions,
                figures=out_figures,
                split_x=split_x_viz,
                out_path=str(reading_out),
                meta_lines=meta_lines,
                boundary_tol_ratio=args.col_boundary_tol_ratio,
                draw_edges=False,
                edges=[],
            )

            match_out = viz_root / f"{img_stem}_match_overlay.png"
            edges = []
            if args.draw_edges:
                for q in out_questions:
                    ms = sorted(q.get("matches") or [], key=lambda d: d.get("score", -1), reverse=True)
                    ms = ms[: max(1, args.max_edges_per_question)]
                    for m in ms:
                        edges.append((q["det_index"], m["figure_index"]))
            draw_overlay(
                img,
                questions=out_questions,
                figures=out_figures,
                split_x=split_x_viz,
                out_path=str(match_out),
                meta_lines=meta_lines,
                boundary_tol_ratio=args.col_boundary_tol_ratio,
                draw_edges=args.draw_edges,
                edges=edges,
            )
            if do_latency:
                _lat['viz_ms'] = (time.perf_counter() - _t_viz0) * 1000.0

        if do_latency:
            _lat['total_ms'] = (time.perf_counter() - _t_page0) * 1000.0
            _lat['core_ms'] = float(_lat.get('assign_columns_ms', 0.0)) + float(_lat.get('reading_order_ms', 0.0)) + float(_lat.get('match_prep_ms', 0.0)) + float(_lat.get('match_ms', 0.0))
            latency_rows.append(_lat)

    print("[DONE] all pages processed")

    if do_latency and latency_rows:
        def _summ(vals: List[float]) -> Dict[str, float]:
            sv = sorted([float(v) for v in vals if v is not None])
            return {
                "mean": float(mean(sv)) if sv else 0.0,
                "p50": float(_percentile(sv, 0.50)),
                "p90": float(_percentile(sv, 0.90)),
                "p95": float(_percentile(sv, 0.95)),
                "max": float(sv[-1]) if sv else 0.0,
            }

        total_ms = [r.get("total_ms", 0.0) for r in latency_rows]
        core_ms = [r.get("core_ms", 0.0) for r in latency_rows]
        s_total = _summ(total_ms)
        s_core = _summ(core_ms)

        stage_keys = [
            "open_image_ms",
            "extract_figures_ms",
            "assign_columns_ms",
            "reading_order_ms",
            "match_prep_ms",
            "match_ms",
            "io_prep_ms",
            "save_crops_ms",
            "write_json_ms",
            "viz_ms",
        ]
        stage_mean = {k: float(mean([float(r.get(k, 0.0)) for r in latency_rows])) for k in stage_keys}

        print(
            "[LATENCY] pages={n} | total_ms mean={tm:.2f} p50={t50:.2f} p90={t90:.2f} p95={t95:.2f} max={tmax:.2f} | "
            "core_ms mean={cm:.2f} p50={c50:.2f} p90={c90:.2f}".format(
                n=len(latency_rows),
                tm=s_total["mean"], t50=s_total["p50"], t90=s_total["p90"], t95=s_total["p95"], tmax=s_total["max"],
                cm=s_core["mean"], c50=s_core["p50"], c90=s_core["p90"],
            )
        )
        print(
            "[LATENCY-STAGES mean(ms)] "
            + ", ".join([f"{k}={stage_mean[k]:.2f}" for k in stage_keys if stage_mean.get(k, 0.0) > 0.0])
        )

        if getattr(args, "latency_json", ""):
            lp = _resolve_out_path(out_dir, str(getattr(args, "latency_json")))
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text(json.dumps(latency_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[LATENCY] wrote {lp}")

        if getattr(args, "latency_csv", ""):
            lp = _resolve_out_path(out_dir, str(getattr(args, "latency_csv")))
            lp.parent.mkdir(parents=True, exist_ok=True)
            keys: List[str] = []
            for r in latency_rows:
                for k in r.keys():
                    if k not in keys:
                        keys.append(k)
            with lp.open("w", encoding="utf-8", newline="") as f:
                wcsv = csv.DictWriter(f, fieldnames=keys)
                wcsv.writeheader()
                for r in latency_rows:
                    wcsv.writerow(r)
            print(f"[LATENCY] wrote {lp}")


if __name__ == "__main__":
    main()

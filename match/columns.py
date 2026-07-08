from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

XYXY = Tuple[float, float, float, float]


def x_center(b: XYXY) -> float:
    return (b[0] + b[2]) * 0.5


def bbox_w(b: XYXY) -> float:
    return max(0.0, b[2] - b[0])


def bbox_h(b: XYXY) -> float:
    return max(0.0, b[3] - b[1])


def _quantile(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    vs = sorted(vals)
    if len(vs) == 1:
        return float(vs[0])
    q = min(1.0, max(0.0, float(q)))
    pos = q * (len(vs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vs[lo])
    t = pos - lo
    return float(vs[lo] * (1.0 - t) + vs[hi] * t)


def _iqr(vals: List[float]) -> float:
    if len(vals) < 4:
        return 0.0
    return _quantile(vals, 0.75) - _quantile(vals, 0.25)


@dataclass
class ColumnInfo:
    num_cols: int
    split_x: List[float]
    centers: List[float]
    counts: List[int]
    inertia: List[float]
    debug: Dict[str, Any]

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------
# 1D k-means
# ------------------------------

def _stable_seed(xs: List[float], k: int, restart: int) -> int:
    h = 2166136261
    h = (h ^ int(k)) * 16777619
    h = (h ^ int(restart)) * 16777619
    for x in sorted(xs):
        xi = int(round(float(x) * 10.0))  # 0.1px 粗量化
        h = (h ^ xi) * 16777619
    return int(h & 0xFFFFFFFF)


def _kmeans_1d_once(xs: List[float], k: int, iters: int = 30, *, rng: Optional[random.Random] = None):
    if rng is None:
        rng = random

    # kmeans++ init
    centers = [rng.choice(xs)]
    while len(centers) < k:
        d2 = []
        for x in xs:
            dist2 = min((x - c) ** 2 for c in centers)
            d2.append(dist2)
        s = sum(d2)
        if s <= 1e-9:
            centers.append(rng.choice(xs))
            continue
        r = rng.random() * s
        acc = 0.0
        picked = xs[-1]
        for x, w in zip(xs, d2):
            acc += w
            if acc >= r:
                picked = x
                break
        centers.append(picked)

    labels = [0] * len(xs)
    for _ in range(iters):
        changed = False
        for i, x in enumerate(xs):
            best = 0
            best_d = (x - centers[0]) ** 2
            for j in range(1, k):
                d = (x - centers[j]) ** 2
                if d < best_d:
                    best_d = d
                    best = j
            if labels[i] != best:
                labels[i] = best
                changed = True

        new_centers = centers[:]
        for j in range(k):
            pts = [x for x, lab in zip(xs, labels) if lab == j]
            if pts:
                new_centers[j] = sum(pts) / len(pts)

        if (not changed) and all(abs(a - b) < 1e-6 for a, b in zip(new_centers, centers)):
            centers = new_centers
            break
        centers = new_centers

    inertia = 0.0
    for x, lab in zip(xs, labels):
        inertia += (x - centers[lab]) ** 2
    return centers, labels, float(inertia)


def _kmeans_1d_best(xs: List[float], k: int, restarts: int = 12):
    best = None
    for r in range(max(1, restarts)):
        rng = random.Random(_stable_seed(xs, k, r))
        centers, labels, inertia = _kmeans_1d_once(xs, k, rng=rng)
        if best is None or inertia < best[2]:
            best = (centers, labels, inertia)
    return best


# ------------------------------
# Column analysis
# ------------------------------

def analyze_columns(
    bboxes: Sequence[XYXY],
    page_w: float,
    page_h: Optional[float] = None,
    *,
    max_cols: int = 3,

    # 关键：判列特征（默认用 x1，更抗“右侧拖宽”）
    feature: str = "x1",  # "x1" | "xc"

    # 用于“判列”的 carrier 框筛选（过滤公式小框、图块、全宽大框等）
    carrier_min_w_frac: float = 0.10,
    carrier_max_w_frac: float = 0.70,
    carrier_min_h_frac: float = 0.04,

    # 跨中线/近乎全宽框：不参与判列
    exclude_cross_mid: bool = True,
    cross_mid_margin_ratio: float = 0.02,
    wide_exclude_w_frac: float = 0.78,

    # 单栏先验：宽框占比高且 x1 很集中 -> 强制 1 列
    singlecol_wide_frac: float = 0.74,
    singlecol_wide_min_frac: float = 0.35,
    singlecol_x1_iqr_max: float = 0.06,   # 归一化到 page_w 后的 IQR
    singlecol_x1_median_max: float = 0.28,  # 归一化到 page_w 后的 median

    # kmeans 选择阈值（保留你的 elbow 思路，但更偏保守）
    improve_min: float = 0.22,
    restarts: int = 12,

    # validity（中心分离 + gutter）
    min_sep_ratio: float = 0.16,
    min_cluster_frac: float = 0.12,
    min_cluster_size_abs: int = 2,
    gutter_min_ratio: float = 0.02,  # 估计列间空白最小宽度占比

    # 小样本 max-gap
    min_samples_for_cluster: int = 8,
    min_samples_for_gap: int = 4,
    gap_min_ratio: float = 0.18,
    edge_margin_ratio: float = 0.08,
    gap_min_cluster_size: int = 2,
    gap_over_median_min: float = 2.2,

    # split 有效性：split 落在大量 bbox 内部 -> 拒绝
    boundary_tol_ratio: float = 0.02,
    min_side_conf_frac: float = 0.60,
    max_amb_frac: float = 0.45,

    # split 计算用的分位数（更贴近 gutter）
    left_rightedge_q: float = 0.85,
    right_leftedge_q: float = 0.15,
) -> ColumnInfo:
    if page_w <= 1:
        page_w = 1.0
    if page_h is None or page_h <= 1:
        # 用 bbox 的最大 y2 粗估 page_h（比不传强）
        page_h = max([b[3] for b in bboxes], default=1.0)
        page_h = max(1.0, float(page_h))

    mid = 0.5 * page_w
    mid_margin = float(cross_mid_margin_ratio) * float(page_w)

    def _feat(b: XYXY) -> float:
        if feature == "xc":
            return x_center(b)
        return float(b[0])  # x1

    # carrier 收集
    carriers: List[Tuple[float, float, float, float, float, int]] = []
    # (feat, x1, x2, w_frac, h_frac, idx)
    excluded: List[int] = []
    used_idx: List[int] = []

    for i, b in enumerate(bboxes):
        w = bbox_w(b)
        h = bbox_h(b)
        if w <= 1 or h <= 1:
            continue
        wf = w / page_w
        hf = h / page_h

        # 过宽：不参与判列
        if wf >= wide_exclude_w_frac:
            excluded.append(i)
            continue

        # 跨中线：不参与判列（注意：这是“判列样本”剔除，不是最终题框丢弃）
        if bool(exclude_cross_mid) and (b[0] < (mid - mid_margin)) and (b[2] > (mid + mid_margin)):
            excluded.append(i)
            continue

        # carrier 条件（过滤噪声小框/图块）
        if (wf < carrier_min_w_frac) or (wf > carrier_max_w_frac) or (hf < carrier_min_h_frac):
            continue

        carriers.append((_feat(b), float(b[0]), float(b[2]), float(wf), float(hf), i))
        used_idx.append(i)

    # ---------- 单栏先验 ----------
    wide_boxes_x1 = []
    for b in bboxes:
        w = bbox_w(b)
        if w <= 1:
            continue
        wf = w / page_w
        if wf >= singlecol_wide_frac:
            wide_boxes_x1.append(float(b[0]) / page_w)

    wide_frac = (len(wide_boxes_x1) / max(1, len(bboxes))) if bboxes else 0.0
    x1_med = _quantile(wide_boxes_x1, 0.5) if wide_boxes_x1 else 1.0
    x1_iqr = _iqr(wide_boxes_x1) if wide_boxes_x1 else 1.0

    params_dbg = {
        "feature": feature,
        "carrier_min_w_frac": carrier_min_w_frac,
        "carrier_max_w_frac": carrier_max_w_frac,
        "carrier_min_h_frac": carrier_min_h_frac,
        "exclude_cross_mid": exclude_cross_mid,
        "cross_mid_margin_ratio": cross_mid_margin_ratio,
        "wide_exclude_w_frac": wide_exclude_w_frac,
        "singlecol_wide_frac": singlecol_wide_frac,
        "singlecol_wide_min_frac": singlecol_wide_min_frac,
        "singlecol_x1_iqr_max": singlecol_x1_iqr_max,
        "singlecol_x1_median_max": singlecol_x1_median_max,
        "improve_min": improve_min,
        "min_sep_ratio": min_sep_ratio,
        "min_cluster_frac": min_cluster_frac,
        "min_cluster_size_abs": min_cluster_size_abs,
        "gutter_min_ratio": gutter_min_ratio,
        "boundary_tol_ratio": boundary_tol_ratio,
        "min_side_conf_frac": min_side_conf_frac,
        "max_amb_frac": max_amb_frac,
        "left_rightedge_q": left_rightedge_q,
        "right_leftedge_q": right_leftedge_q,
    }

    if (wide_frac >= singlecol_wide_min_frac) and (x1_iqr <= singlecol_x1_iqr_max) and (x1_med <= singlecol_x1_median_max):
        return ColumnInfo(
            num_cols=1,
            split_x=[],
            centers=[page_w * 0.5],
            counts=[len(carriers)],
            inertia=[0.0],
            debug={
                "reason": "singlecol_prior",
                "n_total": len(bboxes),
                "n_carriers": len(carriers),
                "excluded_idx": excluded,
                "used_idx": used_idx,
                "wide_frac": wide_frac,
                "wide_x1_med": x1_med,
                "wide_x1_iqr": x1_iqr,
                "params": params_dbg,
            },
        )

    # 判列没有足够 carrier：直接 1 列（不要用噪声硬拆）
    if len(carriers) < max(1, int(min_samples_for_gap)):
        return ColumnInfo(
            num_cols=1,
            split_x=[],
            centers=[page_w * 0.5],
            counts=[len(carriers)],
            inertia=[0.0],
            debug={
                "reason": "too_few_carriers",
                "n_total": len(bboxes),
                "n_carriers": len(carriers),
                "excluded_idx": excluded,
                "used_idx": used_idx,
                "wide_frac": wide_frac,
                "wide_x1_med": x1_med,
                "wide_x1_iqr": x1_iqr,
                "params": params_dbg,
            },
        )

    xs = [c[0] for c in carriers]

    def _boundary_quality(boundary: float) -> Tuple[bool, Dict[str, float]]:
        tol = float(boundary_tol_ratio) * float(page_w)
        left = [c for c in carriers if c[0] < boundary]
        right = [c for c in carriers if c[0] >= boundary]
        if len(left) < int(gap_min_cluster_size) or len(right) < int(gap_min_cluster_size):
            return False, {"n": float(len(carriers)), "n_left": float(len(left)), "n_right": float(len(right))}

        left_conf = sum(1 for _, x1, x2, *_ in left if x2 <= boundary + tol) / max(1, len(left))
        right_conf = sum(1 for _, x1, x2, *_ in right if x1 >= boundary - tol) / max(1, len(right))
        amb = sum(1 for _, x1, x2, *_ in carriers if (x1 < boundary - tol and x2 > boundary + tol)) / max(1, len(carriers))

        ok = (left_conf >= min_side_conf_frac) and (right_conf >= min_side_conf_frac) and (amb <= max_amb_frac)
        return ok, {
            "left_conf": float(left_conf),
            "right_conf": float(right_conf),
            "amb_frac": float(amb),
            "tol": float(tol),
            "n": float(len(carriers)),
            "n_left": float(len(left)),
            "n_right": float(len(right)),
        }

    def _split_from_clusters(left_cluster: List[Tuple[float, float, float, float, float, int]],
                            right_cluster: List[Tuple[float, float, float, float, float, int]]) -> Tuple[float, Dict[str, Any]]:
        # 用分位数估计“列间空白”
        left_x2 = [c[2] for c in left_cluster]
        right_x1 = [c[1] for c in right_cluster]
        a = _quantile(left_x2, left_rightedge_q)
        b = _quantile(right_x1, right_leftedge_q)
        gutter = float(b - a)
        boundary = float((a + b) * 0.5)
        dbg = {"a_left_x2_q": a, "b_right_x1_q": b, "gutter": gutter, "boundary": boundary}
        return boundary, dbg

    def _try_split_by_max_gap(xs_: List[float]) -> Optional[Tuple[float, Dict[str, Any]]]:
        if len(xs_) < int(min_samples_for_gap):
            return None
        xs_sorted = sorted(xs_)
        gaps = [xs_sorted[i + 1] - xs_sorted[i] for i in range(len(xs_sorted) - 1)]
        if not gaps:
            return None
        j = max(range(len(gaps)), key=lambda i: gaps[i])
        max_gap = float(gaps[j])
        boundary = float((xs_sorted[j] + xs_sorted[j + 1]) * 0.5)

        med_gap = float(sorted(gaps)[len(gaps) // 2])
        dbg = {"max_gap": max_gap, "med_gap": med_gap, "boundary": boundary}

        if max_gap / max(1.0, page_w) < float(gap_min_ratio):
            dbg["fail"] = "gap_min_ratio"
            return None
        if med_gap > 1e-9 and (max_gap / med_gap) < float(gap_over_median_min):
            dbg["fail"] = "gap_over_median"
            return None

        margin = float(edge_margin_ratio) * float(page_w)
        if not (margin < boundary < (page_w - margin)):
            dbg["fail"] = "edge_margin"
            return None

        ok_b, st = _boundary_quality(boundary)
        dbg["boundary_quality"] = st
        if not ok_b:
            dbg["fail"] = "boundary_quality"
            return None

        # gutter 也要够
        left_cluster = [c for c in carriers if c[0] < boundary]
        right_cluster = [c for c in carriers if c[0] >= boundary]
        if len(left_cluster) < int(gap_min_cluster_size) or len(right_cluster) < int(gap_min_cluster_size):
            dbg["fail"] = "gap_min_cluster_size"
            return None

        boundary2, dbg2 = _split_from_clusters(left_cluster, right_cluster)
        dbg["quantile_split"] = dbg2
        if (dbg2["gutter"] / page_w) < float(gutter_min_ratio):
            dbg["fail"] = "gutter_too_small"
            return None

        # 用更合理的 boundary2 替换
        return float(boundary2), dbg

    # ---------- Case A: 极小样本优先 max-gap ----------
    if len(xs) < max(1, int(min_samples_for_cluster)):
        split = _try_split_by_max_gap(xs)
        if split is not None:
            boundary, dbg = split
            # centers 用左右簇的均值（debug 用）
            left = [c[0] for c in carriers if c[0] < boundary]
            right = [c[0] for c in carriers if c[0] >= boundary]
            centers = [sum(left) / len(left), sum(right) / len(right)]
            return ColumnInfo(
                num_cols=2,
                split_x=[boundary],
                centers=[float(x) for x in centers],
                counts=[len(left), len(right)],
                inertia=[0.0],
                debug={
                    "reason": "small_sample_max_gap",
                    "n_total": len(bboxes),
                    "n_carriers": len(carriers),
                    "excluded_idx": excluded,
                    "used_idx": used_idx,
                    "params": params_dbg,
                    "dbg": dbg,
                },
            )

        return ColumnInfo(
            num_cols=1,
            split_x=[],
            centers=[page_w * 0.5],
            counts=[len(carriers)],
            inertia=[0.0],
            debug={
                "reason": "small_sample_fallback_1col",
                "n_total": len(bboxes),
                "n_carriers": len(carriers),
                "excluded_idx": excluded,
                "used_idx": used_idx,
                "params": params_dbg,
            },
        )

    # ---------- Case B: 样本足够，用 kmeans ----------
    max_k = min(int(max_cols), max(1, len(xs)))
    inertias: List[float] = []
    best_by_k: Dict[int, Any] = {}

    for k in range(1, max_k + 1):
        centers, labels, inertia = _kmeans_1d_best(xs, k, restarts=restarts)
        best_by_k[k] = (centers, labels, inertia)
        inertias.append(float(inertia))

    chosen_k = 1
    for k in range(2, max_k + 1):
        prev = inertias[k - 2]
        curr = inertias[k - 1]
        if prev <= 1e-9:
            break
        rel_improve = (prev - curr) / prev
        if rel_improve >= float(improve_min):
            chosen_k = k
        else:
            break

    def _pack_k(k: int):
        centers, labels, _ = best_by_k[k]
        order = sorted(range(k), key=lambda j: centers[j])
        centers_sorted = [float(centers[j]) for j in order]
        remap = {old: new for new, old in enumerate(order)}
        labels_sorted = [remap[lab] for lab in labels]

        clusters: List[List[Tuple[float, float, float, float, float, int]]] = [[] for _ in range(k)]
        for c, lab in zip(carriers, labels_sorted):
            clusters[lab].append(c)

        counts = [len(cl) for cl in clusters]

        # split_x 用“簇间边界分位数”推断（更贴近 gutter）
        split_x: List[float] = []
        split_dbg: List[Dict[str, Any]] = []
        for j in range(k - 1):
            boundary, dbg = _split_from_clusters(clusters[j], clusters[j + 1])
            split_x.append(boundary)
            split_dbg.append(dbg)

        return centers_sorted, labels_sorted, counts, split_x, clusters, split_dbg

    def _valid(k: int,
               centers_s: List[float],
               counts_s: List[int],
               split_x: List[float],
               clusters: List[List[Tuple[float, float, float, float, float, int]]],
               split_dbg: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        if k <= 1:
            return True, {"k": 1}

        n = sum(counts_s)
        # cluster size constraints
        if min(counts_s) < max(int(min_cluster_size_abs), int(math.ceil(float(min_cluster_frac) * n))):
            return False, {"fail": "cluster_size", "counts": counts_s, "n": n}

        # center separation constraints（用 centers 只是粗约束）
        for a, b in zip(centers_s[:-1], centers_s[1:]):
            if (b - a) / page_w < float(min_sep_ratio):
                return False, {"fail": "center_sep", "centers": centers_s}

        # gutter constraints（关键：簇间必须有真实空白）
        for dbg in split_dbg:
            if (dbg["gutter"] / page_w) < float(gutter_min_ratio):
                return False, {"fail": "gutter_too_small", "split_dbg": split_dbg}

        # boundary validity constraints
        boundary_stats: List[Dict[str, Any]] = []
        for sx in split_x:
            ok_b, st = _boundary_quality(float(sx))
            st = dict(st)
            st["boundary"] = float(sx)
            st["ok"] = bool(ok_b)
            boundary_stats.append(st)
            if not ok_b:
                return False, {"fail": "boundary_quality", "boundary_stats": boundary_stats, "split_dbg": split_dbg}

        return True, {"ok": True, "boundary_stats": boundary_stats, "split_dbg": split_dbg}

    k_curr = chosen_k
    centers_sorted, labels_sorted, counts, split_x, clusters, split_dbg = _pack_k(k_curr)
    ok, vdbg = _valid(k_curr, centers_sorted, counts, split_x, clusters, split_dbg)

    while k_curr > 1 and not ok:
        k_curr -= 1
        centers_sorted, labels_sorted, counts, split_x, clusters, split_dbg = _pack_k(k_curr)
        ok, vdbg = _valid(k_curr, centers_sorted, counts, split_x, clusters, split_dbg)

    # 如果 kmeans 退化到 1 列，但 max-gap 很明显，则尝试 override（仍需 gutter/quality）
    if k_curr == 1:
        split = _try_split_by_max_gap(xs)
        if split is not None:
            boundary, dbg = split
            left = [c[0] for c in carriers if c[0] < boundary]
            right = [c[0] for c in carriers if c[0] >= boundary]
            centers = [sum(left) / len(left), sum(right) / len(right)]
            return ColumnInfo(
                num_cols=2,
                split_x=[boundary],
                centers=[float(x) for x in centers],
                counts=[len(left), len(right)],
                inertia=[0.0],
                debug={
                    "reason": "gap_override_kmeans1",
                    "n_total": len(bboxes),
                    "n_carriers": len(carriers),
                    "excluded_idx": excluded,
                    "used_idx": used_idx,
                    "params": params_dbg,
                    "dbg": dbg,
                    "kmeans_inertias": inertias[:max_k],
                },
            )

        return ColumnInfo(
            num_cols=1,
            split_x=[],
            centers=[page_w * 0.5],
            counts=[len(carriers)],
            inertia=[inertias[0] if inertias else 0.0],
            debug={
                "reason": "kmeans1",
                "n_total": len(bboxes),
                "n_carriers": len(carriers),
                "excluded_idx": excluded,
                "used_idx": used_idx,
                "params": params_dbg,
                "kmeans_inertias": inertias[:max_k],
                "validity": vdbg,
            },
        )

    return ColumnInfo(
        num_cols=k_curr,
        split_x=[float(x) for x in split_x],
        centers=centers_sorted,
        counts=counts,
        inertia=inertias[:k_curr] if inertias else [0.0],
        debug={
            "reason": "kmeans",
            "n_total": len(bboxes),
            "n_carriers": len(carriers),
            "excluded_idx": excluded,
            "used_idx": used_idx,
            "wide_frac": wide_frac,
            "wide_x1_med": x1_med,
            "wide_x1_iqr": x1_iqr,
            "params": params_dbg,
            "kmeans_inertias": inertias[:max_k],
            "validity": vdbg,
        },
    )


# ------------------------------
# Assign column id (重要：处理跨列框)
# ------------------------------

def assign_col_id(xc: float, split_x: List[float]) -> int:
    """旧接口：仅用中心点，保留兼容。"""
    col = 0
    for sx in split_x:
        if xc >= sx:
            col += 1
        else:
            break
    return col


def assign_col_id_bbox(
    b: XYXY,
    split_x: List[float],
    page_w: float,
    *,
    boundary_tol_ratio: float = 0.02,
) -> int:
    """
    新接口：对跨 split 的 bbox 返回 -1（span），避免把“全宽/跨列题框”硬塞进某列。
    """
    if not split_x:
        return 0
    tol = float(boundary_tol_ratio) * float(max(1.0, page_w))
    x1, x2 = float(b[0]), float(b[2])
    xc = (x1 + x2) * 0.5

    col = 0
    for sx in split_x:
        sx = float(sx)
        # 跨越 split：标为 span
        if (x1 < sx - tol) and (x2 > sx + tol):
            return -1
        if xc >= sx:
            col += 1
        else:
            break
    return col

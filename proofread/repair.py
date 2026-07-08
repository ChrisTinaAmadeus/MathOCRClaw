from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .cache import JsonCache
from .common import extract_qno_from_text_block, normalize_ws, text_similarity
from .verify import (
    VerifyResult,
    verify_question_lenient,
    verify_question_strict,
    verify_span,
    verify_span_readable,
    verify_evidence,
    verify_has_options,
)

MASK_U_DEFAULT = "【无法识别】"
MASK_N_DEFAULT = "【疑似幻觉】"

_OPT_HEAD_RE = re.compile(r"^\s*[\(（]?\s*([A-F])\s*[\)）]?\s*[\.．、:：]\s*(.*)\s*$")

_UNDERSCORE_RE = re.compile(r"(_{3,}|＿{3,})")

GEN_PROMPT = (
    "You are an OCR transcriber. Transcribe only the printed text in the image, and ignore handwritten annotations, corrections, and noise.\n"
    "Preserve the question number, options (A/B/C/D), mathematical symbols, and original line breaks as much as possible.\n"
    "Do not solve the problem, and do not add any text that does not exist in the image.\n"
    "Output plain text only (LaTeX is allowed). Do not include any explanation.\n"
)


def _split_option_blocks(lines: List[str]) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
    pre: List[str] = []
    post: List[str] = []
    opts: Dict[str, List[str]] = {}
    cur_letter: Optional[str] = None
    cur_lines: List[str] = []
    seen_any = False

    def flush() -> None:
        nonlocal cur_letter, cur_lines
        if cur_letter and cur_lines:
            opts[cur_letter] = cur_lines[:]
        cur_letter, cur_lines = None, []

    for ln in lines:
        m = _OPT_HEAD_RE.match(ln)
        if m:
            seen_any = True
            flush()
            cur_letter = m.group(1)
            cur_lines = [ln]
        else:
            if not seen_any:
                pre.append(ln)
            else:
                if cur_letter:
                    cur_lines.append(ln)
                else:
                    post.append(ln)
    flush()
    return pre, opts, post


def _reconstruct(pre: List[str], opts: Dict[str, List[str]], post: List[str]) -> str:
    out: List[str] = []
    out += pre
    for L in ["A", "B", "C", "D", "E", "F"]:
        if L in opts:
            out += opts[L]
    for k in sorted([k for k in opts.keys() if k not in ("A", "B", "C", "D")]):
        out += opts[k]
    out += post
    return normalize_ws("\n".join([ln.rstrip() for ln in out if ln is not None]))


def _detect_strip_crop(img: Image.Image, *, h_over_w: float = 0.22, min_h: int = 90) -> bool:
    w, h = img.size
    if w <= 0 or h <= 0:
        return True
    ratio = h / float(w)
    # extremely thin always treated as strip
    if ratio < h_over_w:
        return True
    # very short AND thin -> strip
    if h < min_h and ratio < 0.35:
        return True
    return False


def _canonicalize_qno_prefix(text: str, qno: Optional[int], *, mask_u_token: str) -> str:
    if not qno:
        return text
    lines = (text or "").splitlines()
    if not lines:
        return f"{qno}. {mask_u_token}"
    first = lines[0].lstrip()
    m = re.match(r"^(?:第\s*)?(\d{1,3})\s*(?:题)?\s*[\.．、]\s*(.*)$", first)
    if not m:
        # no leading number: prepend
        return normalize_ws(f"{qno}. " + first + ("\n" + "\n".join(lines[1:]) if len(lines) > 1 else ""))
    try:
        n0 = int(m.group(1))
    except Exception:
        n0 = qno
    rest = m.group(2)
    if n0 == int(qno):
        return text
    # replace the leading number with the expected qno (fixes "13. 14." artifacts)
    new_first = f"{qno}. {rest}".rstrip()
    return normalize_ws("\n".join([new_first] + lines[1:]))


def partition_lines(lines: List[str], max_chars: int = 420) -> List[str]:
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for ln in lines:
        ln2 = ln.rstrip()
        if not ln2 and not cur:
            continue
        if cur_len + len(ln2) + 1 > max_chars and cur:
            chunks.append("\n".join(cur).strip())
            cur, cur_len = [], 0
        cur.append(ln2)
        cur_len += len(ln2) + 1
    if cur:
        chunks.append("\n".join(cur).strip())
    return [c for c in chunks if c.strip()]


def _combine_verdict(v1: str, v2: str) -> str:
    s = {v1, v2}
    if "Y" in s:
        return "Y"
    if "N" in s and "U" in s:
        return "U"
    if s == {"N"}:
        return "N"
    return "U"


def _safe_prune_suspicious_tail(
    ver_vlm, img: Image.Image, text: str, *, cache: Optional[JsonCache]
) -> Tuple[str, Dict[str, Any]]:
    dbg: Dict[str, Any] = {"did_prune": False}
    t = normalize_ws(text)
    if not t.strip():
        return t, dbg

    m = None
    for m2 in _UNDERSCORE_RE.finditer(t):
        m = m2
    if not m:
        return t, dbg

    us = m.start()
    start = us
    zpos = t.rfind("则", max(0, us - 30), us)
    if zpos != -1:
        start = zpos
    else:
        eqpos = t.rfind("=", max(0, us - 12), us)
        if eqpos != -1:
            start = eqpos

    tail = t[start:].strip()
    if not tail:
        return t, dbg

    ev = verify_evidence(ver_vlm, img, tail, cache=cache)
    dbg["tail"] = tail[:120]
    dbg["tail_evidence"] = ev.verdict
    if ev.verdict == "Y":
        return t, dbg

    # prune
    pruned = normalize_ws(t[:start].rstrip())
    dbg["did_prune"] = True
    dbg["removed_chars"] = int(max(0, len(t) - len(pruned)))
    dbg["removed_frac"] = float(dbg["removed_chars"] / float(max(1, len(t))))
    return pruned, dbg


def generate_regen_text(gen_vlm, img: Image.Image, *, cache: Optional[JsonCache]) -> str:
    from .img_utils import STRONG_ENHANCE, enhance_for_vlm, img_to_data_url

    img2 = enhance_for_vlm(img, STRONG_ENHANCE)
    data_url, img_hash = img_to_data_url(img2)


    regen_temp = float(getattr(gen_vlm, "temperature", 0.7))
    regen_top_p = float(getattr(gen_vlm, "top_p", 0.8))

    ck = f"{gen_vlm.model}::t{regen_temp:.2f}::p{regen_top_p:.2f}::{img_hash}::regen"
    if cache:
        hit = cache.get("gen", ck)
        if isinstance(hit, str) and hit.strip():
            return hit

    raw = gen_vlm.invoke(
        [
            {"type": "text", "text": GEN_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
        temperature=regen_temp,
        top_p=regen_top_p,
        max_tokens=1200,
    )
    out = normalize_ws(raw)
    if cache:
        cache.set("gen", ck, out)
    return out


@dataclass
class RepairConfig:
    max_qchars: int = 900
    max_verify_chars: int = 420   
    mask_u_token: str = MASK_U_DEFAULT
    mask_n_token: str = MASK_N_DEFAULT

    # heuristics
    sim_keep_threshold: float = 0.985 
    strip_h_over_w: float = 0.25      
    strip_min_h: int = 120


def repair_question_v6_6(
    ver_vlm,
    gen_vlm,
    crop_img: Image.Image,
    candidate_text: str,
    *,
    cache: Optional[JsonCache],
    cfg: RepairConfig,
) -> Tuple[str, Dict[str, Any]]:
    """Return (final_text, debug_dict)."""
    dbg: Dict[str, Any] = {"stage": "init"}
    dbg.setdefault("fixed_actions", [])  
    dbg.setdefault("rewrite_actions", []) 
    cand0 = normalize_ws(candidate_text)
    cand = cand0
    qno = extract_qno_from_text_block(cand)

    is_strip = _detect_strip_crop(crop_img, h_over_w=cfg.strip_h_over_w, min_h=cfg.strip_min_h)
    dbg["is_strip"] = is_strip

    _enh_cache = {"img": None}

    def _get_enh_img():
        if _enh_cache["img"] is None:
            try:
                from .img_utils import STRONG_ENHANCE, enhance_for_vlm
                _enh_cache["img"] = enhance_for_vlm(crop_img, STRONG_ENHANCE)
            except Exception:
                _enh_cache["img"] = crop_img
        return _enh_cache["img"] or crop_img

    cand2, tail_dbg = _safe_prune_suspicious_tail(ver_vlm, crop_img, cand, cache=cache)
    dbg["tail_prune"] = tail_dbg
    cand = cand2
    if isinstance(tail_dbg, dict) and tail_dbg.get("did_prune") and tail_dbg.get("tail_evidence") == "N":
        dbg["fixed_actions"].append({
            "action": "tail_prune",
            "removed_tail": tail_dbg.get("tail"),
            "removed_chars": tail_dbg.get("removed_chars"),
            "removed_frac": tail_dbg.get("removed_frac"),
        })

    v1 = verify_question_strict(ver_vlm, crop_img, cand, cache=cache, max_chars=cfg.max_qchars)
    dbg["v_strict"] = v1.verdict
    if v1.verdict == "Y" and (not is_strip):
        out = _canonicalize_qno_prefix(cand, qno, mask_u_token=cfg.mask_u_token)
        dbg["stage"] = "accept_strict"
        dbg["delta_frac"] = float(abs(len(out) - len(cand0)) / float(max(1, len(cand0))))
        return out, dbg

    v2: Optional[VerifyResult] = None
    if v1.verdict == "N":
        v2 = verify_question_lenient(ver_vlm, crop_img, cand, cache=cache, max_chars=cfg.max_qchars)
        dbg["v_lenient"] = v2.verdict
        if v2.verdict == "Y" and (not is_strip):
            out = _canonicalize_qno_prefix(cand, qno, mask_u_token=cfg.mask_u_token)
            dbg["stage"] = "accept_lenient"
            dbg["delta_frac"] = float(abs(len(out) - len(cand0)) / float(max(1, len(cand0))))
            return out, dbg

    def _probe_has_options_multi(img0):
        def _norm(v: str) -> str:
            v2 = (v or "").strip().upper()
            return v2 if v2 in ("Y", "N", "U") else "U"

        votes: Dict[str, Any] = {}
        vlist: List[str] = []

        try:
            v0 = verify_has_options(ver_vlm, img0, cache=cache).verdict
        except Exception as e:
            votes["orig_err"] = str(e)
            v0 = "U"
        votes["orig"] = v0
        vlist.append(_norm(v0))

        img_enh = None
        try:
            img_enh = _get_enh_img()
            v1 = verify_has_options(ver_vlm, img_enh, cache=cache).verdict
        except Exception as e:
            votes["enh_err"] = str(e)
            v1 = "U"

        votes["enh"] = v1
        vlist.append(_norm(v1))

        try:
            img_up = img_enh or img0
            if img_up is not None:
                from PIL import Image
                w, h = img_up.size
                m = max(w, h)
                target = 1400
                if m > 0 and m < target:
                    scale = target / float(m)
                    img_up = img_up.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
            v2 = verify_has_options(ver_vlm, img_up, cache=cache).verdict
        except Exception as e:
            votes["up_err"] = str(e)
            v2 = "U"
        votes["up"] = v2
        vlist.append(_norm(v2))
        s = set(vlist)
        if "Y" in s:
            return "Y", votes
        if s == {"N"}:
            return "N", votes
        return "U", votes

    has_opt_v, has_opt_votes = _probe_has_options_multi(crop_img)
    from types import SimpleNamespace
    has_opt = SimpleNamespace(verdict=has_opt_v)
    dbg["has_options_probe"] = has_opt_v
    dbg["has_options_probe_votes"] = has_opt_votes

    cand_lines = cand.splitlines()
    cand_pre, cand_opts, cand_post = _split_option_blocks(cand_lines)
    dbg["cand_opt_letters"] = sorted(list(cand_opts.keys()))
    cand_has_opts = (len(cand_opts) >= 2)

    need_regen = (
        (v1.verdict != "Y")
        or is_strip
        or (cand_has_opts and has_opt.verdict in ("N", "U"))
        or bool(tail_dbg.get("did_prune", False))
    )

    regen = ""
    reg_pre: List[str] = []
    reg_opts: Dict[str, List[str]] = {}
    reg_post: List[str] = []
    if need_regen:
        regen = generate_regen_text(gen_vlm, crop_img, cache=cache)
        dbg["regen_len"] = len(regen)
        dbg["sim_cand_regen"] = float(text_similarity(cand, regen)) if regen else None
        reg_lines = (regen or "").splitlines()
        reg_pre, reg_opts, reg_post = _split_option_blocks(reg_lines)

    new_opts: Dict[str, List[str]] = {}
    opt_dbg: Dict[str, Any] = {}

    if False and has_opt.verdict == "N" and cand_opts:
        for L in cand_opts.keys():
            new_opts[L] = [f"{L}. {cfg.mask_n_token}"]
            opt_dbg[L] = {"reason": "probe_no_options", "final": "N->MASK_N"}
    else:
        for L, cand_block in cand_opts.items():
            cand_opt_text = normalize_ws("\n".join(cand_block))

            ev = verify_evidence(ver_vlm, crop_img, cand_opt_text, cache=cache)
            if ev.verdict == "Y":
                new_opts[L] = cand_block
                opt_dbg[L] = {"evidence": "Y", "final": "keep"}
                continue

            v_span = verify_span(ver_vlm, crop_img, cand_opt_text, cache=cache)
            v_read = verify_span_readable(ver_vlm, crop_img, cand_opt_text, cache=cache)
            vv = _combine_verdict(v_span.verdict, v_read.verdict)
            opt_dbg[L] = {"evidence": ev.verdict, "span": v_span.verdict, "readable": v_read.verdict, "combined": vv}
            if has_opt.verdict == "N" and vv != "Y":
                new_opts[L] = [f"{L}. {cfg.mask_n_token}"]
                opt_dbg[L]["probe_no_options"] = True
                opt_dbg[L]["final"] = "mask_n_by_no_options_probe"
                continue


            if vv == "Y":
                new_opts[L] = cand_block
                opt_dbg[L]["final"] = "keep"
                continue

            if vv == "U":
                reg_block = reg_opts.get(L)
                if reg_block:
                    reg_opt_text = normalize_ws("\n".join(reg_block))
                    ev2 = verify_evidence(ver_vlm, crop_img, reg_opt_text, cache=cache)
                    if ev2.verdict == "Y":
                        new_opts[L] = reg_block
                        opt_dbg[L]["final"] = "regen_by_evidence"
                        dbg["rewrite_actions"].append("regen_by_evidence")
                        continue
                new_opts[L] = [f"{L}. {cfg.mask_u_token}"]
                opt_dbg[L]["final"] = "mask_u"
                continue

            reg_block = reg_opts.get(L)
            if reg_block:
                reg_opt_text = normalize_ws("\n".join(reg_block))
                ev2 = verify_evidence(ver_vlm, crop_img, reg_opt_text, cache=cache)
                if ev2.verdict == "Y":
                    new_opts[L] = reg_block
                    opt_dbg[L]["final"] = "regen_by_evidence"
                    dbg["rewrite_actions"].append("regen_by_evidence")
                else:
                    new_opts[L] = [f"{L}. {cfg.mask_n_token}"]
                    opt_dbg[L]["final"] = "mask_n"
            else:
                new_opts[L] = [f"{L}. {cfg.mask_n_token}"]
                opt_dbg[L]["final"] = "mask_n(no_regen)"

        inserted: List[str] = []
        if has_opt.verdict == "Y":
            for L in ["A", "B", "C", "D", "E", "F"]:
                if L in new_opts:
                    continue
                if L in reg_opts:
                    reg_block = reg_opts[L]
                    reg_opt_text = normalize_ws("\n".join(reg_block))
                    ev2 = verify_evidence(ver_vlm, crop_img, reg_opt_text, cache=cache)
                    if ev2.verdict == "Y":
                        new_opts[L] = reg_block
                        inserted.append(L)
                        dbg["rewrite_actions"].append("insert_option_by_evidence")
        if inserted:
            dbg["inserted_opts"] = inserted

    dbg["options"] = opt_dbg

    new_pre = cand_pre
    new_post = cand_post

    if not is_strip:
        spans = partition_lines(cand_pre, max_chars=420)
        out_spans: List[str] = []
        stem_dbg: Dict[str, Any] = {}
        for sp in spans:
            ev = verify_evidence(ver_vlm, crop_img, sp, cache=cache)
            if ev.verdict == "Y":
                out_spans.append(sp)
                continue
            v_span = verify_span(ver_vlm, crop_img, sp, cache=cache)
            v_read = verify_span_readable(ver_vlm, crop_img, sp, cache=cache)
            vv = _combine_verdict(v_span.verdict, v_read.verdict)
            if vv == "N":
                out_spans.append(cfg.mask_n_token)
            elif vv == "U":

                try:
                    sp_probe = sp[:180]
                    ev2 = verify_evidence(ver_vlm, _get_enh_img(), sp_probe, cache=cache)
                    dbg.setdefault("enhanced_evidence", []).append({"span": sp_probe, "v": ev2.verdict})
                    if ev2.verdict == "Y":
                        out_spans.append(sp)
                        continue
                except Exception as e:
                    dbg.setdefault("enhanced_evidence_err", []).append(str(e))
                out_spans.append(cfg.mask_u_token)
            else:
                out_spans.append(sp)
            stem_dbg[sp[:40]] = {"evidence": ev.verdict, "span": v_span.verdict, "readable": v_read.verdict, "combined": vv}
        new_pre = normalize_ws("\n".join(out_spans)).splitlines() if out_spans else cand_pre
        dbg["stem"] = stem_dbg

        if cand_post:
            spans = partition_lines(cand_post, max_chars=420)
            out_spans = []
            for sp in spans:
                ev = verify_evidence(ver_vlm, crop_img, sp, cache=cache)
                if ev.verdict == "Y":
                    out_spans.append(sp)
                    continue
                v_span = verify_span(ver_vlm, crop_img, sp, cache=cache)
                v_read = verify_span_readable(ver_vlm, crop_img, sp, cache=cache)
                vv = _combine_verdict(v_span.verdict, v_read.verdict)
                if vv == "N":
                    out_spans.append(cfg.mask_n_token)
                elif vv == "U":

                    try:
                        sp_probe = sp[:180]
                        ev2 = verify_evidence(ver_vlm, _get_enh_img(), sp_probe, cache=cache)
                        dbg.setdefault("enhanced_evidence", []).append({"span": sp_probe, "v": ev2.verdict})
                        if ev2.verdict == "Y":
                            out_spans.append(sp)
                            continue
                    except Exception as e:
                        dbg.setdefault("enhanced_evidence_err", []).append(str(e))
                    out_spans.append(cfg.mask_u_token)
                else:
                    out_spans.append(sp)
            new_post = normalize_ws("\n".join(out_spans)).splitlines() if out_spans else cand_post

    repaired = _reconstruct(new_pre, new_opts, new_post)
    repaired = _canonicalize_qno_prefix(repaired, qno, mask_u_token=cfg.mask_u_token)

    dbg["repaired_len"] = len(repaired)

    v3 = verify_question_lenient(ver_vlm, crop_img, repaired, cache=cache, max_chars=cfg.max_qchars)
    dbg["v_after_repair"] = v3.verdict

    if v3.verdict == "Y":
        dbg["stage"] = "accept_repaired"
        dbg["delta_frac"] = float(abs(len(repaired) - len(cand0)) / float(max(1, len(cand0))))
        return repaired, dbg

    if (cfg.mask_u_token in repaired) or (cfg.mask_n_token in repaired) or tail_dbg.get("did_prune", False) or (has_opt.verdict == "N" and cand_opts):
        dbg["stage"] = "keep_conservative_repaired"
        dbg["delta_frac"] = float(abs(len(repaired) - len(cand0)) / float(max(1, len(cand0))))
        return repaired, dbg

    if regen and text_similarity(cand, regen) >= cfg.sim_keep_threshold:
        dbg["stage"] = "keep_candidate_by_similarity"
        out = _canonicalize_qno_prefix(cand, qno, mask_u_token=cfg.mask_u_token)
        dbg["delta_frac"] = float(abs(len(out) - len(cand0)) / float(max(1, len(cand0))))
        return out, dbg

    def _has_any_verified_keep(dbg: Dict[str, Any]) -> bool:
        stem_dbg = dbg.get("stem", {})
        if isinstance(stem_dbg, dict):
            for _, sd in stem_dbg.items():
                if isinstance(sd, dict) and (sd.get("combined") == "Y" or sd.get("span") == "Y" or sd.get("readable") == "Y"):
                    return True

        opt_dbg = dbg.get("options", {})
        if isinstance(opt_dbg, dict):
            for _, od in opt_dbg.items():
                if not isinstance(od, dict):
                    continue
                if od.get("evidence") == "Y":
                    return True
                final = str(od.get("final") or "")
                if ("EVI_Y_KEEP" in final) or ("regen_by_evidence" in final) or ("KEEP" in final):
                    return True
        return False

    if v1.verdict == "N" and v2 and v2.verdict == "N":
        if is_strip or _has_any_verified_keep(dbg):
            dbg["stage"] = "strip_keep_repaired_despite_NN" if is_strip else "keep_repaired_despite_NN"
            out = repaired
            dbg["delta_frac"] = float(1.0 - (len(out) / max(1, len(cand0))))
            return out, dbg

        confirm = {"mask": False, "reason": "not_run"}
        try:
            probe = ""
            for ln in cand0.splitlines():
                ln2 = ln.strip()
                if ln2:
                    probe = ln2[:120]
                    break
            if not probe:
                probe = cand0[:120]

            evp = verify_evidence(ver_vlm, _get_enh_img(), probe, cache=cache) if probe else None
            confirm["probe"] = probe
            confirm["probe_evi"] = (evp.verdict if evp else None)

            if evp and evp.verdict == "Y":
                confirm["mask"] = False
                confirm["reason"] = "probe_found_keep"
            else:
                v_enh = verify_question_lenient(
                    ver_vlm, _get_enh_img(), cand0[: cfg.max_verify_chars], cache=cache
                ).verdict
                confirm["v_enh_lenient"] = v_enh
                # Only mask whole question if enhanced view is still strong N.
                confirm["mask"] = (v_enh == "N")
                confirm["reason"] = "enh_N_confirm" if confirm["mask"] else "enh_not_N_keep"
        except Exception as e:
            confirm = {"mask": False, "reason": "confirm_err", "err": str(e)}

        dbg["mask_whole_confirm"] = confirm

        if confirm.get("mask"):
            dbg["stage"] = "mask_whole_question"
            out = f"{qno}. {cfg.mask_u_token}"
            dbg["delta_frac"] = float(1.0 - (len(out) / max(1, len(cand0))))
            return out, dbg

        # Not confirmed: keep the original markdown (most conservative).
        dbg["stage"] = "NN_not_confirmed_keep_md"
        out = _canonicalize_qno_prefix(cand, qno, mask_u_token=cfg.mask_u_token)
        dbg["delta_frac"] = float(abs(len(out) - len(cand0)) / float(max(1, len(cand0))))
        return out, dbg


    dbg["stage"] = "fallback_keep_md"
    out = _canonicalize_qno_prefix(cand, qno, mask_u_token=cfg.mask_u_token)
    dbg["delta_frac"] = float(abs(len(out) - len(cand0)) / float(max(1, len(cand0))))
    return out, dbg

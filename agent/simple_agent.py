from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from proofread.cache import JsonCache
from proofread.figures import FigureFilterCfg
from proofread.img_utils import STRONG_ENHANCE, enhance_for_vlm, img_to_data_url, safe_open_image
from proofread.match_utils import load_crop_image, load_match_questions
from proofread.pipeline import process_one_page
from proofread.vlm_client import VLMClient


DEFAULT_VLM_MODEL = os.environ.get("MTC_VLM_MODEL", "qwen3.7-plus")

BASELINE_PROMPT = """You are an OCR agent for real-world exam images.
Read the whole page and return JSON only.

Schema:
{
  "questions": [
    {
      "qno": "question number if visible, otherwise empty",
      "question_text": "printed question stem/options/formulas only; ignore handwritten notes",
      "student_answer": "student handwritten answer if visible, otherwise empty",
      "answer_status": "ok|no_answer|uncertain|unreadable"
    }
  ],
  "page_notes": "short note about image quality or occlusion"
}

Rules:
- Do not solve the problem.
- Preserve math symbols and line breaks where useful.
- Keep printed question text separate from handwritten student answers.
- If a field is not directly visible, leave it empty or mark uncertain/unreadable.
- Output valid JSON only, without markdown fences.
"""

ANSWER_PROMPT = """You are extracting student handwriting from one cropped exam-question image.
Look for handwritten student work, marks, selected options, filled blanks, calculations, or final answers.
Ignore the printed question text except when needed for context.

Return JSON only:
{
  "student_answer": "transcribed handwriting only; empty if none",
  "status": "ok|no_answer|uncertain|unreadable",
  "evidence_note": "brief visual evidence, e.g. where the handwriting appears"
}

Do not solve the problem and do not invent missing handwriting.
"""

ANSWER_VERIFY_PROMPT = """You are a handwriting evidence verifier.
Given a cropped exam-question image and a candidate student_answer, judge whether the candidate can be directly supported by visible handwritten content in the image.

Output one letter only:
Y = the candidate answer is visibly supported by handwriting in the image.
N = the candidate answer is not visible or contradicts the handwriting.
U = the image/handwriting is unclear or the candidate is too ambiguous to verify.
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_page_image(src: Path, image_root: Path) -> Path:
    image_root.mkdir(parents=True, exist_ok=True)
    dst = image_root / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def _run_stage_script(script_name: str, args: List[str]) -> None:
    root = _repo_root()
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(root / "scripts" / script_name),
        *args,
    ]
    subprocess.run(cmd, cwd=root, check=True)


def _extract_json_obj(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {"raw": raw}
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, dict) else {"raw": raw}
        except Exception:
            pass
    return {"raw": raw}


def _json_cache_key(prefix: str, model: str, img_hash: str) -> str:
    return f"{model}::{img_hash}::{prefix}"


def _invoke_image_json(
    vlm: VLMClient,
    img: Image.Image,
    prompt: str,
    *,
    cache: Optional[JsonCache],
    cache_ns: str,
    cache_prefix: str,
    max_tokens: int,
) -> Dict[str, Any]:
    img2 = enhance_for_vlm(img, STRONG_ENHANCE)
    data_url, img_hash = img_to_data_url(img2)
    ck = _json_cache_key(cache_prefix, vlm.cache_tag, img_hash)
    if cache:
        hit = cache.get(cache_ns, ck)
        if isinstance(hit, dict):
            return hit

    raw = vlm.invoke(
        [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
        temperature=0.0,
        top_p=0.7,
        max_tokens=max_tokens,
    )
    obj = _extract_json_obj(raw)
    obj["_raw"] = raw
    if cache:
        cache.set(cache_ns, ck, obj)
    return obj


def _verify_answer(
    vlm: VLMClient,
    img: Image.Image,
    candidate: str,
    *,
    cache: Optional[JsonCache],
) -> Dict[str, Any]:
    candidate = (candidate or "").strip()
    if not candidate:
        return {"verdict": "U", "raw": "", "reason": "empty_candidate"}

    img2 = enhance_for_vlm(img, STRONG_ENHANCE)
    data_url, img_hash = img_to_data_url(img2)
    ck = f"{vlm.cache_tag}::{img_hash}::answer_verify::{candidate[:240]}"
    if cache:
        hit = cache.get("answer_verify", ck)
        if isinstance(hit, dict):
            return hit

    raw = vlm.invoke(
        [
            {"type": "text", "text": ANSWER_VERIFY_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "candidate student_answer:\n" + candidate[:900]},
        ],
        temperature=0.0,
        top_p=0.7,
        max_tokens=8,
    )
    m = re.search(r"[YNU]", raw.upper())
    verdict = m.group(0) if m else "U"
    out = {"verdict": verdict, "raw": raw}
    if cache:
        cache.set("answer_verify", ck, out)
    return out


def _baseline_to_markdown(baseline: Dict[str, Any]) -> str:
    qs = baseline.get("questions")
    if not isinstance(qs, list) or not qs:
        raw = baseline.get("raw") or baseline.get("_raw") or ""
        return str(raw).strip()

    blocks: List[str] = []
    for i, q in enumerate(qs, start=1):
        if not isinstance(q, dict):
            continue
        qno = str(q.get("qno") or "").strip()
        text = str(q.get("question_text") or "").strip()
        if not text:
            continue
        if qno and not re.match(rf"^\s*{re.escape(qno)}\b", text):
            text = f"{qno}. {text}"
        elif not qno and not re.match(r"^\s*\d{1,3}\b", text):
            text = f"{i}. {text}"
        blocks.append(text)
    return "\n\n".join(blocks).strip()


def _question_sort_key(q: Dict[str, Any]) -> Tuple[int, int]:
    read_index = q.get("read_index")
    det_index = q.get("det_index")
    try:
        r = int(read_index)
    except Exception:
        r = 999999
    try:
        d = int(det_index)
    except Exception:
        d = 999999
    return r, d


def _collect_answer_evidence(
    page_dir: Path,
    vlm: VLMClient,
    *,
    cache: Optional[JsonCache],
    max_questions: int = 80,
) -> List[Dict[str, Any]]:
    _, questions = load_match_questions(page_dir / "match.json")
    items: List[Dict[str, Any]] = []
    for idx, qrec in enumerate(sorted(questions, key=_question_sort_key)[:max_questions], start=1):
        img = load_crop_image(page_dir, qrec)
        if img is None:
            items.append(
                {
                    "index": idx,
                    "read_index": qrec.get("read_index"),
                    "det_index": qrec.get("det_index"),
                    "status": "crop_open_fail",
                    "student_answer": "",
                }
            )
            continue

        ans = _invoke_image_json(
            vlm,
            img,
            ANSWER_PROMPT,
            cache=cache,
            cache_ns="answer_extract",
            cache_prefix="answer_extract",
            max_tokens=700,
        )
        student_answer = str(ans.get("student_answer") or "").strip()
        verify = _verify_answer(vlm, img, student_answer, cache=cache) if student_answer else {
            "verdict": "U",
            "reason": "empty_candidate",
        }
        items.append(
            {
                "index": idx,
                "read_index": qrec.get("read_index"),
                "det_index": qrec.get("det_index"),
                "class_name": qrec.get("class_name"),
                "score": qrec.get("score"),
                "crop_path": str(qrec.get("crop_path") or ""),
                "student_answer": student_answer,
                "answer_status": ans.get("status") or ("ok" if student_answer else "no_answer"),
                "answer_evidence_note": ans.get("evidence_note") or "",
                "answer_verify": verify,
                "raw_answer_extract": ans.get("_raw", ""),
            }
        )
    return items


def _read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def run_agent(args: argparse.Namespace) -> Dict[str, Any]:
    root = _repo_root()
    image = Path(args.image)
    if not image.exists():
        raise FileNotFoundError(image)

    image_root = Path(args.image_root)
    stage2_root = Path(args.stage2_root)
    page_md_root = Path(args.page_md_root)
    stage3_root = Path(args.stage3_root)
    agent_out_root = Path(args.out_root)
    for p in (page_md_root, stage3_root, agent_out_root):
        p.mkdir(parents=True, exist_ok=True)

    page_image = _ensure_page_image(image, image_root)
    page_name = page_image.stem
    page_dir = stage2_root / page_name
    page_agent_out = agent_out_root / page_name
    page_agent_out.mkdir(parents=True, exist_ok=True)

    cache = JsonCache(Path(args.cache_path) if args.cache_path else page_agent_out / "_agent_cache.json") if args.cache else None

    vlm = VLMClient(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        temperature=0.0,
        top_p=0.7,
    )

    page_img = safe_open_image(page_image)
    if page_img is None:
        raise RuntimeError(f"cannot open image: {page_image}")

    baseline = _invoke_image_json(
        vlm,
        page_img,
        BASELINE_PROMPT,
        cache=cache,
        cache_ns="baseline",
        cache_prefix="whole_page_baseline",
        max_tokens=args.baseline_max_tokens,
    )
    baseline_md = _baseline_to_markdown(baseline)
    page_md_path = page_md_root / f"{page_name}.md"
    page_md_path.write_text(baseline_md, encoding="utf-8")
    (page_agent_out / "baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    (page_agent_out / "baseline.md").write_text(baseline_md, encoding="utf-8")

    if not args.skip_layout:
        _run_stage_script(
            "run_stage1.ps1",
            [
                "-ImageDir",
                str(image_root),
                "-RfdetrOut",
                str(args.rfdetr_out),
                "-DoclayoutOut",
                str(args.doclayout_out),
                "-Checkpoint",
                str(args.checkpoint),
                "-DoclayoutDevice",
                str(args.doclayout_device),
            ],
        )
        _run_stage_script(
            "run_stage2.ps1",
            [
                "-ImageDir",
                str(image_root),
                "-RfdetrJsonl",
                str(Path(args.rfdetr_out) / "rfdetr_infer_results.jsonl"),
                "-DoclayoutJsonDir",
                str(Path(args.doclayout_out) / "json"),
                "-OutDir",
                str(stage2_root),
            ],
        )

    if not (page_dir / "match.json").exists():
        raise RuntimeError(f"match.json not found: {page_dir / 'match.json'}")

    proofread_out = stage3_root / page_name
    fig_cfg = FigureFilterCfg(
        min_edge=28,
        max_aspect=8.0,
        max_blank_frac=0.97,
        do_vlm_cls=False if args.no_fig else True,
        do_vlm_rel=False if args.no_fig else True,
    )
    process_one_page(
        page_dir,
        page_md_path,
        out_dir=proofread_out,
        ver_vlm=vlm,
        gen_vlm=None if args.no_patcher else vlm,
        fig_vlm=None if args.no_fig else vlm,
        cache=cache,
        skip_partial=True,
        partial_main_min_hfrac=0.12,
        use_offset_search=True,
        use_crop_qno=bool(args.use_crop_qno),
        qno_vlm=vlm if args.use_crop_qno else None,
        mask_u_token="[UNREADABLE]",
        mask_n_token="[HALLUCINATION]",
        fig_cfg=fig_cfg,
        ablation_no_patcher=bool(args.no_patcher),
        verdict_comment=bool(args.verdict_comment),
    )

    answer_items = _collect_answer_evidence(page_dir, vlm, cache=cache, max_questions=args.max_questions)
    (page_agent_out / "answer_evidence.json").write_text(
        json.dumps(answer_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    proofread_md_path = proofread_out / f"{page_name}_proofread.md"
    proofread_report_path = proofread_out / f"{page_name}_report.json"
    proofread_md = _read_text_if_exists(proofread_md_path)
    proofread_report = {}
    if proofread_report_path.exists():
        try:
            proofread_report = json.loads(proofread_report_path.read_text(encoding="utf-8"))
        except Exception:
            proofread_report = {}

    final = {
        "page": page_name,
        "image": str(page_image),
        "baseline": baseline,
        "verified_question_markdown": proofread_md,
        "question_verification_report": proofread_report,
        "answer_evidence": answer_items,
        "outputs": {
            "baseline_json": str(page_agent_out / "baseline.json"),
            "baseline_md": str(page_agent_out / "baseline.md"),
            "proofread_md": str(proofread_md_path),
            "proofread_report": str(proofread_report_path),
            "answer_evidence": str(page_agent_out / "answer_evidence.json"),
        },
    }
    (page_agent_out / "final_result.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [f"# {page_name}", "", "## Verified Questions", "", proofread_md.strip(), "", "## Student Answer Evidence"]
    for item in answer_items:
        ans = item.get("student_answer") or ""
        status = item.get("answer_status") or ""
        verdict = (item.get("answer_verify") or {}).get("verdict")
        md_lines.append("")
        md_lines.append(f"- crop {item.get('index')} read_index={item.get('read_index')} status={status} verify={verdict}")
        if ans:
            md_lines.append(f"  answer: {ans}")
    (page_agent_out / "final_result.md").write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    if cache:
        cache.save()
    return final


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("simple_exam_agent")
    p.add_argument("--image", required=True, help="input exam page image")
    p.add_argument("--image-root", default="workflow/images")
    p.add_argument("--rfdetr-out", default="workflow/stage1_rfdetr")
    p.add_argument("--doclayout-out", default="workflow/stage1_doclayout")
    p.add_argument("--stage2-root", default="workflow/stage2_match")
    p.add_argument("--page-md-root", default="workflow/stage3_page_md")
    p.add_argument("--stage3-root", default="workflow/stage3_out")
    p.add_argument("--out-root", default="workflow/agent_out")
    p.add_argument("--checkpoint", default="checkpoint_best_total.pth")
    p.add_argument("--doclayout-device", default="cpu")
    p.add_argument("--skip-layout", action="store_true", help="reuse existing Stage 1/2 outputs")
    p.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--api-key", default="")
    p.add_argument("--model", default=DEFAULT_VLM_MODEL)
    p.add_argument("--baseline-max-tokens", type=int, default=3000)
    p.add_argument("--max-questions", type=int, default=80)
    p.add_argument("--use-crop-qno", action="store_true")
    p.add_argument("--no-patcher", action="store_true", default=True)
    p.add_argument("--with-patcher", dest="no_patcher", action="store_false")
    p.add_argument("--no-fig", action="store_true", default=True)
    p.add_argument("--with-fig", dest="no_fig", action="store_false")
    p.add_argument("--verdict-comment", action="store_true")
    p.add_argument("--cache", action="store_true", default=True)
    p.add_argument("--no-cache", dest="cache", action="store_false")
    p.add_argument("--cache-path", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    final = run_agent(args)
    print(json.dumps(final.get("outputs", {}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

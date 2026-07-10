from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from proofread.cache import JsonCache
from proofread.figures import FigureFilterCfg
from proofread.img_utils import (
    STRONG_ENHANCE,
    enhance_for_vlm,
    img_to_data_url,
    safe_open_image,
    scan_document_for_ocr,
)
from proofread.match_utils import load_crop_image, load_match_questions
from proofread.md_utils import split_page_into_blocks
from proofread.pipeline import process_one_page
from proofread.vlm_client import VLMClient


DEFAULT_VLM_MODEL = os.environ.get("MTC_VLM_MODEL", "qwen3.7-plus")


@dataclass(frozen=True)
class WorkflowPaths:
    """The four user-facing output groups and their internal subdirectories."""

    root: Path

    @property
    def preprocessed(self) -> Path:
        return self.root / "preprocessed"

    @property
    def api_markdown(self) -> Path:
        return self.root / "api_markdown"

    @property
    def code_outputs(self) -> Path:
        return self.root / "code_outputs"

    @property
    def rfdetr(self) -> Path:
        return self.code_outputs / "rfdetr"

    @property
    def doclayout(self) -> Path:
        return self.code_outputs / "doclayout"

    @property
    def match(self) -> Path:
        return self.code_outputs / "match"

    @property
    def agent_outputs(self) -> Path:
        return self.root / "agent_outputs"

    @property
    def cache(self) -> Path:
        return _repo_root() / ".cache" / "mathocrclaw"

    def ensure(self) -> None:
        for path in (self.preprocessed, self.api_markdown, self.code_outputs, self.agent_outputs, self.cache):
            path.mkdir(parents=True, exist_ok=True)


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


def _collect_answer_evidence(
    page_dir: Path,
    vlm: VLMClient,
    verification_items: List[Dict[str, Any]],
    *,
    cache: Optional[JsonCache],
) -> List[Dict[str, Any]]:
    _, questions = load_match_questions(page_dir / "match.json")
    items: List[Dict[str, Any]] = []
    targets = [item for item in verification_items if isinstance(item, dict) and item.get("kind") == "q"]
    for idx, target in enumerate(targets, start=1):
        try:
            match_qi = int(target.get("mapped_match_qi"))
            qrec = questions[match_qi]
        except (TypeError, ValueError, IndexError):
            items.append(
                {
                    "qno": target.get("qno"),
                    "question_index": idx,
                    "text": "",
                    "status": "crop_missing",
                    "verdict": "U",
                    "evidence_note": "No aligned question crop was available.",
                    "crop_path": "",
                }
            )
            continue

        img = load_crop_image(page_dir, qrec)
        if img is None:
            items.append(
                {
                    "qno": target.get("qno"),
                    "question_index": idx,
                    "text": "",
                    "status": "crop_open_fail",
                    "verdict": "U",
                    "evidence_note": "The aligned question crop could not be opened.",
                    "crop_path": str(qrec.get("crop_path") or ""),
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
                "qno": target.get("qno"),
                "question_index": idx,
                "read_index": qrec.get("read_index"),
                "det_index": qrec.get("det_index"),
                "crop_path": str(page_dir / str(qrec.get("crop_path") or "")),
                "text": student_answer,
                "status": ans.get("status") or ("ok" if student_answer else "no_answer"),
                "verdict": verify.get("verdict") or "U",
                "evidence_note": ans.get("evidence_note") or "",
            }
        )
    return items


def _build_question_results(
    verified_markdown: str,
    verification_report: Dict[str, Any],
    answer_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    blocks_by_qno = {
        int(block.qno): block.text
        for block in split_page_into_blocks(verified_markdown)
        if block.kind == "q" and block.qno is not None
    }
    answers_by_qno = {item.get("qno"): item for item in answer_items}
    results: List[Dict[str, Any]] = []
    for item in verification_report.get("items") or []:
        if not isinstance(item, dict) or item.get("kind") != "q":
            continue
        qno = item.get("qno")
        try:
            question_markdown = blocks_by_qno.get(int(qno), "")
        except (TypeError, ValueError):
            question_markdown = ""
        answer = answers_by_qno.get(qno) or {
            "qno": qno,
            "text": "",
            "status": "uncertain",
            "verdict": "U",
            "evidence_note": "No answer evidence record was produced.",
            "crop_path": "",
        }
        repair = item.get("repair") if isinstance(item.get("repair"), dict) else {}
        results.append(
            {
                "qno": qno,
                "question_markdown": question_markdown,
                "question_verification": {
                    "status": item.get("status") or "unknown",
                    "verdict": item.get("v_after_repair") or repair.get("v_strict") or "U",
                    "crop_qno": item.get("crop_qno"),
                    "weak_crop": bool(item.get("weak_crop")),
                },
                "handwritten_answer": {
                    "text": answer.get("text") or "",
                    "status": answer.get("status") or "uncertain",
                    "verdict": answer.get("verdict") or "U",
                    "evidence_note": answer.get("evidence_note") or "",
                    "crop_path": answer.get("crop_path") or "",
                },
            }
        )
    return results


def _result_summary(questions: List[Dict[str, Any]]) -> Dict[str, int]:
    answers = [q.get("handwritten_answer") or {} for q in questions]
    return {
        "question_count": len(questions),
        "answers_supported": sum(1 for answer in answers if answer.get("verdict") == "Y"),
        "answers_rejected": sum(1 for answer in answers if answer.get("verdict") == "N"),
        "answers_uncertain_or_empty": sum(1 for answer in answers if answer.get("verdict") == "U"),
    }


def _render_result_markdown(page_name: str, questions: List[Dict[str, Any]]) -> str:
    lines = [f"# {page_name}"]
    for index, question in enumerate(questions, start=1):
        qno = question.get("qno")
        label = str(qno) if qno is not None else str(index)
        verification = question.get("question_verification") or {}
        answer = question.get("handwritten_answer") or {}
        lines.extend(
            [
                "",
                f"## 题目 {label}",
                "",
                str(question.get("question_markdown") or "[UNREADABLE]").strip(),
                "",
                "### 手写答案",
                "",
            ]
        )
        answer_text = str(answer.get("text") or "").strip()
        lines.append(answer_text if answer_text else "_未识别到可验证的手写答案。_")
        lines.extend(
            [
                "",
                f"- 题干校验：`{verification.get('verdict') or 'U'}`",
                f"- 答案状态：`{answer.get('status') or 'uncertain'}`",
                f"- 答案证据：`{answer.get('verdict') or 'U'}`",
            ]
        )
        evidence_note = str(answer.get("evidence_note") or "").strip()
        if evidence_note:
            lines.append(f"- 证据说明：{evidence_note}")
    return "\n".join(lines).strip() + "\n"


def _read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def run_agent(args: argparse.Namespace) -> Dict[str, Any]:
    image = Path(args.image).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)

    paths = WorkflowPaths(Path(args.work_root))
    paths.ensure()
    page_name = image.stem
    page_dir = paths.match / page_name
    page_agent_out = paths.agent_outputs / page_name
    page_agent_out.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache_path) if args.cache_path else paths.cache / f"{page_name}.json"
    cache = JsonCache(cache_path) if args.cache else None

    vlm = VLMClient(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        temperature=0.0,
        top_p=0.7,
    )

    page_img = safe_open_image(image)
    if page_img is None:
        raise RuntimeError(f"cannot open image: {image}")

    preprocessed_img, preprocess_meta = scan_document_for_ocr(page_img)
    preprocessed_image = paths.preprocessed / f"{page_name}.png"
    preprocess_report_path = paths.preprocessed / f"{page_name}.json"
    preprocessed_img.save(preprocessed_image, format="PNG", optimize=True)
    preprocess_report_path.write_text(
        json.dumps(preprocess_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    baseline = _invoke_image_json(
        vlm,
        preprocessed_img,
        BASELINE_PROMPT,
        cache=cache,
        cache_ns="baseline",
        cache_prefix="whole_page_baseline",
        max_tokens=args.baseline_max_tokens,
    )
    baseline_md = _baseline_to_markdown(baseline)
    page_md_path = paths.api_markdown / f"{page_name}.md"
    baseline_json_path = paths.api_markdown / f"{page_name}.json"
    page_md_path.write_text(baseline_md, encoding="utf-8")
    baseline_json_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    if cache:
        cache.save()

    if not args.skip_layout:
        _run_stage_script(
            "run_stage1.ps1",
            [
                "-Image",
                str(preprocessed_image),
                "-RfdetrOut",
                str(paths.rfdetr),
                "-DoclayoutOut",
                str(paths.doclayout),
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
                str(paths.preprocessed),
                "-RfdetrJsonl",
                str(paths.rfdetr / "rfdetr_infer_results.jsonl"),
                "-DoclayoutJsonDir",
                str(paths.doclayout / "json"),
                "-OutDir",
                str(paths.match),
            ],
        )

    if not (page_dir / "match.json").exists():
        raise RuntimeError(f"match.json not found: {page_dir / 'match.json'}")

    proofread_out = paths.cache / "verification" / page_name
    fig_cfg = FigureFilterCfg(
        min_edge=28,
        max_aspect=8.0,
        max_blank_frac=0.97,
        do_vlm_cls=False if args.no_fig else True,
        do_vlm_rel=False if args.no_fig else True,
    )
    proofread_report = process_one_page(
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

    proofread_md_path = proofread_out / f"{page_name}_proofread.md"
    proofread_md = _read_text_if_exists(proofread_md_path)
    if not proofread_md.strip():
        raise RuntimeError(f"verified question Markdown is empty: {proofread_md_path}")

    answer_items = _collect_answer_evidence(
        page_dir,
        vlm,
        proofread_report.get("items") or [],
        cache=cache,
    )
    questions = _build_question_results(proofread_md, proofread_report, answer_items)
    if not questions:
        raise RuntimeError("no aligned question-answer results were produced")

    final = {
        "page": page_name,
        "source_image": str(image),
        "preprocessed_image": str(preprocessed_image),
        "summary": _result_summary(questions),
        "questions": questions,
        "artifacts": {
            "preprocessing_report": str(preprocess_report_path),
            "api_markdown": str(page_md_path),
            "api_response": str(baseline_json_path),
            "code_outputs": str(paths.code_outputs),
            "verification_report": str(page_agent_out / "verification.json"),
        },
    }
    result_json_path = page_agent_out / "result.json"
    result_md_path = page_agent_out / "result.md"
    verification_path = page_agent_out / "verification.json"
    result_json_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    result_md_path.write_text(_render_result_markdown(page_name, questions), encoding="utf-8")
    verification_path.write_text(json.dumps(proofread_report, ensure_ascii=False, indent=2), encoding="utf-8")

    if cache:
        cache.save()
    final["outputs"] = {
        "result_json": str(result_json_path),
        "result_markdown": str(result_md_path),
        "verification_report": str(verification_path),
    }
    return final


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("mathocr_workflow")
    p.add_argument("--image", required=True, help="input exam page image")
    p.add_argument("--work-root", default="workflow", help="root containing the four output groups")
    p.add_argument("--checkpoint", default="checkpoint_best_total.pth")
    p.add_argument("--doclayout-device", default="cpu")
    p.add_argument("--skip-layout", action="store_true", help="reuse existing Stage 1/2 outputs")
    p.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--api-key", default="")
    p.add_argument("--model", default=DEFAULT_VLM_MODEL)
    p.add_argument("--baseline-max-tokens", type=int, default=3000)
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

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from agent.handwriting_regions import (
    draw_handwriting_overlay,
    save_handwriting_views,
    score_and_divide_question_frame,
)
from proofread.cache import JsonCache
from proofread.common import sha1_bytes
from proofread.img_utils import (
    STRONG_ENHANCE,
    enhance_for_vlm,
    img_to_data_url,
    safe_open_image,
    scan_document_for_ocr,
)
from proofread.match_utils import load_match_questions
from proofread.vlm_client import VLMClient


DEFAULT_VLM_MODEL = os.environ.get("MTC_VLM_MODEL", "qwen3.7-plus")


@dataclass(frozen=True)
class WorkflowPaths:
    """The user-facing output groups and their internal subdirectories."""

    root: Path

    @property
    def image(self) -> Path:
        return self.root / "image"

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
        for path in (
            self.image,
            self.preprocessed,
            self.api_markdown,
            self.code_outputs,
            self.agent_outputs,
            self.cache,
        ):
            path.mkdir(parents=True, exist_ok=True)


BASELINE_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "benchmark" / "prompts" / "extract_v2.txt"
)
BASELINE_PROMPT = BASELINE_PROMPT_PATH.read_text(encoding="utf-8").strip()


FINAL_REVIEW_PROMPT = """You are the second and final OCR pass for an exam page.
You receive the first-pass whole-page Markdown, the original normalized page image, locally detected layout metadata, and labeled per-question context images. Review the draft globally and return the final question/handwriting result.

Return JSON only:
{
  "page_notes": "brief final note about unresolved page-level ambiguity",
  "questions": [
    {
      "qno": "visible question number, otherwise stable reading-order number",
      "question_markdown": "corrected printed question stem/options/formulas only",
      "handwritten_answer": {
        "text": "complete faithful Markdown/LaTeX transcription of visible handwriting in reading order; empty if none",
        "answer_parts": [
          {
            "label": "subquestion label such as (1), or overall",
            "transcription": "all visible work belonging to this part",
            "final_answer": "visible final answer if identifiable, otherwise empty",
            "status": "ok|partial|uncertain|unreadable"
          }
        ],
        "uncertain_fragments": [
          {
            "text": "best-effort visible fragment",
            "alternatives": ["other plausible readings"],
            "location": "where it appears"
          }
        ],
        "status": "ok|partial|no_answer|uncertain|unreadable",
        "evidence_note": "brief visual account of handwriting, corrections, or ambiguity"
      },
      "source_context_ids": ["IDs of local context images used for this question"]
    }
  ]
}

Rules:
- Produce exactly one record per real visible question, in page reading order.
- Correct omissions, hallucinations, question boundaries, formulas, and handwriting assignment in the first draft.
- Treat detector boxes, scores, classes, and ordinal pairing as routing hints, not truth.
- Use the full-page image to resolve cross-question ownership and the labeled context images for detail.
- Do not solve, grade, or replace the student's work with a mathematically correct solution.
- Preserve non-deleted intermediate work. If writing is clearly crossed out, erased,
  overwritten, struck through, or otherwise cancelled, omit it completely and keep
  only the final retained version.
- Never reproduce cancelled content in text, answer_parts, uncertain_fragments, or
  evidence_note. Never emit `划去`, `已划掉`, `crossed out`, `deleted`, or `~~...~~`.
- Keep genuine visual ambiguity as uncertain_fragments, but do not treat crossed-out
  content as an ambiguity candidate.
- Printed question_markdown must not absorb handwriting; handwritten_answer must not silently summarize printed text.
- question_markdown must begin at line start with `N. ` using the visible question
  number. Exclude section titles, page headers, score summaries, and neighboring text.
- Every mathematical expression in question_markdown, handwritten_answer.text,
  transcriptions, and final answers must use standard LaTeX inside `$...$` or
  `$$...$$`; never replace LaTeX with Unicode/plain-text pseudo-formulas.
- Transcribe `∵`/`∴` faithfully as `\because`/`\therefore` or the visible symbols,
  preserve one logical handwritten step per line, and insert `<插图>` wherever the
  printed question has a diagram.
- The Markdown content must follow benchmark/prompts/extract_v2.txt: no `[模糊]`,
  no alternate guesses in the main transcription, and no editorial annotations.
- Output valid JSON only, without markdown fences.
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _page_work_root(work_root: Path, page_name: str) -> Path:
    """Return the page-owned root while avoiding accidental page/page nesting."""
    root = work_root.expanduser().resolve()
    return root if root.name == page_name else root / page_name


def _run_stage_script(script_name: str, args: List[str]) -> None:
    root = _repo_root()
    cmd = ["bash", str(root / "scripts" / script_name), *args]
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


def _strip_markdown_fence(text: str) -> str:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:markdown|md)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _invoke_image_markdown(
    vlm: VLMClient,
    img: Image.Image,
    prompt: str,
    *,
    cache: Optional[JsonCache],
    cache_ns: str,
    cache_prefix: str,
    max_tokens: int,
) -> str:
    img2 = enhance_for_vlm(img, STRONG_ENHANCE)
    data_url, img_hash = img_to_data_url(img2)
    prompt_hash = sha1_bytes(prompt.encode("utf-8"))
    cache_key = f"{vlm.cache_tag}::{img_hash}::{cache_prefix}::{prompt_hash}"
    if cache:
        hit = cache.get(cache_ns, cache_key)
        if isinstance(hit, dict) and isinstance(hit.get("markdown"), str):
            return str(hit["markdown"])

    raw = vlm.invoke(
        [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
        temperature=0.0,
        top_p=0.7,
        max_tokens=max_tokens,
    )
    markdown = _strip_markdown_fence(raw)
    if cache:
        cache.set(cache_ns, cache_key, {"markdown": markdown})
    return markdown


def _question_key(value: Any) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d+", text):
        return str(int(text))
    return text


def _normalize_qno(value: Any, fallback: int) -> Any:
    key = _question_key(value)
    return int(key) if key.isdigit() else (key or fallback)


def _draft_questions(baseline: Dict[str, Any]) -> List[Dict[str, str]]:
    questions: List[Dict[str, str]] = []
    for question in baseline.get("questions") or []:
        if not isinstance(question, dict):
            continue
        questions.append(
            {
                "qno": str(question.get("qno") or "").strip(),
                "question_text": str(question.get("question_text") or "").strip(),
                "student_answer": str(question.get("student_answer") or "").strip(),
                "answer_status": str(question.get("answer_status") or "uncertain").strip(),
            }
        )
    if questions:
        return questions
    return _draft_questions_from_markdown(str(baseline.get("page_markdown") or ""))


_DRAFT_QUESTION_RE = re.compile(r"(?m)^\s*(\d{1,3})[.．、]\s*")
_DRAFT_ANSWER_RE = re.compile(r"(?im)^\s*#{1,6}\s*手写答案\s*$")


def _draft_questions_from_markdown(markdown: str) -> List[Dict[str, str]]:
    starts = list(_DRAFT_QUESTION_RE.finditer(markdown or ""))
    questions: List[Dict[str, str]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(markdown)
        block = markdown[start.start() : end].strip()
        answer_heading = _DRAFT_ANSWER_RE.search(block)
        if answer_heading:
            question_text = block[: answer_heading.start()].strip()
            student_answer = block[answer_heading.end() :].strip()
        else:
            question_text = block
            student_answer = ""
        no_answer = not student_answer or "未识别到手写答案" in student_answer
        questions.append(
            {
                "qno": str(int(start.group(1))),
                "question_text": question_text,
                "student_answer": student_answer,
                "answer_status": "no_answer" if no_answer else "ok",
            }
        )
    return questions


def _is_main_question(question: Dict[str, Any]) -> bool:
    class_name = str(question.get("class_name") or "").strip().lower()
    return class_name != "partial_question" and question.get("read_index") is not None


def _prepare_question_contexts(
    page_dir: Path,
    page_img: Image.Image,
    baseline: Dict[str, Any],
) -> List[Dict[str, Any]]:
    _, all_questions = load_match_questions(page_dir / "match.json")
    draft_questions = _draft_questions(baseline)
    main_indices = [index for index, question in enumerate(all_questions) if _is_main_question(question)]
    contexts: List[Dict[str, Any]] = []

    for ordinal, match_index in enumerate(main_indices, start=1):
        question = all_questions[match_index]
        draft = draft_questions[ordinal - 1] if ordinal <= len(draft_questions) else {
            "qno": "",
            "question_text": "",
            "student_answer": "",
            "answer_status": "not_available",
        }
        context_id = f"C{ordinal:03d}"
        views: List[Dict[str, Any]] = []
        region: Dict[str, Any] = {}
        context_error = ""
        try:
            region = score_and_divide_question_frame(
                page_img.size,
                all_questions,
                match_index,
                question_text=draft["question_text"],
            )
            crop_path = str(question.get("crop_path") or "").replace("\\", "/")
            output_dir = (
                (page_dir / crop_path).parent / "handwriting"
                if crop_path
                else page_dir / "contexts" / context_id
            )
            views = save_handwriting_views(page_img, region, output_dir)
        except (IndexError, TypeError, ValueError, OSError) as exc:
            context_error = str(exc)

        contexts.append(
            {
                "context_id": context_id,
                "draft": draft,
                "alignment": {
                    "paired_by": "reading_order_ordinal",
                    "match_index": match_index,
                    "read_index": question.get("read_index"),
                    "det_index": question.get("det_index"),
                    "class_name": question.get("class_name") or "",
                    "detector_score": question.get("score"),
                    "source_bbox_xyxy": region.get("source_bbox_xyxy") or [],
                    "answer_bbox_xyxy": region.get("frame_bbox_xyxy") or [],
                    "question_type": region.get("question_type") or {},
                    "strategy": region.get("strategy") or "",
                    "boundary_kind": region.get("boundary_kind") or "",
                    "context_error": context_error,
                },
                "visual_context": {"views": views},
            }
        )

    for ordinal in range(len(main_indices) + 1, len(draft_questions) + 1):
        contexts.append(
            {
                "context_id": f"C{ordinal:03d}",
                "draft": draft_questions[ordinal - 1],
                "alignment": {
                    "paired_by": "unmatched_first_pass_question",
                    "match_index": None,
                    "read_index": None,
                    "det_index": None,
                    "class_name": "",
                    "detector_score": None,
                    "source_bbox_xyxy": [],
                    "answer_bbox_xyxy": [],
                    "question_type": {},
                    "strategy": "",
                    "boundary_kind": "",
                    "context_error": "No local question detection was paired.",
                },
                "visual_context": {"views": []},
            }
        )
    return contexts


def _review_context_digest(contexts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    digest: List[Dict[str, Any]] = []
    for context in contexts:
        digest.append(
            {
                "context_id": context.get("context_id"),
                "draft": context.get("draft") or {},
                "alignment": context.get("alignment") or {},
                "available_views": [
                    {
                        "kind": view.get("kind"),
                        "bbox_xyxy": view.get("bbox_xyxy") or [],
                        "purpose": view.get("purpose") or "",
                    }
                    for view in (context.get("visual_context") or {}).get("views") or []
                ],
            }
        )
    return digest


def _selected_review_views(
    context: Dict[str, Any],
    detail_views: int,
) -> List[Dict[str, Any]]:
    views = (context.get("visual_context") or {}).get("views") or []
    primary = [view for view in views if view.get("kind") == "context"][:1]
    details = [
        view
        for view in views
        if str(view.get("kind") or "").startswith("answer_detail_")
    ][: max(0, detail_views)]
    selected = primary + details
    selected_ids = {id(view) for view in selected}
    for view in views:
        view["sent_to_second_api"] = id(view) in selected_ids
    return selected


def _invoke_final_review(
    vlm: VLMClient,
    page_img: Image.Image,
    draft_markdown: str,
    baseline: Dict[str, Any],
    contexts: List[Dict[str, Any]],
    *,
    cache: Optional[JsonCache],
    max_tokens: int,
    detail_views: int = 0,
) -> Dict[str, Any]:
    review_payload = {
        "first_pass_markdown": draft_markdown,
        "first_pass_page_notes": baseline.get("page_notes") or "",
        "local_question_contexts": _review_context_digest(contexts),
    }
    prompt = FINAL_REVIEW_PROMPT + "\nReview payload:\n" + json.dumps(
        review_payload,
        ensure_ascii=False,
        indent=2,
    )
    messages: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    image_hashes: List[str] = []

    page_for_api = enhance_for_vlm(page_img, STRONG_ENHANCE)
    page_url, page_hash = img_to_data_url(page_for_api)
    image_hashes.append(page_hash)
    messages.extend(
        [
            {"type": "text", "text": "Image: full normalized page"},
            {"type": "image_url", "image_url": {"url": page_url}},
        ]
    )

    for context in contexts:
        context_id = str(context.get("context_id") or "")
        for view in _selected_review_views(context, detail_views):
            image = safe_open_image(view.get("path"))
            if image is None:
                continue
            image_for_api = enhance_for_vlm(image, STRONG_ENHANCE)
            data_url, image_hash = img_to_data_url(image_for_api)
            image_hashes.append(image_hash)
            messages.extend(
                [
                    {
                        "type": "text",
                        "text": f"Image: {context_id} / {view.get('kind') or 'context'}",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            )

    prompt_hash = sha1_bytes(prompt.encode("utf-8"))
    cache_key = f"{vlm.cache_tag}::global_review_v1::{prompt_hash}::{'::'.join(image_hashes)}"
    if cache:
        hit = cache.get("global_final_review", cache_key)
        if isinstance(hit, dict):
            return hit

    raw = vlm.invoke(
        messages,
        temperature=0.0,
        top_p=0.7,
        max_tokens=max_tokens,
    )
    review = _extract_json_obj(raw)
    review["_raw"] = raw
    if cache:
        cache.set("global_final_review", cache_key, review)
    return review


def _normalize_answer_parts(value: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for part in value if isinstance(value, list) else []:
        if not isinstance(part, dict):
            continue
        normalized.append(
            {
                "label": str(part.get("label") or "overall").strip(),
                "transcription": str(part.get("transcription") or "").strip(),
                "final_answer": str(part.get("final_answer") or "").strip(),
                "status": str(part.get("status") or "uncertain").strip(),
            }
        )
    return normalized


def _normalize_uncertain_fragments(value: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for fragment in value if isinstance(value, list) else []:
        if isinstance(fragment, str):
            normalized.append({"text": fragment.strip(), "alternatives": [], "location": ""})
            continue
        if not isinstance(fragment, dict):
            continue
        alternatives = fragment.get("alternatives")
        normalized.append(
            {
                "text": str(fragment.get("text") or "").strip(),
                "alternatives": [
                    str(alternative).strip()
                    for alternative in alternatives
                    if str(alternative).strip()
                ]
                if isinstance(alternatives, list)
                else [],
                "location": str(fragment.get("location") or "").strip(),
            }
        )
    return normalized


def _normalize_final_questions(
    review: Dict[str, Any],
    contexts: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_questions = review.get("questions")
    if not isinstance(raw_questions, list):
        return []

    context_ids = [str(context.get("context_id") or "") for context in contexts]
    questions: List[Dict[str, Any]] = []
    for index, raw_question in enumerate(raw_questions, start=1):
        if not isinstance(raw_question, dict):
            continue
        fallback_context = contexts[index - 1] if index <= len(contexts) else {}
        draft = (
            fallback_context.get("draft")
            if isinstance(fallback_context.get("draft"), dict)
            else {}
        )
        raw_answer = raw_question.get("handwritten_answer")
        answer = raw_answer if isinstance(raw_answer, dict) else {}
        parts = _normalize_answer_parts(answer.get("answer_parts"))
        uncertain = _normalize_uncertain_fragments(answer.get("uncertain_fragments"))
        text = str(answer.get("text") or raw_question.get("student_answer") or "").strip()
        if not text and parts:
            rendered_parts = []
            for part in parts:
                content = part["transcription"] or part["final_answer"]
                if content:
                    rendered_parts.append(
                        f"{part['label']} {content}"
                        if part["label"] != "overall"
                        else content
                    )
            text = "\n".join(rendered_parts)

        source_ids = raw_question.get("source_context_ids")
        normalized_source_ids = (
            [
                str(context_id).strip()
                for context_id in source_ids
                if str(context_id).strip() in context_ids
            ]
            if isinstance(source_ids, list)
            else []
        )
        if not normalized_source_ids and fallback_context:
            normalized_source_ids = [str(fallback_context.get("context_id") or "")]

        question_markdown = str(raw_question.get("question_markdown") or "").strip()
        if not question_markdown:
            question_markdown = str(draft.get("question_text") or "").strip()
        status = str(answer.get("status") or ("ok" if text else "no_answer")).strip()
        questions.append(
            {
                "qno": _normalize_qno(
                    raw_question.get("qno") or draft.get("qno"),
                    index,
                ),
                "question_markdown": question_markdown,
                "question_review": {
                    "status": "reviewed_in_global_second_pass",
                    "source_context_ids": normalized_source_ids,
                },
                "handwritten_answer": {
                    "text": text,
                    "answer_parts": parts,
                    "uncertain_fragments": uncertain,
                    "status": status,
                    "evidence_note": str(answer.get("evidence_note") or "").strip(),
                    "source_context_ids": normalized_source_ids,
                },
            }
        )
    return questions


def _build_question_context_manifest(
    page_name: str,
    source_image: Path,
    draft_markdown_path: Path,
    contexts: List[Dict[str, Any]],
    final_questions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    final_by_context: Dict[str, List[Dict[str, Any]]] = {}
    for question in final_questions:
        source_ids = (question.get("question_review") or {}).get("source_context_ids") or []
        for context_id in source_ids:
            final_by_context.setdefault(str(context_id), []).append(
                {
                    "qno": question.get("qno"),
                    "question_markdown": question.get("question_markdown") or "",
                    "handwritten_answer": question.get("handwritten_answer") or {},
                }
            )

    manifest_contexts: List[Dict[str, Any]] = []
    for context in contexts:
        context_id = str(context.get("context_id") or "")
        manifest_contexts.append(
            {
                **context,
                "consumed_by_second_api": any(
                    bool(view.get("sent_to_second_api"))
                    for view in (context.get("visual_context") or {}).get("views") or []
                ),
                "final_records": final_by_context.get(context_id, []),
            }
        )
    return {
        "schema_version": 3,
        "page": page_name,
        "source_image": str(source_image),
        "first_pass_markdown": str(draft_markdown_path),
        "api_strategy": {
            "call_1": "whole-page draft OCR",
            "between_calls": "local detection, layout, context construction",
            "call_2": "single global correction producing final result",
        },
        "contexts": manifest_contexts,
    }


def _result_summary(questions: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    answers = [question.get("handwritten_answer") or {} for question in questions]
    return {
        "question_count": len(questions),
        "answers_with_transcription": sum(1 for answer in answers if answer.get("text")),
        "answers_partial_or_uncertain": sum(
            1
            for answer in answers
            if answer.get("status") in {"partial", "uncertain", "unreadable"}
        ),
        "answers_empty": sum(1 for answer in answers if not answer.get("text")),
    }


def _api_metrics(
    first_pass_vlm: VLMClient,
    second_pass_vlm: VLMClient,
) -> Dict[str, Any]:
    passes = {
        "call_1": list(first_pass_vlm.request_log),
        "call_2": list(second_pass_vlm.request_log),
    }
    requests = [request for values in passes.values() for request in values]

    def _token_sum(name: str) -> Optional[int]:
        values = [request.get(name) for request in requests if request.get(name) is not None]
        return sum(int(value) for value in values) if values else None

    return {
        "logical_call_count": 2,
        "network_request_count": len(requests),
        "successful_request_count": sum(bool(request.get("success")) for request in requests),
        "failed_request_count": sum(not bool(request.get("success")) for request in requests),
        "prompt_tokens": _token_sum("prompt_tokens"),
        "completion_tokens": _token_sum("completion_tokens"),
        "total_tokens": _token_sum("total_tokens"),
        "latency_s": round(sum(float(request.get("elapsed_s") or 0.0) for request in requests), 6),
        "passes": passes,
    }


def _render_result_markdown(page_name: str, questions: Sequence[Dict[str, Any]]) -> str:
    del page_name
    lines: List[str] = []
    for index, question in enumerate(questions, start=1):
        qno = question.get("qno")
        label = str(qno) if qno is not None else str(index)
        answer = question.get("handwritten_answer") or {}
        question_markdown = str(question.get("question_markdown") or "[无法识别]").strip()
        if not re.match(rf"^\s*{re.escape(label)}[.．、]\s*", question_markdown):
            question_markdown = f"{label}. {question_markdown}"
        if lines:
            lines.append("")
        lines.extend([question_markdown, "", "### 手写答案", ""])
        answer_text = str(answer.get("text") or "").strip()
        if answer_text:
            lines.append(answer_text)
        elif str(answer.get("status") or "").strip() in {"uncertain", "unreadable"}:
            lines.append("[无法识别]")
        else:
            lines.append("_未识别到手写答案。_")
    return "\n".join(lines).strip() + "\n"


def run_agent(args: argparse.Namespace) -> Dict[str, Any]:
    image = Path(args.image).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)

    page_name = image.stem
    paths = WorkflowPaths(_page_work_root(Path(args.work_root), page_name))
    paths.ensure()
    original_image = paths.image / image.name
    if image != original_image.resolve():
        shutil.copy2(image, original_image)
    page_dir = paths.match
    page_agent_out = paths.agent_outputs
    page_agent_out.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache_path) if args.cache_path else paths.cache / f"{page_name}.json"
    cache = JsonCache(cache_path) if args.cache else None
    first_pass_vlm = VLMClient(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        max_retries=1,
        temperature=0.0,
        top_p=0.7,
    )
    second_pass_vlm = VLMClient(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        timeout_s=args.review_timeout,
        max_retries=1,
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

    # API call 1: the canonical benchmark prompt returns Markdown directly.
    draft_markdown = _invoke_image_markdown(
        first_pass_vlm,
        preprocessed_img,
        BASELINE_PROMPT,
        cache=cache,
        cache_ns="baseline_markdown_v3",
        cache_prefix="benchmark_extract_v2",
        max_tokens=args.baseline_max_tokens,
    )
    if not draft_markdown:
        raise RuntimeError("first-pass whole-page Markdown is empty")
    baseline = {
        "schema_version": 3,
        "response_format": "benchmark_markdown",
        "prompt": {
            "version": "extract_v2",
            "path": str(BASELINE_PROMPT_PATH.relative_to(_repo_root())),
            "sha1": sha1_bytes(BASELINE_PROMPT.encode("utf-8")),
        },
        "page_markdown": draft_markdown,
        "questions": _draft_questions_from_markdown(draft_markdown),
        "page_notes": "",
    }
    draft_markdown_path = paths.api_markdown / f"{page_name}.md"
    baseline_json_path = paths.api_markdown / f"{page_name}.json"
    draft_markdown_path.write_text(draft_markdown, encoding="utf-8")
    baseline_json_path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if cache:
        cache.save()

    # Local-only stages between the two API calls.
    if not args.skip_layout:
        rfdetr_page_out = paths.rfdetr
        doclayout_page_out = paths.doclayout
        _run_stage_script(
            "run_stage1.sh",
            [
                "--image",
                str(preprocessed_image),
                "--rfdetr-out",
                str(rfdetr_page_out),
                "--doclayout-out",
                str(doclayout_page_out),
                "--checkpoint",
                str(args.checkpoint),
                "--doclayout-device",
                str(args.doclayout_device),
                "--flat-output",
            ],
        )
        if page_dir.exists():
            shutil.rmtree(page_dir)
        _run_stage_script(
            "run_stage2.sh",
            [
                "--image-dir",
                str(paths.preprocessed),
                "--rfdetr-jsonl",
                str(rfdetr_page_out / "rfdetr_infer_results.jsonl"),
                "--doclayout-json-dir",
                str(doclayout_page_out / "json"),
                "--out-dir",
                str(paths.match),
                "--flat-output",
            ],
        )
    if not (page_dir / "match.json").exists():
        raise RuntimeError(f"match.json not found: {page_dir / 'match.json'}")

    contexts = _prepare_question_contexts(page_dir, preprocessed_img, baseline)

    # API call 2: one global review over the draft and every local context.
    review = _invoke_final_review(
        second_pass_vlm,
        preprocessed_img,
        draft_markdown,
        baseline,
        contexts,
        cache=cache,
        max_tokens=args.review_max_tokens,
        detail_views=args.answer_detail_views,
    )
    questions = _normalize_final_questions(review, contexts)
    if not questions:
        raise RuntimeError("second-pass global review produced no final questions")

    question_contexts_path = page_dir / "question_contexts.json"
    for question in questions:
        question_review = question.get("question_review") or {}
        question_review["context_manifest"] = str(question_contexts_path)
        question["question_review"] = question_review
        handwritten_answer = question.get("handwritten_answer") or {}
        handwritten_answer["context_manifest"] = str(question_contexts_path)
        question["handwritten_answer"] = handwritten_answer
    question_contexts_path.write_text(
        json.dumps(
            _build_question_context_manifest(
                page_name,
                preprocessed_image,
                draft_markdown_path,
                contexts,
                questions,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    handwriting_regions = [
        {
            "context_id": context.get("context_id"),
            "qno": (context.get("draft") or {}).get("qno"),
            "region": {
                "source_bbox_xyxy": (context.get("alignment") or {}).get("source_bbox_xyxy") or [],
                "frame_bbox_xyxy": (context.get("alignment") or {}).get("answer_bbox_xyxy") or [],
                "question_type": (context.get("alignment") or {}).get("question_type") or {},
                "strategy": (context.get("alignment") or {}).get("strategy") or "",
                "score": (context.get("alignment") or {}).get("detector_score") or 0.0,
            },
        }
        for context in contexts
        if (context.get("alignment") or {}).get("source_bbox_xyxy")
        and (context.get("alignment") or {}).get("answer_bbox_xyxy")
    ]
    handwriting_regions_path = page_dir / "handwriting_regions.json"
    handwriting_overlay_path = page_dir / "viz" / f"{page_name}_handwriting_overlay.png"
    handwriting_regions_path.write_text(
        json.dumps(
            {
                "page": page_name,
                "source_image": str(preprocessed_image),
                "legend": {"stem_box": "green", "handwriting_region": "magenta"},
                "questions": handwriting_regions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    draw_handwriting_overlay(preprocessed_img, handwriting_regions, handwriting_overlay_path)

    result_json_path = page_agent_out / "result.json"
    result_md_path = page_agent_out / "result.md"
    verification_path = page_agent_out / "verification.json"
    outputs = {
        "result_json": str(result_json_path),
        "result_markdown": str(result_md_path),
        "verification_report": str(verification_path),
        "question_contexts": str(question_contexts_path),
        "handwriting_regions": str(handwriting_regions_path),
        "handwriting_overlay": str(handwriting_overlay_path),
    }
    final = {
        "page": page_name,
        "source_image": str(original_image),
        "input_image": str(image),
        "preprocessed_image": str(preprocessed_image),
        "api_strategy": {
            "call_count": 2,
            "call_1": "whole-page draft Markdown",
            "call_2": "global context-aware correction and final result",
        },
        "api_metrics": _api_metrics(first_pass_vlm, second_pass_vlm),
        "summary": _result_summary(questions),
        "questions": questions,
        "page_notes": review.get("page_notes") or "",
        "artifacts": {
            "original_image": str(original_image),
            "preprocessing_report": str(preprocess_report_path),
            "first_pass_markdown": str(draft_markdown_path),
            "first_pass_response": str(baseline_json_path),
            "code_outputs": str(paths.code_outputs),
            "question_contexts": str(question_contexts_path),
            "handwriting_regions": str(handwriting_regions_path),
            "handwriting_overlay": str(handwriting_overlay_path),
            "second_pass_report": str(verification_path),
        },
        "outputs": outputs,
    }
    result_json_path.write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result_md_path.write_text(
        _render_result_markdown(page_name, questions),
        encoding="utf-8",
    )
    verification_path.write_text(
        json.dumps(
            {
                "mode": "single_global_second_pass",
                "first_pass_markdown": str(draft_markdown_path),
                "context_manifest": str(question_contexts_path),
                "page_notes": review.get("page_notes") or "",
                "api_metrics": final["api_metrics"],
                "raw": review.get("_raw") or review.get("raw") or "",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if cache:
        cache.save()
    return final


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("mathocr_workflow")
    parser.add_argument("--image", required=True, help="input exam page image")
    parser.add_argument(
        "--work-root",
        default="workflow",
        help="root containing workflow output groups",
    )
    parser.add_argument("--checkpoint", default="checkpoint_best_total.pth")
    parser.add_argument("--doclayout-device", default="cpu")
    parser.add_argument(
        "--skip-layout",
        action="store_true",
        help="reuse existing local detection and matching output",
    )
    parser.add_argument(
        "--api-base",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default=DEFAULT_VLM_MODEL)
    parser.add_argument("--baseline-max-tokens", type=int, default=5000)
    parser.add_argument("--review-max-tokens", type=int, default=7000)
    parser.add_argument("--review-timeout", type=int, default=240)
    parser.add_argument(
        "--answer-detail-views",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="magnified answer details per question added to the single second API request",
    )
    parser.add_argument("--cache", action="store_true", default=True)
    parser.add_argument("--no-cache", dest="cache", action="store_false")
    parser.add_argument("--cache-path", default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    final = run_agent(args)
    print(json.dumps(final.get("outputs", {}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

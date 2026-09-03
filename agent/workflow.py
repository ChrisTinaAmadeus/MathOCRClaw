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
    classify_question_type,
    draw_handwriting_overlay,
    save_handwriting_views,
    score_and_divide_question_frame,
)
from agent.recognition_profiles import (
    build_recognition_profile,
    canonical_choice_mode,
    profile_symbols,
    profile_tags,
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
from proofread.match_utils import find_figure_candidates, load_match_questions
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


FINAL_REVIEW_PROMPT = """You are API2, the conservative final visual verifier for a science-exam OCR workflow.
You receive one authoritative API1 draft split into question fields, a normalized full-page image, routing metadata, and labeled context images. API1 is the starting document. Do not rewrite or repeat unchanged text. Return only minimal, visually justified patches.

Return valid JSON only, without a Markdown fence, using this exact schema:
{
  "schema_version": "api2_patch_v4",
  "page_notes": "brief unresolved page-level ambiguity, otherwise empty",
  "question_reviews": [
    {
      "qno": "the draft qno",
      "source_context_ids": ["C001"],
      "question_type": "choice|fill|solution",
      "question_type_confidence": "high|medium|low",
      "stem": {
        "action": "keep|edit",
        "edits": [
          {
            "old": "one exact nonempty substring copied from draft.question_text",
            "new": "its complete replacement, with no ellipsis",
            "kind": "ocr_correction|insert_missing|remove_cancelled",
            "confidence": "high|medium|low",
            "context_id": "C001|FULL_PAGE",
            "evidence": "specific visible evidence"
          }
        ],
        "figure_observations": [
          {
            "audit_id": "required FIG-C001-01 target ID",
            "source_type": "printed_figure|student_diagram|mixed_figure|scratch_sketch|not_a_figure|uncertain",
            "content_role": "printed_question_figure|retained_solution_diagram|discarded_work|text_or_formula_placeholder|uncertain",
            "confidence": "high|medium|low",
            "context_id": "C001|FULL_PAGE",
            "printed_anchors": ["visible typeset labels, regular dashes, grid, uniform line work, or printed caption"],
            "handwriting_features": ["visible pen color, freehand line quality, or handwritten labels"],
            "visible_cancellation": false,
            "transcribed_content": "only for not_a_figure; otherwise empty",
            "observed_features": "specific visual evidence for source and role"
          }
        ]
      },
      "answer": {
        "action": "keep|replace_choice|replace_fill|edit_solution|set_no_answer",
        "confidence": "high|medium|low",
        "final_answers": [
          {"label": "overall or (1)", "value": "final answer only", "status": "ok|no_answer|uncertain"}
        ],
        "edits": [],
        "symbol_observations": [
          {
            "audit_id": "required symbol_audit target ID when resolving a listed target, otherwise empty",
            "tag": "one tag from recognition_profile.symbol_families",
            "location": "precise location of the handwritten mark",
            "candidates": ["two or more visually confusable candidates"],
            "selected": "candidate selected from visible strokes",
            "observed_features": "discriminating strokes actually visible, not semantic reasoning",
            "context_id": "C001",
            "reference_context_ids": ["optional same-writer clear glyph contexts"]
          }
        ],
        "evidence": "specific visible evidence"
      }
    }
  ]
}

Patch protocol:
- Emit exactly one question_reviews record for every draft question, in the same order. Never add, delete, merge, split, or renumber questions.
- Use keep when a field is already acceptable or the image is inconclusive. keep must have empty edits/final_answers. High precision is more important than making a change.
- For stem.action=edit and answer.action=edit_solution, every edit.old must be copied byte-for-byte from the corresponding draft field and occur there exactly once. Make the smallest local replacement possible. Never return an entire rewritten field.
- Never use `...`, `…`, `see full text`, summaries, placeholders, or references to omitted text. new must contain the complete replacement.
- Mark an edit high confidence only when the cited full-page/context image visibly proves it. Metadata and detector labels are routing hints, not evidence.
- Do not change punctuation, line breaks, LaTeX style, formula grouping, or wording merely for consistency. Preserve one handwritten logical step per line and preserve existing math-span boundaries.
- A remove_cancelled edit is allowed only for text visibly crossed out, erased, overwritten, or struck through. Do not mention or restore cancelled content elsewhere.

Figure ownership audit protocol:
- Every record in required_figure_audits is mandatory. Return exactly one stem.figure_observations record with the same audit_id for every existing draft `<插图>` marker. If the image is inconclusive, classify it as uncertain; never omit the audit or guess.
- Determine two independent properties: source_type says who produced the visible graphic, while content_role says whether it is retained exam/solution content. A student drawing is not automatically scratch work.
- printed_figure requires a visible printed anchor such as typeset labels, regular dashed lines, a grid/table/circuit framework, a printed caption, or uniform thin line work matching the question. Record the anchors actually visible.
- mixed_figure means a printed skeleton remains visible underneath handwriting or colored annotations. Handwritten overlays never turn a printed figure into scratch_sketch.
- student_diagram + retained_solution_diagram is an uncancelled construction, coordinate sketch, graph, or schematic used as part of the student's retained reasoning. Preserve it even though it is handwritten.
- scratch_sketch + discarded_work requires visible crossing-out, erasure, or overwriting and visible_cancellation=true. Mere isolation, freehand line quality, small size, or placement in the answer area is insufficient; use uncertain instead.
- not_a_figure is only for an API1 marker covering directly visible printed text or a formula. Put the complete visible transcription in transcribed_content. Do not use this class for a blurry or partly covered figure.
- Use uncertain whenever source or role cannot be proved. Color context is authoritative for ink ownership; grayscale answer_ink is only a stroke-visibility companion. Detector crops and routing metadata are never evidence by themselves.
- This audit is separate from text patching. In figure_audit_v1 every existing marker is locally preserved regardless of the model's label; do not submit a stem edit that removes or replaces `<插图>`. The audit creates evidence for later policy changes without risking the final document.

Question-conditioned symbol protocol:
- Each local context has a deterministic recognition_profile. Its tags contain candidate alphabets and contrastive stroke checks for the question type/topic. They are inspection checklists, never evidence and never an answer key.
- Before deciding keep, inspect every recognition_profile.symbol_audit.targets record. These are mandatory visual questions generated only from suspicious API1 surface notation; none identifies the correct candidate. Do not claim that the draft is accurate while leaving a clearly visible target unaudited.
- To resolve a symbol audit target, copy its unique draft_fragment byte-for-byte into edit.old, replace only suspect_surface using exactly one listed candidate_replacements value, and attach a symbol_observations record with the same audit_id. Describe stroke count, orientation, attachment, and any independent underline or arrowhead actually visible.
- same_writer_peer_audit_ids are an unlabeled personal glyph atlas from the same answer region. Compare slant, stroke count, spacing, and underline placement across those marks, but inspect every mark independently and never assume peers share a label because the proof would be correct.
- A geometry relation written by API1 as `//` is not canonical output. If the image clearly shows two parallel strokes, patch it to `\\parallel`; if a separate underline is also clearly visible, patch it to `\\mathrel{\\underline{\\parallel}}`. Never infer the underline from equality in the proof.
- For a vector-angle audit, distinguish a paired opening/closing delimiter from one-sided inequalities using the visible enclosure. Use the listed canonical replacement rather than raw `<...>` when the pair is visibly an angle delimiter.
- For a blurry mark, first locate the final retained handwriting, then compare only the relevant candidates, then inspect the listed discriminating strokes in the color context and the answer_ink companion when supplied.
- Never select a symbol because it makes the mathematics or science correct. The printed stem may narrow the symbol alphabet but cannot determine what the student wrote.
- replace_choice and replace_fill require at least one symbol_observations record. Every observation must cite a supplied context, use a tag present in that context's recognition_profile, list competing candidates, and state a concrete visible stroke feature. If this cannot be done, use keep.
- The payload groups contexts that share a symbol tag. For repeated closed-alphabet marks, you may compare an ambiguous glyph with independently clear same-page handwriting and cite those context IDs as reference_context_ids. Treat API1 readings as unverified: similarity of writer style is supporting evidence, but presumed mathematical correctness or repeated API1 labels are not.
- When API1 already has a supported answer that satisfies the printed answer grammar, a choice/fill change or an unlisted symbol change requires at least one independently clear same-page glyph in reference_context_ids. A solution symbol_audit target is the narrow exception: its own visible stroke structure is sufficient only when audit_id, exact draft_fragment, selected candidate, and candidate replacement all agree. Unsupported API1 answers and printed grammar/cardinality violations remain the other exceptions. Without one of these evidence paths, use keep.
- For answer.action=keep, symbol_observations should normally be empty. If mandatory audit targets remain genuinely unresolved, list their IDs briefly in page_notes instead of asserting that all symbols are accurate. Observations are an audit trail for changes, not hidden chain-of-thought.

Content contract:
- Determine the type from printed structure, not algebraic letters inside formulas and not detector labels. Printed A.--D. options mean choice; a printed answer blank means fill; proof/derivation/open response means solution.
- Choice: obey recognition_profile.answer_grammar exactly. A single-choice question must contain exactly one final A--D letter. A multiple-choice question may contain more than one. Never treat a multi-letter token such as `AC` as one glyph observation; cite each retained letter separately. Exclude all scratch work and diagram annotations.
- Fill: replace_fill contains only ordered final values/formulas and necessary units. Exclude derivations. Put mathematical values in standard LaTeX.
- Solution: edit_solution may only make exact local corrections to the complete API1 process. Never compress, paraphrase, reorder, or regenerate the solution. Use set_no_answer only when no student answer is visibly supported.
- Never replace a nonempty supported API1 answer with no_answer merely because a crop is incomplete. Use the full page to resolve ownership.
- P0 figure freeze: preserve every existing draft `<插图>` byte-for-byte. Do not remove or replace an existing marker in this review. Classify it only through the separate figure_audit_v1 representation above.
- When the only visible printed material is the question number/score and all geometry appears among the student's proof or coordinate setup, treat the printed stem as unavailable; do not use the student sketch to manufacture stem content.
- If a printed stem is not visible, keep `N. [无法识别]` or the existing minimal draft; do not invent a diagram or question text from the student's solution.
- Preserve option labels, subquestion labels, circled enumerators, signs, inequalities, exponents, subscripts, units, and balanced `$...$`/`$$...$$` delimiters.
- Do not solve, grade, or correct the student's mathematics. Output what is visibly retained.
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


def _draft_questions(baseline: Dict[str, Any]) -> List[Dict[str, Any]]:
    page_markdown = str(baseline.get("page_markdown") or "")
    if page_markdown.strip():
        parsed = _draft_questions_from_markdown(page_markdown)
        if parsed:
            return parsed

    questions: List[Dict[str, Any]] = []
    for question in baseline.get("questions") or []:
        if not isinstance(question, dict):
            continue
        question_text = str(question.get("question_text") or "").strip()
        questions.append(
            {
                "qno": str(question.get("qno") or "").strip(),
                "question_text": question_text,
                "student_answer": str(question.get("student_answer") or "").strip(),
                "answer_status": str(question.get("answer_status") or "uncertain").strip(),
                "question_type": _canonical_question_type(
                    "",
                    question_text,
                ),
                "choice_mode": canonical_choice_mode(question_text=question_text),
                "section_heading_before": "",
            }
        )
    return questions


_DRAFT_QUESTION_RE = re.compile(r"(?m)^\s*(\d{1,3})[.．、]\s*")
_DRAFT_ANSWER_RE = re.compile(r"(?im)^\s*#{1,6}\s*手写答案\s*$")
_SECTION_HEADING_RE = re.compile(
    r"^\s*(?:[一二三四五六七八九十]+|[IVX]+)\s*[、.．]\s*"
    r"(?:单选题|多选题|单项选择题|多项选择题|选择题|填空题|解答题|证明题|计算题|应用题).*?$",
    re.I,
)
_OPTION_MARKER_RE = re.compile(r"(?m)^\s*([A-D])[.．、:：]\s*")
_SUBPART_LABEL_RE = re.compile(r"(?:\(|（)(\d+)(?:\)|）)")
_CIRCLED_ENUM_RE = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩]")
_MATH_SPAN_RE = re.compile(r"\$\$(.+?)\$\$|(?<!\\)\$(.+?)(?<!\\)\$", re.S)
_NO_ANSWER_MARKER_RE = re.compile(
    r"(?:_+\s*)?(?:未识别到手写答案。?|\[无法识别\]|\[unreadable\])(?:\s*_+)?",
    re.I,
)
_EDITORIAL_TRUNCATION_RE = re.compile(
    r"(?:\.\s*\.\s*\.|…{1,}|see\s+(?:the\s+)?full\s+text|"
    r"full\s+text\s+(?:above|omitted)|下文省略|内容省略|同上|详见(?:上|下)文)",
    re.I,
)


def _canonical_question_type(
    value: Any,
    question_text: str = "",
    section_heading: str = "",
) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "choice": "choice",
        "multiple_choice": "choice",
        "multiple_choice_question": "choice",
        "fill": "fill",
        "fill_blank": "fill",
        "fill_blank_question": "fill",
        "solution": "solution",
        "short_answer": "solution",
        "problem_solving_question": "solution",
    }
    text_type = classify_question_type({}, question_text).get("type")
    if text_type == "choice":
        return "choice"
    if text_type == "fill_blank":
        return "fill"
    section = str(section_heading or "")
    if re.search(r"选择题|单选|多选", section):
        return "choice"
    if re.search(r"填空题", section):
        return "fill"
    if re.search(r"解答题|证明题|计算题|应用题", section):
        return "solution"
    if raw in aliases:
        return aliases[raw]
    if text_type == "short_answer":
        return "solution"
    return "solution"


def _split_section_headings(text: str) -> tuple[str, List[str]]:
    content: List[str] = []
    headings: List[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped and _SECTION_HEADING_RE.match(stripped):
            headings.append(stripped)
        else:
            content.append(line)
    return "\n".join(content).strip(), headings


def _draft_questions_from_markdown(markdown: str) -> List[Dict[str, Any]]:
    starts = list(_DRAFT_QUESTION_RE.finditer(markdown or ""))
    questions: List[Dict[str, Any]] = []
    _, prefix_headings = _split_section_headings(
        markdown[: starts[0].start()] if starts else markdown
    )
    pending_section = prefix_headings[-1] if prefix_headings else ""
    active_choice_mode = (
        canonical_choice_mode(section_heading=pending_section)
        if re.search(r"选择题|单选|多选", pending_section)
        else ""
    )
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(markdown)
        block = markdown[start.start() : end].strip()
        answer_heading = _DRAFT_ANSWER_RE.search(block)
        if answer_heading:
            question_text = block[: answer_heading.start()].strip()
            student_answer, trailing_sections = _split_section_headings(
                block[answer_heading.end() :]
            )
        else:
            question_text = block
            student_answer = ""
            trailing_sections = []
        no_answer = not student_answer or "未识别到手写答案" in student_answer
        question_type = _canonical_question_type(
            "",
            question_text,
            pending_section,
        )
        section_choice_mode = (
            canonical_choice_mode(section_heading=pending_section)
            if re.search(r"选择题|单选|多选", pending_section)
            else ""
        )
        if section_choice_mode:
            active_choice_mode = section_choice_mode
        choice_mode = (
            active_choice_mode or canonical_choice_mode(question_text=question_text)
            if question_type == "choice"
            else ""
        )
        questions.append(
            {
                "qno": str(int(start.group(1))),
                "question_text": question_text,
                "student_answer": student_answer,
                "answer_status": "no_answer" if no_answer else "ok",
                "question_type": question_type,
                "choice_mode": choice_mode,
                "section_heading_before": pending_section,
            }
        )
        pending_section = trailing_sections[-1] if trailing_sections else ""
        if pending_section and not re.search(r"选择题|单选|多选", pending_section):
            active_choice_mode = ""
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
            "question_type": "solution",
            "choice_mode": "",
            "section_heading_before": "",
        }
        context_id = f"C{ordinal:03d}"
        recognition_profile = build_recognition_profile(
            draft["question_text"],
            draft.get("question_type"),
            draft.get("student_answer") or "",
            section_heading=draft.get("section_heading_before") or "",
            choice_mode=draft.get("choice_mode") or "",
        )
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
            if "<插图>" in str(draft.get("question_text") or ""):
                for figure_index, figure_path in enumerate(
                    find_figure_candidates(page_dir, question)[:3],
                    start=1,
                ):
                    views.append(
                        {
                            "kind": f"figure_candidate_{figure_index:02d}",
                            "bbox_xyxy": [],
                            "path": str(figure_path),
                            "purpose": (
                                "color detector crop for figure ownership and role audit; "
                                "routing evidence only until visually inspected"
                            ),
                        }
                    )
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
                    "figure_candidate_count": sum(
                        1
                        for view in views
                        if str(view.get("kind") or "").startswith(
                            "figure_candidate_"
                        )
                    ),
                    "context_error": context_error,
                },
                "recognition_profile": recognition_profile,
                "visual_context": {"views": views},
            }
        )

    for ordinal in range(len(main_indices) + 1, len(draft_questions) + 1):
        draft = draft_questions[ordinal - 1]
        contexts.append(
            {
                "context_id": f"C{ordinal:03d}",
                "draft": draft,
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
                "recognition_profile": build_recognition_profile(
                    draft["question_text"],
                    draft.get("question_type"),
                    draft.get("student_answer") or "",
                    section_heading=draft.get("section_heading_before") or "",
                    choice_mode=draft.get("choice_mode") or "",
                ),
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
                "recognition_profile": context.get("recognition_profile") or {},
                "figure_audit": {
                    "version": "figure_audit_v1",
                    "targets": _figure_audit_targets(context),
                },
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


def _same_page_symbol_groups(contexts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[str]] = {}
    for context in contexts:
        context_id = str(context.get("context_id") or "")
        if not (context.get("visual_context") or {}).get("views"):
            continue
        profile = context.get("recognition_profile") or {}
        for tag in sorted(profile_tags(profile if isinstance(profile, dict) else {})):
            grouped.setdefault(tag, []).append(context_id)
    return [
        {
            "tag": tag,
            "context_ids": context_ids,
            "usage": "unverified same-writer comparison routes; inspect every cited glyph",
        }
        for tag, context_ids in sorted(grouped.items())
        if len(context_ids) >= 2
    ]


def _required_symbol_audits(
    contexts: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    audits: List[Dict[str, Any]] = []
    for context in contexts:
        context_id = str(context.get("context_id") or "")
        profile = context.get("recognition_profile") or {}
        symbol_audit = profile.get("symbol_audit") if isinstance(profile, dict) else {}
        for target in (
            symbol_audit.get("targets")
            if isinstance(symbol_audit, dict)
            and isinstance(symbol_audit.get("targets"), list)
            else []
        ):
            if not isinstance(target, dict):
                continue
            audits.append({"context_id": context_id, **target})
    grouped_ids: Dict[tuple[str, str], List[str]] = {}
    for audit in audits:
        key = (str(audit.get("context_id") or ""), str(audit.get("tag") or ""))
        grouped_ids.setdefault(key, []).append(str(audit.get("audit_id") or ""))
    for audit in audits:
        key = (str(audit.get("context_id") or ""), str(audit.get("tag") or ""))
        audit["same_writer_peer_audit_ids"] = [
            audit_id
            for audit_id in grouped_ids.get(key, [])
            if audit_id and audit_id != str(audit.get("audit_id") or "")
        ]
    return audits


_FIGURE_SOURCE_TYPES = {
    "printed_figure",
    "student_diagram",
    "mixed_figure",
    "scratch_sketch",
    "not_a_figure",
    "uncertain",
}
_FIGURE_CONTENT_ROLES = {
    "printed_question_figure",
    "retained_solution_diagram",
    "discarded_work",
    "text_or_formula_placeholder",
    "uncertain",
}


def _figure_audit_targets(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create stable audit targets for API1 figure markers without interpreting them."""
    context_id = str(context.get("context_id") or "")
    draft = context.get("draft") if isinstance(context.get("draft"), dict) else {}
    question_text = str(draft.get("question_text") or "")
    targets: List[Dict[str, Any]] = []
    for marker_index, match in enumerate(
        re.finditer(re.escape("<插图>"), question_text),
        start=1,
    ):
        excerpt_start = max(0, match.start() - 120)
        excerpt_end = min(len(question_text), match.end() + 80)
        targets.append(
            {
                "audit_id": f"FIG-{context_id}-{marker_index:02d}",
                "context_id": context_id,
                "marker_index": marker_index,
                "nearby_draft_text": question_text[excerpt_start:excerpt_end],
                "classification_options": {
                    "source_type": sorted(_FIGURE_SOURCE_TYPES),
                    "content_role": sorted(_FIGURE_CONTENT_ROLES),
                },
                "default_disposition": "preserve",
            }
        )
    return targets


def _required_figure_audits(
    contexts: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    audits: List[Dict[str, Any]] = []
    for context in contexts:
        audits.extend(_figure_audit_targets(context))
    return audits


def _normalized_figure_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _fallback_figure_observation(target: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "audit_id": str(target.get("audit_id") or ""),
        "marker_index": int(target.get("marker_index") or 0),
        "source_type": "uncertain",
        "content_role": "uncertain",
        "confidence": "low",
        "context_id": str(target.get("context_id") or ""),
        "printed_anchors": [],
        "handwriting_features": [],
        "visible_cancellation": False,
        "transcribed_content": "",
        "observed_features": (
            "No valid API2 figure observation was supplied; the local gate preserved "
            "the API1 marker."
        ),
        "local_action": "preserve",
        "model_observation_valid": False,
    }


def _normalize_figure_observations(
    value: Any,
    *,
    targets: Sequence[Dict[str, Any]],
    valid_context_ids: Sequence[str],
    available_context_ids: Sequence[str],
) -> tuple[List[Dict[str, Any]], List[str], int]:
    """Validate figure classifications while keeping marker disposition local."""
    target_by_id = {
        str(target.get("audit_id") or ""): target
        for target in targets
        if target.get("audit_id")
    }
    actions: List[str] = []
    if not target_by_id:
        if isinstance(value, list) and value:
            actions.append("rejected_unexpected_figure_observations")
        return [], actions, 0

    raw_observations = value if isinstance(value, list) else []
    cited = {str(context_id) for context_id in valid_context_ids if context_id}
    available = {str(context_id) for context_id in available_context_ids if context_id}
    accepted: Dict[str, Dict[str, Any]] = {}
    compatible_roles = {
        "printed_figure": {"printed_question_figure"},
        "student_diagram": {"retained_solution_diagram"},
        "mixed_figure": {
            "printed_question_figure",
            "retained_solution_diagram",
        },
        "scratch_sketch": {"discarded_work"},
        "not_a_figure": {"text_or_formula_placeholder"},
        "uncertain": {"uncertain"},
    }

    for raw in raw_observations:
        if not isinstance(raw, dict):
            actions.append("rejected_figure_observation_not_object")
            continue
        audit_id = str(raw.get("audit_id") or "").strip()
        target = target_by_id.get(audit_id)
        if target is None:
            actions.append("rejected_figure_observation_unknown_audit_id")
            continue
        if audit_id in accepted:
            actions.append(f"rejected_duplicate_figure_observation_{audit_id}")
            continue
        context_id = str(raw.get("context_id") or "").strip()
        target_context_id = str(target.get("context_id") or "")
        if (
            context_id not in {target_context_id, "FULL_PAGE"}
            or context_id not in (available | {"FULL_PAGE"})
            or (context_id != "FULL_PAGE" and context_id not in cited)
        ):
            actions.append(f"rejected_figure_observation_invalid_context_{audit_id}")
            continue
        source_type = str(raw.get("source_type") or "").strip().lower()
        content_role = str(raw.get("content_role") or "").strip().lower()
        confidence = str(raw.get("confidence") or "").strip().lower()
        observed_features = str(raw.get("observed_features") or "").strip()
        printed_anchors = _normalized_figure_string_list(raw.get("printed_anchors"))
        handwriting_features = _normalized_figure_string_list(
            raw.get("handwriting_features")
        )
        transcribed_content = str(raw.get("transcribed_content") or "").strip()
        visible_cancellation = raw.get("visible_cancellation") is True
        if source_type not in _FIGURE_SOURCE_TYPES:
            actions.append(f"rejected_figure_observation_invalid_source_{audit_id}")
            continue
        if content_role not in compatible_roles.get(source_type, set()):
            actions.append(f"rejected_figure_observation_incompatible_role_{audit_id}")
            continue
        if confidence not in {"high", "medium", "low"} or not observed_features:
            actions.append(f"rejected_figure_observation_weak_evidence_{audit_id}")
            continue
        if source_type != "uncertain" and confidence == "low":
            actions.append(f"rejected_figure_observation_low_confidence_{audit_id}")
            continue
        if source_type in {"printed_figure", "mixed_figure"} and not printed_anchors:
            actions.append(f"rejected_figure_observation_missing_print_anchor_{audit_id}")
            continue
        if source_type in {
            "student_diagram",
            "mixed_figure",
            "scratch_sketch",
        } and not handwriting_features:
            actions.append(
                f"rejected_figure_observation_missing_handwriting_features_{audit_id}"
            )
            continue
        if source_type == "printed_figure" and handwriting_features:
            actions.append(
                f"rejected_figure_observation_printed_with_handwriting_{audit_id}"
            )
            continue
        if source_type in {"student_diagram", "scratch_sketch"} and printed_anchors:
            actions.append(
                f"rejected_figure_observation_handwritten_with_print_anchors_{audit_id}"
            )
            continue
        if source_type == "scratch_sketch" and not visible_cancellation:
            actions.append(
                f"rejected_figure_observation_scratch_without_cancellation_{audit_id}"
            )
            continue
        if source_type == "student_diagram" and visible_cancellation:
            actions.append(
                f"rejected_figure_observation_retained_with_cancellation_{audit_id}"
            )
            continue
        if source_type == "not_a_figure" and (
            not transcribed_content or "<插图>" in transcribed_content
        ):
            actions.append(f"rejected_figure_observation_missing_transcription_{audit_id}")
            continue

        accepted[audit_id] = {
            "audit_id": audit_id,
            "marker_index": int(target.get("marker_index") or 0),
            "source_type": source_type,
            "content_role": content_role,
            "confidence": confidence,
            "context_id": context_id,
            "printed_anchors": printed_anchors,
            "handwriting_features": handwriting_features,
            "visible_cancellation": visible_cancellation,
            "transcribed_content": transcribed_content,
            "observed_features": observed_features,
            # figure_audit_v1 is intentionally observational. A model label can
            # never authorize deletion from the final document.
            "local_action": "preserve",
            "model_observation_valid": True,
        }
        actions.append(f"recorded_figure_observation_{audit_id}_{source_type}")

    normalized: List[Dict[str, Any]] = []
    for audit_id, target in target_by_id.items():
        observation = accepted.get(audit_id)
        if observation is None:
            observation = _fallback_figure_observation(target)
            actions.append(f"preserved_missing_figure_observation_{audit_id}")
        normalized.append(observation)
    return normalized, actions, len(accepted)


def _selected_review_views(
    context: Dict[str, Any],
    detail_views: int,
) -> List[Dict[str, Any]]:
    views = (context.get("visual_context") or {}).get("views") or []
    primary = [view for view in views if view.get("kind") == "context"][:1]
    stem_views = [view for view in views if view.get("kind") == "stem"][:1]
    figure_views = [
        view
        for view in views
        if str(view.get("kind") or "").startswith("figure_candidate_")
    ][:3]
    tiled_details = [
        view
        for view in views
        if str(view.get("kind") or "").startswith("answer_detail_")
    ]
    ink_details = [view for view in views if view.get("kind") == "answer_ink"][:1]
    question_type = str(
        ((context.get("recognition_profile") or {}).get("question_type") or "solution")
    )
    has_symbol_audits = bool(
        (((context.get("recognition_profile") or {}).get("symbol_audit") or {}).get("targets"))
    )
    has_figure_audits = bool(_figure_audit_targets(context))
    detail_candidates = (
        ink_details + tiled_details
        if question_type in {"choice", "fill"}
        else tiled_details + ink_details
    )
    details = detail_candidates[: max(0, detail_views)]
    if has_symbol_audits and detail_views > 0 and ink_details:
        details = details + [view for view in ink_details if view not in details]
    selected = primary + (stem_views + figure_views if has_figure_audits else []) + details
    selected = list({id(view): view for view in selected}.values())
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
        "draft_contract": "Each local_question_contexts[].draft field is the authoritative API1 text to patch.",
        "first_pass_page_notes": baseline.get("page_notes") or "",
        "required_symbol_audits": _required_symbol_audits(contexts),
        "required_figure_audits": _required_figure_audits(contexts),
        "same_page_symbol_groups": _same_page_symbol_groups(contexts),
        "local_question_contexts": _review_context_digest(contexts),
    }
    del draft_markdown
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
    cache_key = f"{vlm.cache_tag}::api2_patch_v4::{prompt_hash}::{'::'.join(image_hashes)}"
    if cache:
        hit = cache.get("api2_patch_review_v4", cache_key)
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
        cache.set("api2_patch_review_v4", cache_key, review)
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


def _high_confidence_patch(value: Any) -> bool:
    return str(value or "").strip().lower() == "high"


def _normalize_symbol_observations(
    value: Any,
    *,
    profile: Dict[str, Any],
    valid_context_ids: Sequence[str],
    available_context_ids: Sequence[str] = (),
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Validate API2's auditable, contrastive readings of ambiguous glyphs."""
    observations: List[Dict[str, Any]] = []
    actions: List[str] = []
    if value in (None, []):
        return observations, actions
    if not isinstance(value, list):
        return observations, ["rejected_symbol_observations_not_list"]

    allowed_tags = profile_tags(profile)
    raw_audit = profile.get("symbol_audit") if isinstance(profile, dict) else {}
    audit_targets = {
        str(target.get("audit_id") or ""): target
        for target in (
            raw_audit.get("targets")
            if isinstance(raw_audit, dict)
            and isinstance(raw_audit.get("targets"), list)
            else []
        )
        if isinstance(target, dict) and target.get("audit_id")
    }
    allowed_contexts = {str(item) for item in valid_context_ids if str(item)}
    available_contexts = {
        str(item) for item in available_context_ids if str(item)
    } or set(allowed_contexts)
    for index, raw in enumerate(value, start=1):
        tag = f"symbol_observation_{index}"
        if not isinstance(raw, dict):
            actions.append(f"rejected_{tag}_not_object")
            continue
        family = str(raw.get("tag") or "").strip()
        audit_id = str(raw.get("audit_id") or "").strip()
        location = str(raw.get("location") or "").strip()
        selected = str(raw.get("selected") or "").strip()
        features = str(raw.get("observed_features") or "").strip()
        context_id = str(raw.get("context_id") or "").strip()
        candidates = [
            str(candidate).strip()
            for candidate in raw.get("candidates") or []
            if str(candidate).strip()
        ] if isinstance(raw.get("candidates"), list) else []
        candidates = list(dict.fromkeys(candidates))
        raw_reference_ids = raw.get("reference_context_ids")
        reference_context_ids = [
            str(reference_id).strip()
            for reference_id in raw_reference_ids
            if str(reference_id).strip() in available_contexts
            and str(reference_id).strip() != context_id
        ] if isinstance(raw_reference_ids, list) else []
        reference_context_ids = list(dict.fromkeys(reference_context_ids))

        if family not in allowed_tags:
            actions.append(f"rejected_{tag}_unknown_profile_tag")
            continue
        audit_target = audit_targets.get(audit_id) if audit_id else None
        if audit_id and not audit_target:
            actions.append(f"rejected_{tag}_unknown_audit_id")
            continue
        if audit_target and str(audit_target.get("tag") or "") != family:
            actions.append(f"rejected_{tag}_audit_tag_mismatch")
            continue
        canonical_selected = _canonical_profile_symbol(selected)
        if audit_target:
            for option in audit_target.get("candidate_replacements") or []:
                if not isinstance(option, dict):
                    continue
                if canonical_selected == _canonical_profile_symbol(
                    option.get("replacement")
                ):
                    selected = str(option.get("symbol") or "").strip()
                    canonical_selected = _canonical_profile_symbol(selected)
                    break
        allowed_symbols = {
            _canonical_profile_symbol(symbol)
            for symbol in profile_symbols(profile, family)
        }
        canonical_candidates = {
            _canonical_profile_symbol(candidate) for candidate in candidates
        }
        if (
            not allowed_symbols
            or canonical_selected not in allowed_symbols
            or not canonical_candidates.issubset(allowed_symbols)
        ):
            actions.append(f"rejected_{tag}_outside_profile_symbols")
            continue
        if audit_target:
            audit_symbols = {
                _canonical_profile_symbol(symbol)
                for symbol in audit_target.get("candidate_symbols") or []
            }
            if (
                canonical_selected not in audit_symbols
                or not canonical_candidates.issubset(audit_symbols)
            ):
                actions.append(f"rejected_{tag}_outside_audit_candidates")
                continue
        if context_id not in allowed_contexts:
            actions.append(f"rejected_{tag}_invalid_context")
            continue
        if not location or not features:
            actions.append(f"rejected_{tag}_missing_visual_detail")
            continue
        if (
            not selected
            or len(candidates) < 2
            or canonical_selected not in canonical_candidates
        ):
            actions.append(f"rejected_{tag}_invalid_contrastive_candidates")
            continue
        observations.append(
            {
                "audit_id": audit_id,
                "tag": family,
                "location": location,
                "candidates": candidates,
                "selected": selected,
                "observed_features": features,
                "context_id": context_id,
                "reference_context_ids": reference_context_ids,
            }
        )
        actions.append(f"accepted_{tag}")
    return observations, actions


def _canonical_profile_symbol(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "")).strip("$`*_ ")
    replacements = (
        (r"\leqslant", r"\le"),
        (r"\leq", r"\le"),
        ("≤", r"\le"),
        (r"\geqslant", r"\ge"),
        (r"\geq", r"\ge"),
        ("≥", r"\ge"),
        (r"\neq", r"\ne"),
        ("≠", r"\ne"),
        (r"\emptyset", r"\varnothing"),
        ("∅", r"\varnothing"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _visual_symbol_in_answer(symbol: Any, answer: Any) -> bool:
    selected = _canonical_profile_symbol(symbol)
    return bool(selected) and selected in _canonical_profile_symbol(answer)


def _symbol_audit_targets(profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw_audit = profile.get("symbol_audit") if isinstance(profile, dict) else {}
    targets = (
        raw_audit.get("targets")
        if isinstance(raw_audit, dict) and isinstance(raw_audit.get("targets"), list)
        else []
    )
    return {
        str(target.get("audit_id") or ""): target
        for target in targets
        if isinstance(target, dict) and target.get("audit_id")
    }


def _materialize_symbol_audit_edits(
    observations: Sequence[Dict[str, Any]],
    profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Turn validated visual choices into deterministic exact notation patches."""
    targets = _symbol_audit_targets(profile)
    edits: List[Dict[str, Any]] = []
    seen_audits: set[str] = set()
    for observation in observations:
        audit_id = str(observation.get("audit_id") or "")
        if not audit_id or audit_id in seen_audits or audit_id not in targets:
            continue
        target = targets[audit_id]
        selected = _canonical_profile_symbol(observation.get("selected"))
        replacement = next(
            (
                str(option.get("replacement") or "")
                for option in target.get("candidate_replacements") or []
                if isinstance(option, dict)
                and _canonical_profile_symbol(option.get("symbol")) == selected
            ),
            "",
        )
        old = str(target.get("draft_fragment") or "")
        surface = str(target.get("suspect_surface") or "")
        if (
            not old
            or not surface
            or old.count(surface) != 1
            or not replacement
            or replacement == surface
        ):
            continue
        edits.append(
            {
                "old": old,
                "new": old.replace(surface, replacement, 1),
                "kind": "ocr_correction",
                "confidence": "high",
                "context_id": str(observation.get("context_id") or ""),
                "evidence": (
                    f"symbol_audit {audit_id}: "
                    f"{str(observation.get('observed_features') or '').strip()}"
                ),
            }
        )
        seen_audits.add(audit_id)
    return edits


def _solution_edits_touch_symbol_audits(
    edits: Any,
    profile: Dict[str, Any],
) -> bool:
    if not isinstance(edits, list):
        return False
    targets = list(_symbol_audit_targets(profile).values())
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        old = str(edit.get("old") or "")
        if any(
            old == str(target.get("draft_fragment") or "")
            or (
                str(target.get("suspect_surface") or "")
                and str(target.get("suspect_surface") or "") in old
            )
            for target in targets
        ):
            return True
    return False


def _solution_edits_covered_by_symbol_audits(
    edits: Any,
    observations: Sequence[Dict[str, Any]],
    profile: Dict[str, Any],
) -> bool:
    """Allow only candidate-listed notation patches without a cross-question glyph.

    This narrow exception never authorizes digit/value changes: every edit must use
    a system-generated unique draft anchor and must equal exactly one candidate
    replacement selected by a validated visual observation.
    """
    if not isinstance(edits, list) or not edits:
        return False
    targets = _symbol_audit_targets(profile)
    observations_by_audit: Dict[str, List[Dict[str, Any]]] = {}
    for observation in observations:
        audit_id = str(observation.get("audit_id") or "")
        if audit_id:
            observations_by_audit.setdefault(audit_id, []).append(observation)

    for edit in edits:
        if not isinstance(edit, dict):
            return False
        old = str(edit.get("old") or "")
        new = str(edit.get("new") or "")
        context_id = str(edit.get("context_id") or "")
        covered = False
        for audit_id, target in targets.items():
            if old != str(target.get("draft_fragment") or ""):
                continue
            surface = str(target.get("suspect_surface") or "")
            if not surface or old.count(surface) != 1:
                continue
            options = {
                _canonical_profile_symbol(option.get("symbol")): str(
                    option.get("replacement") or ""
                )
                for option in target.get("candidate_replacements") or []
                if isinstance(option, dict) and option.get("symbol")
            }
            for observation in observations_by_audit.get(audit_id, []):
                if str(observation.get("context_id") or "") != context_id:
                    continue
                selected = _canonical_profile_symbol(observation.get("selected"))
                replacement = options.get(selected, "")
                if not replacement or replacement == surface:
                    continue
                if new == old.replace(surface, replacement, 1):
                    covered = True
                    break
            if covered:
                break
        if not covered:
            return False
    return True


def _apply_exact_edits(
    original: str,
    edits: Any,
    *,
    valid_context_ids: Sequence[str],
    field_name: str,
) -> tuple[str, List[str]]:
    """Apply small, visually cited API2 patches without permitting free rewrites."""
    result = str(original or "")
    actions: List[str] = []
    if not isinstance(edits, list):
        return result, [f"rejected_{field_name}_edits_not_list"]

    allowed_contexts = {str(value) for value in valid_context_ids if str(value)}
    allowed_contexts.add("FULL_PAGE")
    original_length = len(result)
    for index, edit in enumerate(edits, start=1):
        tag = f"{field_name}_edit_{index}"
        if not isinstance(edit, dict):
            actions.append(f"rejected_{tag}_not_object")
            continue
        old = str(edit.get("old") or "")
        new = str(edit.get("new") or "")
        kind = str(edit.get("kind") or "ocr_correction").strip()
        context_id = str(edit.get("context_id") or "").strip()
        evidence = str(edit.get("evidence") or "").strip()
        if not _high_confidence_patch(edit.get("confidence")):
            actions.append(f"rejected_{tag}_non_high_confidence")
            continue
        if not evidence:
            actions.append(f"rejected_{tag}_missing_evidence")
            continue
        if context_id not in allowed_contexts:
            actions.append(f"rejected_{tag}_invalid_context")
            continue
        if not old or result.count(old) != 1:
            actions.append(f"rejected_{tag}_non_unique_old")
            continue
        if kind not in {"ocr_correction", "insert_missing", "remove_cancelled"}:
            actions.append(f"rejected_{tag}_invalid_kind")
            continue
        if old.strip() == result.strip() and original_length > 80:
            actions.append(f"rejected_{tag}_whole_field_rewrite")
            continue
        if len(old) > max(240, int(0.50 * max(1, original_length))):
            actions.append(f"rejected_{tag}_oversized_old")
            continue
        if _contains_editorial_truncation(new):
            actions.append(f"rejected_{tag}_editorial_truncation")
            continue
        old_start = result.index(old)
        containing_math_span = next(
            (
                match
                for match in _MATH_SPAN_RE.finditer(result)
                if match.start() <= old_start
                and old_start + len(old) <= match.end()
            ),
            None,
        )
        if (
            field_name == "solution"
            and "\n" in new
            and "\n" not in old
            and containing_math_span is not None
            and not containing_math_span.group(0).startswith("$$")
        ):
            actions.append(f"rejected_{tag}_newline_inside_inline_latex")
            continue
        candidate = result.replace(old, new, 1)
        if not _has_balanced_math_delimiters(candidate):
            actions.append(f"rejected_{tag}_unbalanced_latex")
            continue
        if field_name == "solution":
            before_structure = _question_structure(result)
            after_structure = _question_structure(candidate)
            if not before_structure["subparts"].issubset(after_structure["subparts"]):
                actions.append(f"rejected_{tag}_deleted_subpart")
                continue
            if (
                _math_span_count(candidate) < _math_span_count(result)
                and kind != "remove_cancelled"
            ):
                actions.append(f"rejected_{tag}_latex_span_loss")
                continue
        result = candidate
        actions.append(f"applied_{tag}_{kind}")
    return result, actions


def _math_span_count(text: Any) -> int:
    return len(_MATH_SPAN_RE.findall(str(text or "")))


def _has_balanced_math_delimiters(text: Any) -> bool:
    return len(re.findall(r"(?<!\\)\$", str(text or ""))) % 2 == 0


def _is_no_supported_answer_text(text: Any) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    value = re.sub(r"(?m)^\s*(?:\(\d+\)|（\d+）|overall)\s*", "", value)
    value = _NO_ANSWER_MARKER_RE.sub("", value)
    value = re.sub(r"[\s$*_`.,，。;；:：()（）\[\]-]+", "", value)
    return not value


def _contains_editorial_truncation(text: Any) -> bool:
    return bool(_EDITORIAL_TRUNCATION_RE.search(str(text or "")))


def _question_structure(text: Any) -> Dict[str, Any]:
    value = str(text or "")
    return {
        "options": set(_OPTION_MARKER_RE.findall(value)),
        "subparts": set(_SUBPART_LABEL_RE.findall(value)),
        "circled": set(_CIRCLED_ENUM_RE.findall(value)),
        "images": value.count("<插图>"),
    }


def _normalize_question_prefix(text: str, label: str) -> str:
    value, _ = _split_section_headings(text)
    value = value.strip()
    if re.match(r"^\s*\d{1,3}[.．、]\s*", value):
        return re.sub(
            r"^\s*\d{1,3}[.．、]\s*",
            f"{label}. ",
            value,
            count=1,
        ).strip()
    return f"{label}. {value}".strip()


def _guard_question_markdown(
    candidate: str,
    draft: str,
    label: str,
    *,
    allowed_image_removals: int = 0,
) -> tuple[str, List[str]]:
    actions: List[str] = []
    draft_value = _normalize_question_prefix(draft, label) if str(draft or "").strip() else ""
    candidate_value = _normalize_question_prefix(candidate, label) if str(candidate or "").strip() else ""

    if not candidate_value or re.fullmatch(rf"{re.escape(label)}\.\s*", candidate_value):
        actions.append("restored_missing_question_stem")
        candidate_value = draft_value or f"{label}. [无法识别]"

    if not _has_balanced_math_delimiters(candidate_value):
        actions.append("rolled_back_unbalanced_question_latex")
        candidate_value = draft_value or f"{label}. [无法识别]"

    if draft_value:
        draft_structure = _question_structure(draft_value)
        candidate_structure = _question_structure(candidate_value)
        missing_labels = any(
            not draft_structure[key].issubset(candidate_structure[key])
            for key in ("options", "subparts", "circled")
        )
        removed_images = max(
            0,
            int(draft_structure["images"]) - int(candidate_structure["images"]),
        )
        if missing_labels or removed_images > max(0, int(allowed_image_removals)):
            actions.append("rolled_back_deleted_question_structure")
            candidate_value = draft_value

        if "[无法识别]" in draft_value and re.fullmatch(
            rf"{re.escape(label)}\.\s*(?:\[无法识别\])?\s*",
            candidate_value,
        ):
            candidate_value = draft_value
            if "restored_unreadable_stem_sentinel" not in actions:
                actions.append("restored_unreadable_stem_sentinel")

    if re.fullmatch(rf"{re.escape(label)}\.\s*", candidate_value):
        candidate_value = f"{label}. [无法识别]"
        actions.append("inserted_unreadable_stem_sentinel")
    return candidate_value, actions


def _choice_letters(value: Any) -> str:
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    for line in [str(value or "").strip(), *lines]:
        compact = line.strip().strip("$`*_ ")
        compact = re.sub(r"^(?:答案|选择|选|故选)\s*(?:为|是)?\s*[:：]?\s*", "", compact)
        compact = compact.strip("()（）[]【】,，、;；:：.。 ")
        if re.fullmatch(r"[A-Da-d](?:[\s,，、;；]*[A-Da-d])*", compact):
            letters = re.sub(r"[^A-D]", "", compact.upper())
            return "".join(dict.fromkeys(letters))
    return ""


def _choice_answer_valid_for_profile(value: Any, profile: Dict[str, Any]) -> bool:
    letters = _choice_letters(value)
    if not letters:
        return False
    grammar = profile.get("answer_grammar") or {}
    allowed = {
        str(token).strip().upper()
        for token in grammar.get("allowed_final_tokens") or []
        if re.fullmatch(r"[A-Da-d]", str(token).strip())
    } or {"A", "B", "C", "D"}
    if not set(letters).issubset(allowed):
        return False
    minimum = max(1, int(grammar.get("min_selected") or 1))
    maximum = max(minimum, int(grammar.get("max_selected") or len(allowed)))
    return minimum <= len(letters) <= maximum


def _choice_observations_cover_answer(
    value: Any,
    observations: Sequence[Dict[str, Any]],
) -> bool:
    letters = _choice_letters(value)
    observed = {
        _canonical_profile_symbol(observation.get("selected")).upper()
        for observation in observations
        if str(observation.get("tag") or "") == "CHOICE_LETTER"
    }
    return bool(letters) and set(letters).issubset(observed)


def _has_independent_symbol_reference(
    observations: Sequence[Dict[str, Any]],
) -> bool:
    return any(observation.get("reference_context_ids") for observation in observations)


def _ensure_math_answer(value: Any) -> str:
    text = str(value or "").strip()
    if not text or _is_no_supported_answer_text(text):
        return ""
    if _MATH_SPAN_RE.fullmatch(text) and _has_balanced_math_delimiters(text):
        return text
    text = text.strip("$").strip()
    return f"${text}$" if text else ""


def _last_nonempty_line(value: Any) -> str:
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _part_final_answer(parts: Sequence[Dict[str, Any]]) -> str:
    for part in reversed(list(parts)):
        value = str(part.get("final_answer") or "").strip()
        if value:
            return value
    return ""


def _render_structured_parts(parts: Sequence[Dict[str, Any]]) -> str:
    rendered: List[str] = []
    for part in parts:
        label = str(part.get("label") or "overall").strip()
        status = str(part.get("status") or "uncertain").strip()
        content = str(part.get("transcription") or part.get("final_answer") or "").strip()
        if status == "no_answer":
            content = "_未识别到手写答案。_"
        elif status == "unreadable" and not content:
            content = "[无法识别]"
        if not content:
            continue
        if label != "overall" and not re.match(rf"^\s*{re.escape(label)}(?:\s|$)", content):
            rendered.append(f"{label} {content}")
        else:
            rendered.append(content)
    return "\n".join(rendered).strip()


def _canonicalize_answer(
    question_type: str,
    answer: Dict[str, Any],
    parts: List[Dict[str, Any]],
    draft_answer: str,
) -> tuple[str, List[Dict[str, Any]], str, List[str]]:
    actions: List[str] = []
    raw_text = str(answer.get("text") or "").strip()
    raw_status = str(answer.get("status") or "").strip()

    if question_type == "choice":
        selected = _choice_letters(_part_final_answer(parts))
        if not selected:
            selected = _choice_letters(answer.get("final_answer"))
        if not selected:
            selected = _choice_letters(raw_text)
        if not selected:
            selected = _choice_letters(draft_answer)
            if selected:
                actions.append("restored_choice_from_first_pass")
        if selected:
            canonical_parts = [
                {
                    "label": "overall",
                    "transcription": selected,
                    "final_answer": selected,
                    "status": "ok",
                }
            ]
            if raw_text.strip() != selected:
                actions.append("removed_choice_scratch_work")
            return selected, canonical_parts, "ok", actions
        status = raw_status if raw_status in {"uncertain", "unreadable"} else "no_answer"
        return "", parts, status, actions

    if question_type == "fill":
        canonical_parts: List[Dict[str, Any]] = []
        for part in parts:
            label = str(part.get("label") or "overall").strip()
            part_status = str(part.get("status") or "uncertain").strip()
            final_value = str(part.get("final_answer") or "").strip()
            if not final_value:
                final_value = _last_nonempty_line(part.get("transcription"))
            if part_status == "no_answer" or _is_no_supported_answer_text(final_value):
                canonical_parts.append(
                    {
                        "label": label,
                        "transcription": "",
                        "final_answer": "",
                        "status": "no_answer",
                    }
                )
                continue
            if final_value:
                final_value = _ensure_math_answer(final_value)
                canonical_parts.append(
                    {
                        "label": label,
                        "transcription": final_value,
                        "final_answer": final_value,
                        "status": part_status if part_status in {"partial", "uncertain"} else "ok",
                    }
                )
        if canonical_parts:
            rendered = _render_structured_parts(canonical_parts)
            populated = any(part.get("final_answer") for part in canonical_parts)
            if populated:
                if raw_text.strip() != rendered:
                    actions.append("removed_fill_scratch_work")
                status = (
                    "partial"
                    if any(part.get("status") == "no_answer" for part in canonical_parts)
                    else (raw_status or "ok")
                )
                return rendered, canonical_parts, status, actions

        final_value = str(answer.get("final_answer") or "").strip()
        if not final_value:
            final_value = _last_nonempty_line(raw_text)
        if not final_value:
            final_value = _last_nonempty_line(draft_answer)
            if final_value and not _is_no_supported_answer_text(final_value):
                actions.append("restored_fill_from_first_pass")
        final_value = _ensure_math_answer(final_value)
        if final_value:
            canonical_parts = [
                {
                    "label": "overall",
                    "transcription": final_value,
                    "final_answer": final_value,
                    "status": "ok",
                }
            ]
            if raw_text.strip() != final_value:
                actions.append("removed_fill_scratch_work")
            return final_value, canonical_parts, "ok", actions
        status = raw_status if raw_status in {"uncertain", "unreadable"} else "no_answer"
        return "", parts, status, actions

    all_parts_no_answer = bool(parts) and all(
        str(part.get("status") or "").strip() == "no_answer" for part in parts
    )
    if all_parts_no_answer and draft_answer.strip():
        text = draft_answer.strip()
        actions.append("preserved_first_pass_no_answer_shape")
    else:
        text = _render_structured_parts(parts) if parts else raw_text
    if not text:
        text = draft_answer.strip()
        if text:
            actions.append("restored_solution_from_first_pass")

    if text and draft_answer.strip() and _contains_editorial_truncation(text):
        text = draft_answer.strip()
        parts = []
        actions.append("rolled_back_editorial_solution_truncation")

    if text and draft_answer.strip() and _math_span_count(text) < _math_span_count(draft_answer):
        text = draft_answer.strip()
        parts = []
        actions.append("rolled_back_solution_latex_span_loss")

    compact_text = re.sub(r"\s+", "", text)
    compact_draft = re.sub(r"\s+", "", draft_answer)
    if (
        compact_text
        and compact_draft
        and len(compact_text) < int(0.72 * len(compact_draft))
    ):
        text = draft_answer.strip()
        parts = []
        actions.append("rolled_back_truncated_solution")

    if not _has_balanced_math_delimiters(text):
        text = draft_answer.strip()
        parts = []
        actions.append("rolled_back_unbalanced_solution_latex")

    status = raw_status or ("ok" if text else "no_answer")
    if all_parts_no_answer:
        status = "no_answer"
    return text, parts, status, actions


def _normalize_patch_questions(
    review: Dict[str, Any],
    contexts: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_reviews = [
        item for item in review.get("question_reviews") or [] if isinstance(item, dict)
    ]
    unused_reviews = set(range(len(raw_reviews)))
    all_context_ids = {
        str(context.get("context_id") or "") for context in contexts if context.get("context_id")
    }
    visual_context_ids = {
        str(context.get("context_id") or "")
        for context in contexts
        if context.get("context_id")
        and (context.get("visual_context") or {}).get("views")
    }
    strict_symbol_protocol = str(review.get("schema_version") or "").strip() in {
        "api2_patch_v3",
        "api2_patch_v4",
    }
    questions: List[Dict[str, Any]] = []

    for index, context in enumerate(contexts, start=1):
        draft = context.get("draft") if isinstance(context.get("draft"), dict) else {}
        draft_key = _question_key(draft.get("qno"))
        review_index = next(
            (
                candidate_index
                for candidate_index in sorted(unused_reviews)
                if draft_key
                and _question_key(raw_reviews[candidate_index].get("qno")) == draft_key
            ),
            None,
        )
        patch_review = raw_reviews[review_index] if review_index is not None else {}
        if review_index is not None:
            unused_reviews.remove(review_index)

        source_ids = patch_review.get("source_context_ids")
        cited_context_ids = (
            [
                str(context_id).strip()
                for context_id in source_ids
                if str(context_id).strip() in all_context_ids
            ]
            if isinstance(source_ids, list)
            else []
        )
        normalized_source_ids = list(cited_context_ids)
        fallback_context_id = str(context.get("context_id") or "")
        if not normalized_source_ids and fallback_context_id:
            normalized_source_ids = [fallback_context_id]

        actions: List[str] = []
        if not patch_review:
            actions.append("kept_first_pass_missing_api2_review")

        qno = _normalize_qno(draft.get("qno"), index)
        label = str(qno)
        draft_question = str(draft.get("question_text") or "")
        question_candidate = draft_question
        stem_patch = patch_review.get("stem")
        stem_patch = stem_patch if isinstance(stem_patch, dict) else {}
        figure_targets = _figure_audit_targets(context)
        figure_observations, figure_actions, figure_observed_count = (
            _normalize_figure_observations(
                stem_patch.get("figure_observations"),
                targets=figure_targets,
                valid_context_ids=cited_context_ids,
                available_context_ids=visual_context_ids,
            )
        )
        actions.extend(figure_actions)
        stem_action = str(stem_patch.get("action") or "keep").strip().lower()
        if stem_action == "edit":
            question_candidate, edit_actions = _apply_exact_edits(
                draft_question,
                stem_patch.get("edits"),
                valid_context_ids=cited_context_ids,
                field_name="stem",
            )
            actions.extend(edit_actions)
        elif stem_action != "keep":
            actions.append("rejected_invalid_stem_action")

        if (
            _question_structure(question_candidate)["images"]
            < _question_structure(draft_question)["images"]
        ):
            actions.append("rejected_stem_image_removal_p0")
        question_markdown, question_actions = _guard_question_markdown(
            question_candidate,
            draft_question,
            label,
            allowed_image_removals=0,
        )
        actions.extend(question_actions)
        question_type = _canonical_question_type(
            draft.get("question_type") or patch_review.get("question_type"),
            question_markdown,
            str(draft.get("section_heading_before") or ""),
        )

        draft_answer = str(draft.get("student_answer") or "").strip()
        profile = (
            context.get("recognition_profile")
            if isinstance(context.get("recognition_profile"), dict)
            else {}
        )
        choice_mode = str(
            profile.get("choice_mode")
            or (profile.get("answer_grammar") or {}).get("choice_mode")
            or draft.get("choice_mode")
            or (canonical_choice_mode(question_text=question_markdown) if question_type == "choice" else "")
        )
        answer_patch = patch_review.get("answer")
        answer_patch = answer_patch if isinstance(answer_patch, dict) else {}
        raw_symbol_observations = answer_patch.get("symbol_observations")
        symbol_protocol_requested = raw_symbol_observations not in (None, [])
        answer_action = str(answer_patch.get("action") or "keep").strip().lower()
        patch_evidence = str(answer_patch.get("evidence") or "").strip()
        patch_is_high = _high_confidence_patch(answer_patch.get("confidence"))
        symbol_observations, symbol_actions = _normalize_symbol_observations(
            answer_patch.get("symbol_observations"),
            profile=profile,
            valid_context_ids=cited_context_ids,
            available_context_ids=visual_context_ids,
        )
        actions.extend(symbol_actions)
        raw_solution_edits = (
            answer_patch.get("edits")
            if isinstance(answer_patch.get("edits"), list)
            else []
        )
        materialized_audit_edits = _materialize_symbol_audit_edits(
            symbol_observations,
            profile,
        )
        non_audit_solution_edits = [
            edit
            for edit in raw_solution_edits
            if not _solution_edits_touch_symbol_audits([edit], profile)
        ]
        discarded_model_audit_edits = len(raw_solution_edits) - len(
            non_audit_solution_edits
        )
        effective_solution_edits = (
            materialized_audit_edits + non_audit_solution_edits
        )
        if discarded_model_audit_edits:
            actions.append(
                f"discarded_{discarded_model_audit_edits}_model_authored_symbol_edits"
            )
        if materialized_audit_edits:
            actions.append(
                f"materialized_{len(materialized_audit_edits)}_symbol_audit_edits"
            )
        solution_edits_touch_audits = _solution_edits_touch_symbol_audits(
            effective_solution_edits,
            profile,
        )
        solution_edits_audit_covered = _solution_edits_covered_by_symbol_audits(
            effective_solution_edits,
            symbol_observations,
            profile,
        )
        answer: Dict[str, Any] = {
            "text": draft_answer,
            "status": str(draft.get("answer_status") or "").strip(),
        }
        parts: List[Dict[str, Any]] = []
        answer_patch_applied = False
        draft_requires_repair = str(
            (profile.get("recognition_risk") or {}).get("priority") or ""
        ) == "high" or _is_no_supported_answer_text(draft_answer)

        if answer_action in {"replace_choice", "replace_fill"}:
            expected_action = "replace_choice" if question_type == "choice" else (
                "replace_fill" if question_type == "fill" else ""
            )
            if answer_action != expected_action:
                actions.append("rejected_answer_action_type_mismatch")
            elif not cited_context_ids:
                actions.append("rejected_answer_replacement_without_valid_context")
            elif not patch_is_high or not patch_evidence:
                actions.append("rejected_answer_replacement_without_high_visual_evidence")
            elif strict_symbol_protocol and not symbol_observations:
                actions.append("rejected_answer_replacement_without_symbol_observation")
            else:
                final_answers = answer_patch.get("final_answers")
                raw_parts = final_answers if isinstance(final_answers, list) else []
                for raw_part in raw_parts:
                    if not isinstance(raw_part, dict):
                        continue
                    value = str(raw_part.get("value") or "").strip()
                    part_status = str(raw_part.get("status") or "ok").strip()
                    if part_status == "no_answer":
                        value = ""
                    parts.append(
                        {
                            "label": str(raw_part.get("label") or "overall").strip(),
                            "transcription": value,
                            "final_answer": value,
                            "status": part_status,
                        }
                    )
                if not parts:
                    actions.append("rejected_empty_final_answer_replacement")
                elif question_type == "choice" and (
                    len(parts) != 1
                    or str(parts[0].get("label") or "overall") != "overall"
                    or not _choice_answer_valid_for_profile(
                        parts[0].get("final_answer"),
                        profile,
                    )
                ):
                    parts = []
                    actions.append("rejected_choice_answer_cardinality_or_grammar")
                elif (
                    strict_symbol_protocol
                    and question_type == "choice"
                    and not _choice_observations_cover_answer(
                        parts[0].get("final_answer"),
                        symbol_observations,
                    )
                ):
                    parts = []
                    actions.append("rejected_choice_answer_without_per_glyph_observations")
                elif strict_symbol_protocol and question_type == "fill" and not any(
                    _visual_symbol_in_answer(
                        observation.get("selected"),
                        "".join(str(part.get("final_answer") or "") for part in parts),
                    )
                    for observation in symbol_observations
                ):
                    parts = []
                    actions.append("rejected_symbol_observation_answer_mismatch")
                elif (
                    strict_symbol_protocol
                    and not draft_requires_repair
                    and not _has_independent_symbol_reference(symbol_observations)
                ):
                    parts = []
                    actions.append(
                        "rejected_answer_replacement_without_independent_symbol_reference"
                    )
                elif (
                    all(part.get("status") == "no_answer" for part in parts)
                    and not _is_no_supported_answer_text(draft_answer)
                ):
                    parts = []
                    actions.append("rejected_answer_deletion_over_supported_first_pass")
                else:
                    answer = {
                        "text": "",
                        "status": "no_answer"
                        if all(part.get("status") == "no_answer" for part in parts)
                        else "ok",
                    }
                    answer_patch_applied = True
                    actions.append(f"accepted_{answer_action}")
        elif answer_action == "edit_solution":
            if question_type != "solution":
                actions.append("rejected_solution_edit_type_mismatch")
            elif solution_edits_touch_audits and not solution_edits_audit_covered:
                actions.append(
                    "rejected_solution_symbol_edit_without_exact_audit_coverage"
                )
            elif (
                strict_symbol_protocol
                and symbol_protocol_requested
                and (
                    not symbol_observations
                    or (
                        not solution_edits_audit_covered
                        and
                        not draft_requires_repair
                        and not _has_independent_symbol_reference(symbol_observations)
                    )
                )
            ):
                actions.append(
                    "rejected_symbol_guided_solution_edit_without_independent_reference"
                )
            else:
                patched_answer, edit_actions = _apply_exact_edits(
                    draft_answer,
                    effective_solution_edits,
                    valid_context_ids=cited_context_ids,
                    field_name="solution",
                )
                answer["text"] = patched_answer
                actions.extend(edit_actions)
                answer_patch_applied = any(
                    action.startswith("applied_solution_edit_") for action in edit_actions
                )
        elif answer_action == "set_no_answer":
            if not cited_context_ids:
                actions.append("rejected_no_answer_without_valid_context")
            elif not patch_is_high or not patch_evidence:
                actions.append("rejected_no_answer_without_high_visual_evidence")
            elif not _is_no_supported_answer_text(draft_answer):
                actions.append("rejected_no_answer_over_supported_first_pass")
            else:
                actions.append("accepted_no_answer_confirmation")
        elif answer_action != "keep":
            actions.append("rejected_invalid_answer_action")

        audit_target_count = len(_symbol_audit_targets(profile))
        resolved_audit_ids = {
            str(observation.get("audit_id") or "")
            for observation in symbol_observations
            if observation.get("audit_id")
        }
        if answer_action == "keep" and audit_target_count:
            actions.append(
                f"kept_with_{audit_target_count}_unresolved_symbol_audit_targets"
            )

        if answer_patch_applied:
            text, parts, status, answer_actions = _canonicalize_answer(
                question_type,
                answer,
                parts,
                draft_answer,
            )
            actions.extend(answer_actions)
        else:
            text = draft_answer
            parts = []
            status = str(draft.get("answer_status") or "").strip() or (
                "ok" if draft_answer else "no_answer"
            )
            actions.append("kept_first_pass_answer_byte_exact")
        draft_section = str(draft.get("section_heading_before") or "").strip()
        questions.append(
            {
                "qno": qno,
                "question_type": question_type,
                "choice_mode": choice_mode,
                "section_heading_before": draft_section,
                "question_markdown": question_markdown,
                "question_review": {
                    "status": "reviewed_via_api2_atomic_patch",
                    "source_context_ids": normalized_source_ids,
                    "recognition_profile_version": str(
                        profile.get("version") or ""
                    ),
                    "symbol_audit_target_count": audit_target_count,
                    "symbol_audit_observed_count": len(resolved_audit_ids),
                    "figure_audit_version": "figure_audit_v1",
                    "figure_audit_target_count": len(figure_targets),
                    "figure_audit_observed_count": figure_observed_count,
                    "figure_observations": figure_observations,
                    "lint_actions": actions,
                },
                "handwritten_answer": {
                    "text": text,
                    "answer_parts": parts,
                    "uncertain_fragments": [],
                    "status": status,
                    "evidence_note": patch_evidence,
                    "symbol_observations": symbol_observations,
                    "source_context_ids": normalized_source_ids,
                },
            }
        )

    if unused_reviews and questions:
        questions[0]["question_review"]["lint_actions"].append(
            f"ignored_{len(unused_reviews)}_extra_api2_review_records"
        )
    return questions


def _normalize_final_questions(
    review: Dict[str, Any],
    contexts: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if isinstance(review.get("question_reviews"), list):
        return _normalize_patch_questions(review, contexts)

    raw_value = review.get("questions")
    if not isinstance(raw_value, list):
        return []

    raw_questions = [question for question in raw_value if isinstance(question, dict)]
    unused_raw = set(range(len(raw_questions)))
    ordered_records: List[tuple[Dict[str, Any], Dict[str, Any], bool]] = []
    for context_index, context in enumerate(contexts):
        draft = context.get("draft") if isinstance(context.get("draft"), dict) else {}
        draft_key = _question_key(draft.get("qno"))
        raw_index = next(
            (
                candidate_index
                for candidate_index in sorted(unused_raw)
                if draft_key
                and _question_key(raw_questions[candidate_index].get("qno")) == draft_key
            ),
            None,
        )
        if raw_index is None and context_index in unused_raw:
            ordinal_question = raw_questions[context_index]
            if not _question_key(ordinal_question.get("qno")):
                raw_index = context_index
        if raw_index is not None:
            unused_raw.remove(raw_index)
            ordered_records.append((raw_questions[raw_index], context, False))
            continue
        ordered_records.append(
            (
                {
                    "qno": draft.get("qno"),
                    "question_type": draft.get("question_type"),
                    "section_heading_before": draft.get("section_heading_before"),
                    "question_markdown": draft.get("question_text"),
                    "handwritten_answer": {
                        "text": draft.get("student_answer"),
                        "status": draft.get("answer_status"),
                    },
                },
                context,
                True,
            )
        )

    for raw_index in sorted(unused_raw):
        ordered_records.append((raw_questions[raw_index], {}, False))

    context_ids = [str(context.get("context_id") or "") for context in contexts]
    questions: List[Dict[str, Any]] = []
    for index, (raw_question, fallback_context, restored_question) in enumerate(
        ordered_records,
        start=1,
    ):
        draft = (
            fallback_context.get("draft")
            if isinstance(fallback_context.get("draft"), dict)
            else {}
        )
        raw_answer = raw_question.get("handwritten_answer")
        answer = raw_answer if isinstance(raw_answer, dict) else {}
        if not answer and raw_question.get("student_answer") is not None:
            answer = {"text": raw_question.get("student_answer")}
        parts = _normalize_answer_parts(answer.get("answer_parts"))
        uncertain = _normalize_uncertain_fragments(answer.get("uncertain_fragments"))

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

        qno = _normalize_qno(
            raw_question.get("qno") or draft.get("qno"),
            index,
        )
        label = str(qno)
        question_markdown, question_actions = _guard_question_markdown(
            str(raw_question.get("question_markdown") or ""),
            str(draft.get("question_text") or ""),
            label,
        )
        if restored_question:
            question_actions.insert(0, "restored_question_from_first_pass")
        detector_type = (
            (fallback_context.get("alignment") or {}).get("question_type") or {}
        ).get("type")
        question_type = _canonical_question_type(
            draft.get("question_type") or raw_question.get("question_type") or detector_type,
            question_markdown,
            str(draft.get("section_heading_before") or ""),
        )
        text, parts, status, answer_actions = _canonicalize_answer(
            question_type,
            answer,
            parts,
            str(draft.get("student_answer") or ""),
        )
        draft_section = str(draft.get("section_heading_before") or "").strip()
        review_section = str(raw_question.get("section_heading_before") or "").strip()
        section_heading = draft_section or (
            review_section if _SECTION_HEADING_RE.match(review_section) else ""
        )
        questions.append(
            {
                "qno": qno,
                "question_type": question_type,
                "section_heading_before": section_heading,
                "question_markdown": question_markdown,
                "question_review": {
                    "status": "reviewed_in_global_second_pass",
                    "source_context_ids": normalized_source_ids,
                    "lint_actions": question_actions + answer_actions,
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
        "schema_version": 5,
        "page": page_name,
        "source_image": str(source_image),
        "first_pass_markdown": str(draft_markdown_path),
        "api_strategy": {
            "call_1": "whole-page draft OCR",
            "between_calls": "local detection, layout, context construction",
            "call_2": (
                "question-conditioned symbol and figure-ownership review proposing "
                "atomic evidence-linked patches"
            ),
            "local_merge": "deterministic validation against the API1 draft",
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
    second_pass_vlms: Sequence[VLMClient],
) -> Dict[str, Any]:
    call_2_runs = [list(vlm.request_log) for vlm in second_pass_vlms]
    passes = {
        "call_1": list(first_pass_vlm.request_log),
        "call_2": [request for run in call_2_runs for request in run],
        "call_2_runs": call_2_runs,
    }
    requests = passes["call_1"] + passes["call_2"]

    def _token_sum(name: str) -> Optional[int]:
        values = [request.get(name) for request in requests if request.get(name) is not None]
        return sum(int(value) for value in values) if values else None

    return {
        "logical_call_count": 1 + len(second_pass_vlms),
        "api2_run_count": len(second_pass_vlms),
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
        if re.fullmatch(rf"{re.escape(label)}[.．、]\s*", question_markdown):
            question_markdown = f"{label}. [无法识别]"
        if lines:
            lines.append("")
        section_heading = str(question.get("section_heading_before") or "").strip()
        if section_heading and _SECTION_HEADING_RE.match(section_heading):
            lines.extend([section_heading, ""])
        lines.extend([question_markdown, "", "### 手写答案", ""])
        answer_text = str(answer.get("text") or "").strip()
        question_type = _canonical_question_type(
            question.get("question_type"),
            question_markdown,
            section_heading,
        )
        if question_type == "solution" and not answer_text:
            answer_text = _render_structured_parts(answer.get("answer_parts") or [])
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
        timeout_s=args.baseline_timeout,
        max_retries=1,
        temperature=0.0,
        top_p=0.7,
    )
    review_runs = int(getattr(args, "review_runs", 1))
    if review_runs < 1:
        raise ValueError("review_runs must be at least 1")
    second_pass_vlms = [
        VLMClient(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            timeout_s=args.review_timeout,
            max_retries=1,
            temperature=0.0,
            top_p=0.7,
        )
        for _ in range(review_runs)
    ]

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

    # API call 2: independent global reviews over the same API1 draft and local context.
    # Repeated reviews intentionally bypass the API2 cache so they remain three real samples.
    reviews: List[Dict[str, Any]] = []
    question_sets: List[List[Dict[str, Any]]] = []
    review_cache = cache if review_runs == 1 else None
    for run_index, second_pass_vlm in enumerate(second_pass_vlms, start=1):
        review = _invoke_final_review(
            second_pass_vlm,
            preprocessed_img,
            draft_markdown,
            baseline,
            contexts,
            cache=review_cache,
            max_tokens=args.review_max_tokens,
            detail_views=args.answer_detail_views,
        )
        questions = _normalize_final_questions(review, contexts)
        if not questions:
            raise RuntimeError(
                f"second-pass global review run {run_index} produced no final questions"
            )
        reviews.append(review)
        question_sets.append(questions)

    review = reviews[0]
    questions = question_sets[0]

    question_contexts_path = page_dir / "question_contexts.json"
    for current_questions in question_sets:
        for question in current_questions:
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
    api2_result_paths = [
        page_agent_out / "api2_runs" / f"run_{run_index:02d}" / "result.json"
        for run_index in range(1, review_runs + 1)
    ]
    outputs = {
        "result_json": str(result_json_path),
        "api2_results": [str(path) for path in api2_result_paths],
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
            "call_count": 1 + review_runs,
            "call_1": "whole-page draft Markdown",
            "call_2": "question-conditioned symbol, figure, and atomic patch review",
            "api2_run_count": review_runs,
            "local_merge": "deterministic validation against the first-pass draft",
        },
        "api_metrics": _api_metrics(first_pass_vlm, second_pass_vlms),
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
    for run_index, (path, current_review, current_questions, vlm) in enumerate(
        zip(api2_result_paths, reviews, question_sets, second_pass_vlms),
        start=1,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "page": page_name,
                    "api2_run": run_index,
                    "questions": current_questions,
                    "page_notes": current_review.get("page_notes") or "",
                    "api_requests": list(vlm.request_log),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    result_md_path.write_text(
        _render_result_markdown(page_name, questions),
        encoding="utf-8",
    )
    verification = {
        "mode": (
            "single_global_api2_symbol_figure_audited_atomic_patch_v4"
            if review_runs == 1
            else "repeated_global_api2_symbol_figure_audited_atomic_patch_v4"
        ),
        "first_pass_markdown": str(draft_markdown_path),
        "context_manifest": str(question_contexts_path),
        "api2_run_count": review_runs,
        "api_metrics": final["api_metrics"],
    }
    if review_runs == 1:
        verification["page_notes"] = review.get("page_notes") or ""
        verification["raw"] = review.get("_raw") or review.get("raw") or ""
    else:
        verification["page_notes"] = [current.get("page_notes") or "" for current in reviews]
        verification["raw_runs"] = [
            current.get("_raw") or current.get("raw") or "" for current in reviews
        ]
    verification_path.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2),
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
    parser.add_argument(
        "--review-runs",
        type=int,
        default=1,
        help="independent API2 reviews over the same API1 draft",
    )
    parser.add_argument(
        "--baseline-timeout",
        type=int,
        default=360,
        help="API1 upload/read timeout in seconds",
    )
    parser.add_argument(
        "--review-timeout",
        type=int,
        default=360,
        help="API2 upload/read timeout in seconds",
    )
    parser.add_argument(
        "--answer-detail-views",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="magnified answer details per question added to each API2 request",
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

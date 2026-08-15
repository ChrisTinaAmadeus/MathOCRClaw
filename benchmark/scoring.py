from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


QUESTION_START_RE = re.compile(r"(?m)^\s*(\d{1,3})[.．、]\s*")
ANSWER_HEADING_RE = re.compile(r"(?im)^\s*#{1,6}\s*手写答案\s*$")
FORMULA_RE = re.compile(r"\$\$(.+?)\$\$|(?<!\\)\$(.+?)(?<!\\)\$", re.S)
SUBPART_RE = re.compile(r"(?:\(|（)(\d+)(?:\)|）)")
OPTION_LINE_RE = re.compile(r"(?m)^\s*([A-D])[.．、]\s*")
IMAGE_RE = re.compile(r"<\s*插图\s*>|!\[[^\]]*\]\([^)]*\)", re.I)
NO_ANSWER_MARKERS = {
    "",
    "[无法识别]",
    "[unreadable]",
    "未识别到手写答案",
    "未识别到手写答案。",
    "no_supported_answer",
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _round_metrics(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round_metrics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_metrics(item) for item in value]
    return value


def normalize_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = IMAGE_RE.sub("<插图>", value)
    value = re.sub(r"(?m)^\s*#{1,6}\s*", "", value)
    value = value.replace("**", "").replace("__", "")
    value = re.sub(r"(?<!\\)[*_]", "", value)
    value = value.replace("，", ",").replace("。", ".").replace("：", ":")
    value = value.replace("；", ";").replace("（", "(").replace("）", ")")
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", "", value)
    return value.casefold()


def normalize_formula(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    aliases = {
        r"\dfrac": r"\frac",
        r"\tfrac": r"\frac",
        r"\cdot": "*",
        "·": "*",
        r"\leqslant": r"\le",
        r"\leq": r"\le",
        r"\geqslant": r"\ge",
        r"\geq": r"\ge",
        r"\mathbf{R}": "R",
        r"\mathbb{R}": "R",
        r"\left": "",
        r"\right": "",
    }
    for source, target in aliases.items():
        value = value.replace(source, target)
    value = value.replace("$", "")
    value = re.sub(r"\s+", "", value)
    return value.casefold()


def extract_formulas(text: Any) -> List[str]:
    result: List[str] = []
    for match in FORMULA_RE.finditer(str(text or "")):
        formula = match.group(1) if match.group(1) is not None else match.group(2)
        if formula is not None and formula.strip():
            result.append(formula.strip())
    return result


def text_without_formulas(text: Any) -> str:
    return FORMULA_RE.sub(" ", str(text or ""))


def is_no_supported_answer(text: Any, status: Any = "") -> bool:
    normalized = normalize_text(text).strip("._-()[]")
    if normalized in {normalize_text(marker).strip("._-()[]") for marker in NO_ANSWER_MARKERS}:
        return True
    return not normalized and str(status or "").strip().casefold() in {
        "",
        "no_answer",
        "unreadable",
        "u",
    }


def parse_markdown(markdown: str) -> List[Dict[str, Any]]:
    starts = list(QUESTION_START_RE.finditer(markdown or ""))
    questions: List[Dict[str, Any]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(markdown)
        block = markdown[start.start() : end].strip()
        heading = ANSWER_HEADING_RE.search(block)
        if heading:
            stem = block[: heading.start()].strip()
            answer = block[heading.end() :].strip()
        else:
            stem = block
            answer = ""
        stem = QUESTION_START_RE.sub("", stem, count=1).strip()
        questions.append(
            {
                "qno": str(int(start.group(1))),
                "order_index": index,
                "stem": stem,
                "answer": answer,
                "status": "no_answer" if is_no_supported_answer(answer) else "ok",
            }
        )
    return questions


def candidate_from_first_pass(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    parsed = parse_markdown(str(payload.get("page_markdown") or ""))
    if parsed:
        return parsed
    questions: List[Dict[str, Any]] = []
    for index, item in enumerate(payload.get("questions") or []):
        if not isinstance(item, dict):
            continue
        questions.append(
            {
                "qno": str(item.get("qno") or "").strip(),
                "order_index": index,
                "stem": str(item.get("question_text") or "").strip(),
                "answer": str(item.get("student_answer") or "").strip(),
                "status": str(item.get("answer_status") or "").strip(),
            }
        )
    return questions


def candidate_from_result(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    for index, item in enumerate(payload.get("questions") or []):
        if not isinstance(item, dict):
            continue
        answer = item.get("handwritten_answer")
        answer_obj = answer if isinstance(answer, dict) else {}
        qno = str(item.get("qno") or "").strip()
        stem = str(item.get("question_markdown") or "").strip()
        if qno:
            stem = re.sub(rf"^\s*{re.escape(qno)}[.．、]\s*", "", stem, count=1)
        questions.append(
            {
                "qno": qno,
                "order_index": index,
                "stem": stem,
                "answer": str(answer_obj.get("text") or "").strip(),
                "status": str(answer_obj.get("status") or "").strip(),
            }
        )
    return questions


def load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _multiset_f1(gold: Sequence[str], candidate: Sequence[str]) -> float:
    gold_counts = Counter(gold)
    candidate_counts = Counter(candidate)
    if not gold_counts and not candidate_counts:
        return 1.0
    overlap = sum((gold_counts & candidate_counts).values())
    precision = overlap / max(1, sum(candidate_counts.values()))
    recall = overlap / max(1, sum(gold_counts.values()))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _bigrams(value: str) -> List[str]:
    if len(value) <= 1:
        return [value] if value else []
    return [value[index : index + 2] for index in range(len(value) - 1)]


CRITICAL_RE = re.compile(
    r"(?:-?\d+(?:\.\d+)?(?:%|°)?)|(?:[<>]=?|≤|≥|=|≠)|"
    r"(?:不大于|不小于|大于|小于|至少|至多|不是|否定|且|或)|"
    r"(?:千米|厘米|毫米|米|秒|小时|kg|g|m|cm|mm)|(?:\b[A-D]\b)",
    re.I,
)


def critical_spans(text: Any) -> List[str]:
    return [normalize_text(match.group(0)) for match in CRITICAL_RE.finditer(str(text or ""))]


def text_score(gold: Any, candidate: Any) -> Optional[float]:
    gold_text = normalize_text(text_without_formulas(gold))
    candidate_text = normalize_text(text_without_formulas(candidate))
    if not gold_text and not candidate_text:
        return None
    cer = min(1.0, _edit_distance(gold_text, candidate_text) / max(1, len(gold_text)))
    bigram_f1 = _multiset_f1(_bigrams(gold_text), _bigrams(candidate_text))
    semantic_proxy = SequenceMatcher(None, gold_text, candidate_text, autojunk=False).ratio()
    span_f1 = _multiset_f1(critical_spans(gold), critical_spans(candidate))
    score = 0.8 * (0.5 * (1.0 - cer) + 0.35 * bigram_f1 + 0.15 * semantic_proxy)
    score += 0.2 * span_f1
    gold_critical = Counter(critical_spans(gold))
    candidate_critical = Counter(critical_spans(candidate))
    if gold_critical and candidate_critical and gold_critical != candidate_critical:
        score = min(score, 0.70)
    return _clamp(score)


FORMULA_TOKEN_RE = re.compile(
    r"\\[A-Za-z]+|[A-Za-z]+|\d+(?:\.\d+)?|<=|>=|!=|[{}()[\]^_+\-*/=<>|,.:]"
)


def formula_pair_score(gold: str, candidate: str) -> float:
    left = normalize_formula(gold)
    right = normalize_formula(candidate)
    if left == right:
        return 1.0
    left_tokens = FORMULA_TOKEN_RE.findall(left)
    right_tokens = FORMULA_TOKEN_RE.findall(right)
    token_f1 = _multiset_f1(left_tokens, right_tokens)
    structure = SequenceMatcher(None, left, right, autojunk=False).ratio()
    score = 0.6 * structure + 0.4 * token_f1
    critical_pattern = re.compile(r"<=|>=|!=|[<>^_=+\-]|\d+(?:\.\d+)?")
    if Counter(critical_pattern.findall(left)) != Counter(critical_pattern.findall(right)):
        score = min(score, 0.45)
    return _clamp(score)


def _ordered_soft_score(
    gold: Sequence[Any],
    candidate: Sequence[Any],
    pair_score: Callable[[Any, Any], float],
) -> Optional[float]:
    if not gold and not candidate:
        return None
    rows, columns = len(gold), len(candidate)
    dp = [[0.0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            dp[row][column] = max(
                dp[row - 1][column],
                dp[row][column - 1],
                dp[row - 1][column - 1] + pair_score(gold[row - 1], candidate[column - 1]),
            )
    return _clamp(2.0 * dp[rows][columns] / max(1, rows + columns))


def formula_score(gold: Any, candidate: Any) -> Optional[float]:
    return _ordered_soft_score(extract_formulas(gold), extract_formulas(candidate), formula_pair_score)


def _labels(pattern: re.Pattern[str], text: Any) -> List[str]:
    return [match.group(1) for match in pattern.finditer(str(text or ""))]


def local_structure_score(gold: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    values: List[float] = []
    if gold.get("qno") or candidate.get("qno"):
        values.append(float(str(gold.get("qno")) == str(candidate.get("qno"))))
    for pattern in (OPTION_LINE_RE, SUBPART_RE):
        gold_labels = _labels(pattern, gold.get("stem"))
        candidate_labels = _labels(pattern, candidate.get("stem"))
        if gold_labels or candidate_labels:
            values.append(_multiset_f1(gold_labels, candidate_labels))
    gold_images = len(IMAGE_RE.findall(str(gold.get("stem") or "")))
    candidate_images = len(IMAGE_RE.findall(str(candidate.get("stem") or "")))
    if gold_images or candidate_images:
        values.append(1.0 - min(1.0, abs(gold_images - candidate_images) / max(1, gold_images)))
    return _mean(values) if values else 1.0


def _weighted_applicable(components: Sequence[Tuple[float, Optional[float]]]) -> float:
    applicable = [(weight, value) for weight, value in components if value is not None]
    total_weight = sum(weight for weight, _ in applicable)
    if not total_weight:
        return 1.0
    return _clamp(sum(weight * float(value) for weight, value in applicable) / total_weight)


def stem_score(gold: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    return _weighted_applicable(
        [
            (0.45, text_score(gold.get("stem"), candidate.get("stem"))),
            (0.45, formula_score(gold.get("stem"), candidate.get("stem"))),
            (0.10, local_structure_score(gold, candidate)),
        ]
    )


def _question_type(question: Dict[str, Any]) -> str:
    stem = str(question.get("stem") or "")
    answer = str(question.get("answer") or "")
    if len(OPTION_LINE_RE.findall(stem)) >= 2 and re.fullmatch(
        r"[\s,，、;；A-Da-d选答案为:：()（）]+", answer
    ):
        return "choice"
    if re.search(r"_{3,}|\\_\\_\\_|填空", stem):
        return "fill"
    return "solution"


def _option_set(text: Any) -> set[str]:
    value = unicodedata.normalize("NFKC", str(text or "")).upper()
    standalone = re.findall(r"(?<![A-Z])[A-D](?![A-Z])", value)
    if standalone:
        return set(standalone)
    compact = re.sub(r"[^A-D]", "", value)
    return set(compact) if len(compact) <= 4 else set()


def _last_conclusion(text: Any) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines[-1] if lines else str(text or "")


def answer_score(gold: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[float, str]:
    gold_answer = gold.get("answer") or ""
    candidate_answer = candidate.get("answer") or ""
    gold_empty = is_no_supported_answer(gold_answer, gold.get("status"))
    candidate_empty = is_no_supported_answer(candidate_answer, candidate.get("status"))
    kind = _question_type(gold)
    if gold_empty:
        return (1.0 if candidate_empty else 0.0), kind
    if candidate_empty:
        return 0.0, kind
    if kind == "choice":
        gold_options = _option_set(gold_answer)
        candidate_options = _option_set(candidate_answer)
        exact = float(bool(gold_options) and gold_options == candidate_options)
        option_f1 = _multiset_f1(sorted(gold_options), sorted(candidate_options))
        return 0.9 * exact + 0.1 * option_f1, kind
    if kind == "fill":
        formulas = formula_score(gold_answer, candidate_answer)
        value_score = formulas if formulas is not None else text_score(gold_answer, candidate_answer)
        unit_text = text_score(gold_answer, candidate_answer)
        return _weighted_applicable([(0.8, value_score), (0.2, unit_text)]), kind
    return (
        _weighted_applicable(
            [
                (0.55, formula_score(gold_answer, candidate_answer)),
                (0.20, text_score(gold_answer, candidate_answer)),
                (0.20, text_score(_last_conclusion(gold_answer), _last_conclusion(candidate_answer))),
                (
                    0.05,
                    _multiset_f1(
                        _labels(SUBPART_RE, gold_answer),
                        _labels(SUBPART_RE, candidate_answer),
                    ),
                ),
            ]
        ),
        kind,
    )


def _rough_similarity(gold: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    left = normalize_text(gold.get("stem"))
    right = normalize_text(candidate.get("stem"))
    text_similarity = SequenceMatcher(None, left, right, autojunk=False).ratio()
    qno_equal = bool(gold.get("qno")) and str(gold.get("qno")) == str(candidate.get("qno"))
    if not qno_equal and text_similarity < 0.28:
        return -math.inf
    type_equal = _question_type(gold) == _question_type(candidate)
    return (2.0 if qno_equal else -0.25) + 1.2 * text_similarity + 0.15 * type_equal


def align_questions(
    gold: Sequence[Dict[str, Any]],
    candidate: Sequence[Dict[str, Any]],
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    rows, columns = len(gold), len(candidate)
    dp = [[0.0] * (columns + 1) for _ in range(rows + 1)]
    action = [[""] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            choices = [(dp[row - 1][column], "gold_skip"), (dp[row][column - 1], "cand_skip")]
            similarity = _rough_similarity(gold[row - 1], candidate[column - 1])
            if math.isfinite(similarity) and similarity > 0.0:
                choices.append((dp[row - 1][column - 1] + similarity, "match"))
            dp[row][column], action[row][column] = max(choices, key=lambda item: item[0])

    pairs: List[Tuple[int, int]] = []
    row, column = rows, columns
    while row > 0 or column > 0:
        current = action[row][column] if row >= 0 and column >= 0 else ""
        if row > 0 and column > 0 and current == "match":
            pairs.append((row - 1, column - 1))
            row -= 1
            column -= 1
        elif column > 0 and (row == 0 or current == "cand_skip"):
            column -= 1
        else:
            row -= 1
    pairs.reverse()
    matched_gold = {left for left, _ in pairs}
    matched_candidate = {right for _, right in pairs}
    return (
        pairs,
        [index for index in range(rows) if index not in matched_gold],
        [index for index in range(columns) if index not in matched_candidate],
    )


def _macro_state_f1(gold_states: Sequence[bool], candidate_states: Sequence[bool]) -> float:
    scores: List[float] = []
    for label in (False, True):
        if label not in gold_states and label not in candidate_states:
            continue
        true_positive = sum(g == label and c == label for g, c in zip(gold_states, candidate_states))
        false_positive = sum(g != label and c == label for g, c in zip(gold_states, candidate_states))
        false_negative = sum(g == label and c != label for g, c in zip(gold_states, candidate_states))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return _mean(scores)


def score_page(
    gold_questions: Sequence[Dict[str, Any]],
    candidate_questions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    gold = list(gold_questions)
    candidate = list(candidate_questions)
    pairs, missing_gold, extra_candidate = align_questions(gold, candidate)
    pair_by_gold = {gold_index: candidate_index for gold_index, candidate_index in pairs}
    structure = 2 * len(pairs) / max(1, len(gold) + len(candidate))
    stem_scores: List[float] = []
    answer_scores: List[float] = []
    per_type: Dict[str, List[float]] = {"choice": [], "fill": [], "solution": []}
    gold_states: List[bool] = []
    candidate_states: List[bool] = []
    omission_count = 0
    hallucination_count = len(extra_candidate)
    details: List[Dict[str, Any]] = []

    for gold_index, gold_question in enumerate(gold):
        gold_supported = not is_no_supported_answer(
            gold_question.get("answer"), gold_question.get("status")
        )
        gold_states.append(gold_supported)
        if gold_index not in pair_by_gold:
            stem_scores.append(0.0)
            answer_scores.append(0.0)
            candidate_states.append(False)
            if gold_supported:
                omission_count += 1
            kind = _question_type(gold_question)
            per_type[kind].append(0.0)
            details.append(
                {
                    "qno": gold_question.get("qno"),
                    "alignment": "missing",
                    "stem_score": 0.0,
                    "answer_score": 0.0,
                    "answer_type": kind,
                }
            )
            continue
        candidate_index = pair_by_gold[gold_index]
        candidate_question = candidate[candidate_index]
        candidate_supported = not is_no_supported_answer(
            candidate_question.get("answer"), candidate_question.get("status")
        )
        candidate_states.append(candidate_supported)
        if gold_supported and not candidate_supported:
            omission_count += 1
        if not gold_supported and candidate_supported:
            hallucination_count += 1
        current_stem = stem_score(gold_question, candidate_question)
        current_answer, kind = answer_score(gold_question, candidate_question)
        stem_scores.append(current_stem)
        answer_scores.append(current_answer)
        per_type[kind].append(current_answer)
        details.append(
            {
                "qno": gold_question.get("qno"),
                "candidate_qno": candidate_question.get("qno"),
                "alignment": "matched",
                "stem_score": current_stem,
                "answer_score": current_answer,
                "answer_type": kind,
            }
        )

    state_score = _macro_state_f1(gold_states, candidate_states) if gold else 0.0
    raw = 0.15 * structure + 0.35 * _mean(stem_scores) + 0.40 * _mean(answer_scores)
    raw += 0.10 * state_score
    hallucination_rate = min(1.0, hallucination_count / max(1, len(gold)))
    total = 100.0 * raw * (1.0 - 0.25 * hallucination_rate)
    gold_formula_spans = sum(
        len(extract_formulas(question.get("stem")))
        + len(extract_formulas(question.get("answer")))
        for question in gold
    )
    candidate_formula_spans = sum(
        len(extract_formulas(question.get("stem")))
        + len(extract_formulas(question.get("answer")))
        for question in candidate
    )
    result = {
        "score": total,
        "raw_score": raw,
        "structure_score": structure,
        "stem_score": _mean(stem_scores),
        "answer_score": _mean(answer_scores),
        "state_macro_f1": state_score,
        "answer_by_type": {
            kind: (_mean(scores) if scores else None) for kind, scores in per_type.items()
        },
        "gold_question_count": len(gold),
        "candidate_question_count": len(candidate),
        "matched_question_count": len(pairs),
        "omission_count": omission_count,
        "omission_rate": omission_count / max(1, len(gold)),
        "hallucination_count": hallucination_count,
        "hallucination_rate": hallucination_rate,
        "format_diagnostics": {
            "gold_latex_formula_spans": gold_formula_spans,
            "candidate_latex_formula_spans": candidate_formula_spans,
            "formula_span_count_ratio": (
                min(1.0, candidate_formula_spans / gold_formula_spans)
                if gold_formula_spans
                else (1.0 if not candidate_formula_spans else 0.0)
            ),
        },
        "missing_gold_indices": missing_gold,
        "extra_candidate_indices": extra_candidate,
        "details": details,
    }
    return _round_metrics(result)


def score_files(
    gold_markdown_path: Path,
    first_pass_json_path: Path,
    final_result_json_path: Path,
) -> Dict[str, Any]:
    gold = parse_markdown(gold_markdown_path.read_text(encoding="utf-8"))
    if not gold:
        raise ValueError(f"no questions parsed from gold Markdown: {gold_markdown_path}")
    first_pass = candidate_from_first_pass(load_json(first_pass_json_path))
    final = candidate_from_result(load_json(final_result_json_path))
    baseline_score = score_page(gold, first_pass)
    workflow_score = score_page(gold, final)
    return {
        "evaluator": {
            "name": "mathocrclaw_deterministic_v3_local",
            "spec": "benchmark/EVALUATION.md",
            "judge_model_calls": 0,
            "semantic_mode": "local_sequence_similarity",
            "formula_mode": "normalized_latex_ordered_soft_alignment",
        },
        "baseline": baseline_score,
        "workflow": workflow_score,
        "gain": round(workflow_score["score"] - baseline_score["score"], 6),
    }

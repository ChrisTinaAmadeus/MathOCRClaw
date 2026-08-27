from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


_NO_ANSWER_RE = re.compile(r"未识别到手写答案|无法识别|unreadable", re.I)
_OPTION_RE = re.compile(r"(?m)^\s*([A-D])[.．、:：]\s*")
_SUBPART_RE = re.compile(r"(?:\(|（)(\d+)(?:\)|）)")


_SYMBOL_FAMILIES: Dict[str, Dict[str, Any]] = {
    "CHOICE_LETTER": {
        "symbols": ["A", "B", "C", "D"],
        "confusion_checks": [
            "B vs D: inspect whether the left stem has two lobes or one continuous outer bow",
            "A vs 4: inspect the apex/crossbar and whether a vertical right stem continues downward",
            "C vs an open D: inspect for a straight left stem and closure at the top/bottom",
        ],
    },
    "DIGIT": {
        "symbols": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "confusion_checks": [
            "1 vs 7, 3 vs 5, and 0 vs 6/9: inspect crossbars, loop closure, and stroke tails",
            "A small decimal point must be visibly separated from dust, print, or a pen stop",
        ],
    },
    "SIGN_RELATION": {
        "symbols": ["+", "-", "=", "<", ">", "\\le", "\\ge", "\\ne", "\\pm"],
        "confusion_checks": [
            "< vs \\le and > vs \\ge: require a visible lower equality bar",
            "- vs =: count distinct horizontal strokes; + vs -: require a crossing vertical stroke",
            "A faint leading minus must be distinguished from a fraction bar belonging to nearby terms",
        ],
    },
    "SET_INTERVAL": {
        "symbols": ["\\{", "\\}", "(", ")", "[", "]", "\\cup", "\\cap", "\\in", "\\notin", "\\varnothing"],
        "confusion_checks": [
            "Interval ( ) vs [ ]: inspect each endpoint separately for a straight closing edge",
            "\\cup vs \\cap: inspect opening direction; \\in vs \\notin: require the diagonal cancellation stroke",
            "\\varnothing vs 0/phi: inspect the diagonal slash and whether the loop is closed",
        ],
    },
    "FRACTION_ROOT": {
        "symbols": ["\\frac", "/", "\\sqrt"],
        "confusion_checks": [
            "A fraction bar must span a numerator and denominator; do not turn an isolated minus into a fraction",
            "For a radical, inspect both the check-shaped hook and the overline endpoint",
        ],
    },
    "SCRIPT_POSITION": {
        "symbols": ["superscript", "subscript", "coefficient"],
        "confusion_checks": [
            "Use vertical position and baseline attachment to distinguish exponents, subscripts, and neighboring terms",
            "Inspect multi-digit scripts as a group; do not silently move a digit onto the baseline",
        ],
    },
    "GEOMETRY_MARK": {
        "symbols": [
            "\\parallel",
            "\\mathrel{\\underline{\\parallel}}",
            "\\perp",
            "\\angle",
            "\\triangle",
            "\\overrightarrow{}",
        ],
        "confusion_checks": [
            "\\parallel vs = and \\perp vs T: inspect orientation and whether the strokes belong to a labeled relation",
            "\\mathrel{\\underline{\\parallel}} means parallel and equal: require two parallel strokes plus a separate visible underline; do not collapse it to plain \\parallel or =",
            "A vector arrow requires a visible arrowhead; do not infer it from the printed stem",
        ],
    },
    "GREEK_LATIN": {
        "symbols": ["\\alpha", "a", "\\beta", "b", "\\theta", "0", "\\rho", "p", "\\phi", "\\varnothing"],
        "confusion_checks": [
            "Use loop closure, descenders, and surrounding handwritten baseline to compare Greek and Latin candidates",
            "\\phi vs \\varnothing and \\theta vs 0 require the internal/diagonal stroke to be visibly present",
        ],
    },
    "FUNCTION_OPERATOR": {
        "symbols": ["\\sin", "\\cos", "\\tan", "\\log", "\\ln", "e", "\\lim"],
        "confusion_checks": [
            "Read operator letters as one token and keep adjacent base/subscript/exponent positions separate",
        ],
    },
    "PHYSICS_UNIT": {
        "symbols": ["m/s", "m/s^2", "N", "J", "W", "V", "A", "\\Omega", "Hz"],
        "confusion_checks": [
            "Preserve case, unit division, powers, decimal points, and the distinction between a value and its unit",
            "\\Omega vs 0/O requires the open lower gap or feet of the omega glyph to be visible",
        ],
    },
    "CHEM_FORMULA": {
        "symbols": ["+", "\\rightarrow", "\\rightleftharpoons", "(g)", "(l)", "(s)", "(aq)"],
        "confusion_checks": [
            "O vs 0 and l vs 1: use neighboring element letters and stroke shape, never chemical plausibility alone",
            "Separate leading coefficients from subscripts by baseline position; inspect charge superscripts independently",
            "A reaction arrow/equilibrium arrow must be visibly drawn and must not be inferred by balancing the equation",
        ],
    },
}


def _canonical_type(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in {"choice", "multiple_choice", "multiple_choice_question"}:
        return "choice"
    if raw in {"fill", "fill_blank", "fill_blank_question"}:
        return "fill"
    return "solution"


def _append_unique(items: List[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in items:
            items.append(value)


def _family_names(question_text: str, question_type: str) -> List[str]:
    text = str(question_text or "")
    names: List[str] = []
    if question_type == "choice":
        return ["CHOICE_LETTER"]
    names.append("DIGIT")

    if re.search(r"不等式|取值范围|大小关系|\\le|\\ge|\\ne|[<>≤≥≠±]", text):
        _append_unique(names, ["SIGN_RELATION"])
    if re.search(
        r"集合|区间|\\(?:cup|cap|in|notin|subset|supset|varnothing|emptyset)|"
        r"[∪∩∈∉⊂⊆∅]|\\[{}]|\[[^\]]+\]",
        text,
    ):
        _append_unique(names, ["SET_INTERVAL"])
    if re.search(r"\\frac|\\sqrt|分式|根式|平方根", text):
        _append_unique(names, ["FRACTION_ROOT"])
    if re.search(r"(?<!\\)[_^]|指数|幂|下标|上标", text):
        _append_unique(names, ["SCRIPT_POSITION"])
    if re.search(
        r"几何|三角形|四边形|棱|平面|直线|向量|法向量|垂直|平行|"
        r"\\(?:triangle|angle|perp|parallel|vec|overrightarrow)",
        text,
    ):
        _append_unique(names, ["GEOMETRY_MARK"])
    if re.search(
        r"\\(?:alpha|beta|gamma|delta|theta|lambda|mu|rho|phi|omega)|"
        r"[αβγδθλμρφω]",
        text,
    ):
        _append_unique(names, ["GREEK_LATIN"])
    if re.search(r"函数|三角|对数|\\(?:sin|cos|tan|log|ln|lim)|指数函数", text):
        _append_unique(names, ["FUNCTION_OPERATOR"])
    if re.search(r"物理|加速度|电流|电压|电阻|动能|势能|机械能|电场|磁场", text) or re.search(
        r"(?:速度|质量|功率|频率).{0,40}(?:m/s|kg|kW|W|Hz)",
        text,
        re.I,
    ):
        _append_unique(names, ["PHYSICS_UNIT"])
    if re.search(r"化学|反应方程式|物质的量|mol|溶液|化合物|气体|沉淀", text, re.I):
        _append_unique(names, ["CHEM_FORMULA"])
    return names


def _risk(question_type: str, draft_answer: str) -> Dict[str, Any]:
    answer = str(draft_answer or "").strip()
    reasons: List[str] = []
    if not answer or _NO_ANSWER_RE.search(answer):
        reasons.append("api1_has_no_supported_reading")
    if question_type == "choice":
        compact = re.sub(r"[^A-D]", "", answer.upper())
        if not compact or re.sub(r"[A-D\s$`*_，,、;；:：()（）\[\]]", "", answer.upper()):
            reasons.append("choice_answer_outside_closed_grammar")
        reasons.append("small_single_glyph_answer")
    elif question_type == "fill":
        reasons.append("compact_formula_answer")
    priority = "high" if "api1_has_no_supported_reading" in reasons else (
        "medium" if reasons else "low"
    )
    return {"priority": priority, "reasons": reasons}


def build_recognition_profile(
    question_text: str,
    question_type: Any,
    draft_answer: str = "",
) -> Dict[str, Any]:
    """Build candidate-only symbol tags that guide visual inspection without solving.

    The profile deliberately describes alphabets and discriminating strokes, not the
    correct answer.  It is safe to expose to API2 as routing metadata because every
    selected symbol must still be justified from a supplied image.
    """
    canonical_type = _canonical_type(question_type)
    options = list(dict.fromkeys(_OPTION_RE.findall(str(question_text or "").upper())))
    if canonical_type == "choice" and not options:
        options = ["A", "B", "C", "D"]
    subparts = list(dict.fromkeys(_SUBPART_RE.findall(str(question_text or ""))))
    family_names = _family_names(str(question_text or ""), canonical_type)
    families = [
        {"tag": name, **_SYMBOL_FAMILIES[name]}
        for name in family_names
    ]

    if canonical_type == "choice":
        grammar = {
            "kind": "closed_choice_alphabet",
            "allowed_final_tokens": options,
            "rule": "one or more unique visibly retained option letters only",
        }
    elif canonical_type == "fill":
        grammar = {
            "kind": "compact_final_value_or_formula",
            "expected_part_labels": [f"({value})" for value in subparts] or ["overall"],
            "rule": "final value/formula and necessary unit only; exclude derivation",
        }
    else:
        grammar = {
            "kind": "ordered_solution_transcription",
            "expected_part_labels": [f"({value})" for value in subparts],
            "rule": "preserve visible retained steps; tags guide only local glyph checks",
        }

    return {
        "version": "symbol_profile_v1",
        "question_type": canonical_type,
        "answer_grammar": grammar,
        "symbol_families": families,
        "recognition_risk": _risk(canonical_type, draft_answer),
        "usage_rule": (
            "Candidate families are an inspection checklist, never evidence or an answer key. "
            "Select a candidate only when the cited image shows its discriminating stroke."
        ),
    }


def profile_tags(profile: Dict[str, Any]) -> set[str]:
    return {
        str(family.get("tag") or "")
        for family in profile.get("symbol_families") or []
        if isinstance(family, dict) and family.get("tag")
    }

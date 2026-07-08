from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image

from .cache import JsonCache
from .common import parse_verdict_letter, sha1_text
from .img_utils import BASE_ENHANCE, STRONG_ENHANCE, enhance_for_vlm, img_to_data_url

VERIFIER_PROMPT_STRICT = (
    "You are a strict comparison verifier.\n"
    "You are given: 1) a cropped question image (containing printed text only); 2) a candidate text.\n"
    "Task: determine whether the candidate text matches the **printed text** in the image.\n"
    "Allowed: equivalent LaTeX expressions / whitespace differences / line-break differences.\n"
    "Not allowed: adding nonexistent content, omitting key content, or changing numbers / symbols / options incorrectly.\n"
    "If the image is blurry or incompletely cropped so that you cannot determine consistency, output U (do not guess N).\n"
    "Your output must be a single letter only: Y / N / U.\n"
    "Meaning: Y = consistent; N = inconsistent; U = uncertain (unclear / poor image quality / incomplete crop / hard to judge).\n"
)

VERIFIER_PROMPT_LENIENT = (
    "You are a conservative comparison verifier.\n"
    "You are given: a cropped question image (printed text only) + a candidate text.\n"
    "If you can clearly determine that the candidate text does not match the image -> N.\n"
    "If you cannot confirm a mismatch (it may be caused only by blur / compression / resolution / incomplete crop) -> U.\n"
    "If you can confirm they match -> Y.\n"
    "Output only a single letter: Y / N / U.\n"
)

SPAN_PROMPT = (
    "You are an Anti-Hallucination Verifier.\n"
    "Judge only based on the printed content in the image, without relying on mathematical common sense.\n"
    "Task: determine whether the **key elements** in the candidate text (numbers / variables / operators / option letters / symbols) can be found in the image.\n"
    "Rules:\n"
    "- If you can confirm that all these key elements appear in the image -> Y.\n"
    "- If you can confirm that some key elements are missing or added without evidence -> N.\n"
    "- If blur / incomplete crop / occlusion prevents confirmation -> U (do not guess N).\n"
    "Output only a single letter: Y / N / U.\n"
)

READABLE_SPAN_PROMPT = (
    "You are a readability verifier. Look only at the printed text in the image.\n"
    "You are given a candidate text span. Judge whether it is **fully, clearly, and character-by-character verifiable** in the image.\n"
    "- If all key characters / symbols are clearly visible -> Y.\n"
    "- If you can confirm that this content is not in the image, or is clearly different -> N.\n"
    "- If any key part cannot be confirmed due to occlusion / correction marks / blur / incomplete crop -> U.\n"
    "Output only a single letter: Y / N / U.\n"
)

EVIDENCE_PROMPT = (
    "You are an evidence verifier. Judge only whether the candidate text can be **directly seen** in the printed text of the image.\n"
    "If you cannot find the corresponding text / symbol in this cropped image (even if it may exist outside the crop), always output N.\n"
    "Only output Y when you can clearly see the corresponding content.\n"
    "Output only a single letter: Y or N (do not output U).\n"
)

OPTION_MARKER_PROMPT = (
    "You are a multiple-choice structure detector. Look only at the printed text in the image.\n"
    "Determine whether the cropped question image contains multiple-choice option markers (such as A./B./C./D., or (A)(B)(C)(D), or A、B、C、D).\n"
    "- If you can clearly see at least one option marker -> Y.\n"
    "- If you can clearly determine that there is no option marker at all -> N.\n"
    "- If blur / occlusion / incomplete crop prevents judgment -> U.\n"
    "Output only a single letter: Y / N / U.\n"
)

def chunk_text(text: str, max_chars: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]
    lines = text.splitlines()
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for ln in lines:
        if cur_len + len(ln) + 1 > max_chars and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(ln)
        cur_len += len(ln) + 1
    if cur:
        chunks.append("\n".join(cur))
    return [c for c in chunks if c is not None]

@dataclass
class VerifyResult:
    verdict: str  # Y/N/U
    raw: str
    chunks: List[Dict[str, Any]]

def _cache_key(ns: str, model: str, img_hash: str, prompt_id: str, text: str) -> str:
    return f"{model}::{img_hash}::{prompt_id}::{sha1_text(text)}"

def verify_text(
    vlm,
    img: Image.Image,
    text: str,
    *,
    prompt: str,
    prompt_id: str,
    enhance_mode: str = "base",
    max_chars: int = 900,
    cache: Optional[JsonCache] = None,
    cache_ns: str = "ver",
) -> VerifyResult:
    cfg = BASE_ENHANCE if enhance_mode == "base" else STRONG_ENHANCE
    img2 = enhance_for_vlm(img, cfg)
    data_url, img_hash = img_to_data_url(img2)

    chunks = chunk_text(text, max_chars=max_chars)
    out_chunks: List[Dict[str, Any]] = []
    worst = "Y"
    raw_all: List[str] = []

    for idx, ch in enumerate(chunks):
        ck = _cache_key(cache_ns, vlm.cache_tag, img_hash, prompt_id, ch or "")
        cached = cache.get(cache_ns, ck) if cache else None
        if isinstance(cached, dict) and "verdict" in cached:
            v = cached.get("verdict", "U")
            raw = cached.get("raw", "")
        else:
            raw = vlm.invoke(
                [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "候选文本：\n" + (ch or "")},
                ],
                #temperature=0.0,
                max_tokens=8,
            )
            v = parse_verdict_letter(raw)
            if cache:
                cache.set(cache_ns, ck, {"verdict": v, "raw": raw})
        raw_all.append(raw)
        out_chunks.append({"i": idx, "verdict": v, "text": ch, "raw": raw})
        if v == "N":
            worst = "N"
        elif v == "U" and worst == "Y":
            worst = "U"

    return VerifyResult(verdict=worst, raw="\n".join(raw_all), chunks=out_chunks)

def verify_question_strict(vlm, img: Image.Image, text: str, *, cache: Optional[JsonCache] = None, max_chars: int = 900) -> VerifyResult:
    return verify_text(vlm, img, text, prompt=VERIFIER_PROMPT_STRICT, prompt_id="q_strict", enhance_mode="base", cache=cache, max_chars=max_chars)

def verify_question_lenient(vlm, img: Image.Image, text: str, *, cache: Optional[JsonCache] = None, max_chars: int = 900) -> VerifyResult:
    return verify_text(vlm, img, text, prompt=VERIFIER_PROMPT_LENIENT, prompt_id="q_lenient", enhance_mode="strong", cache=cache, max_chars=max_chars)

def verify_span(vlm, img: Image.Image, text: str, *, cache: Optional[JsonCache] = None) -> VerifyResult:
    return verify_text(vlm, img, text, prompt=SPAN_PROMPT, prompt_id="span", enhance_mode="strong", cache=cache, max_chars=480, cache_ns="ver_span")

def verify_span_readable(vlm, img: Image.Image, text: str, *, cache: Optional[JsonCache] = None) -> VerifyResult:
    return verify_text(vlm, img, text, prompt=READABLE_SPAN_PROMPT, prompt_id="span_readable", enhance_mode="strong", cache=cache, max_chars=320, cache_ns="ver_span_readable")

def verify_evidence(vlm, img: Image.Image, text: str, *, cache: Optional[JsonCache] = None) -> VerifyResult:
    r = verify_text(vlm, img, text, prompt=EVIDENCE_PROMPT, prompt_id="evidence", enhance_mode="strong", cache=cache, max_chars=260, cache_ns="ver_evi")
    if r.verdict != "Y":
        r.verdict = "N"
    return r

def verify_has_options(vlm, img: Image.Image, *, cache: Optional[JsonCache] = None) -> VerifyResult:
    # image-only probe: pass empty text
    return verify_text(vlm, img, "", prompt=OPTION_MARKER_PROMPT, prompt_id="has_opt", enhance_mode="strong", cache=cache, max_chars=1, cache_ns="ver_struct")

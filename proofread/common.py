from __future__ import annotations

import difflib
import hashlib
import re
from typing import Optional

_QNO_LINE_RE = re.compile(
    r"^\s*(?:第\s*)?(\d{1,3})\s*(?:题)?\s*[\.．、\)]\s*"
)

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def sha1_text(s: str) -> str:
    return sha1_bytes(s.encode("utf-8", errors="ignore"))

def normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\u3000+", " ", s)  # full-width spaces
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def text_similarity(a: str, b: str) -> float:

    na = re.sub(r"\s+", "", a or "")
    nb = re.sub(r"\s+", "", b or "")
    if not na and not nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()

def extract_qno_from_text_block(text: str) -> Optional[int]:
    if not text:
        return None
    for ln in text.splitlines():
        if ln.strip():
            m = _QNO_LINE_RE.match(ln.strip())
            if m:
                try:
                    qno = int(m.group(1))
                    return qno if 1 <= qno <= 300 else None
                except Exception:
                    return None
            break
    return None

def parse_verdict_letter(raw: str) -> str:

    if not raw:
        return "U"
    m = re.search(r"\b([YNU])\b", raw.upper())
    return m.group(1) if m else "U"


from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .common import normalize_ws, extract_qno_from_text_block

_QNO_LINE_RE = re.compile(r"^\s*(?:第\s*)?(\d{1,3})\s*(?:题)?\s*[\.．、]\s*")

_INTER_HEADING_RE = re.compile(
    r"""^\s*(?:
        [一二三四五六七八九十]+\s*[、\.．]\s*.*|     # 一、...  二.
        第\s*[一二三四五六七八九十0-9]{1,3}\s*部分\s*.*|  # 第X部分
        [（(]?[一二三四五六七八九十]+[)）]\s*.*         # （一）...
    )$|^\s*(?:单选题|多选题|选择题|填空题|解答题|阅读题|计算题)\b.*$""",
    re.X,
)

_INLINE_OPT_MARK = re.compile(r"\b([A-D])\s*[\.．、:：]\s*")
_MD_IMAGE_LINE = re.compile(r"^\s*!\[.*?\]\(.*?\)\s*$")

@dataclass
class PageBlock:
    kind: str  # "q" or "inter"
    qno: Optional[int]
    text: str

def split_inline_options(line: str) -> List[str]:
    marks = list(_INLINE_OPT_MARK.finditer(line))
    if len(marks) < 2:
        return [line]
    out: List[str] = []
    for i, m in enumerate(marks):
        st = m.start()
        ed = marks[i + 1].start() if i + 1 < len(marks) else len(line)
        seg = line[st:ed].strip()
        if seg:
            out.append(seg)
    return out or [line]

def clean_page_md(md: str) -> str:
    md = (md or "").lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in md.split("\n")]

    # drop leading status markers
    while lines and lines[0].strip() in ("[格式正常]", "[格式异常]", "[格式可疑]"):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]

    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if s in ("[格式正常]", "[格式异常]", "[格式可疑]"):
            continue
        if _MD_IMAGE_LINE.match(ln):
            continue
        # split inline options only when clearly present
        if _INLINE_OPT_MARK.search(ln):
            out.extend(split_inline_options(ln))
        else:
            out.append(ln.replace("\u3000", " "))

    return normalize_ws("\n".join(out))

def _is_qno_only_block(text: str, qno: Optional[int]) -> bool:
    if not text or qno is None:
        return False
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) != 1:
        return False
    ln = lines[0]
    m = _QNO_LINE_RE.match(ln)
    if not m:
        return False
    try:
        n = int(m.group(1))
    except Exception:
        return False
    if n != int(qno):
        return False
    rest = ln[m.end():].strip()
    return (rest == "")

def split_page_into_blocks(md: str) -> List[PageBlock]:
    md = clean_page_md(md)
    lines = [ln.rstrip() for ln in md.split("\n")]

    blocks: List[PageBlock] = []
    cur_kind = "inter"
    cur_qno: Optional[int] = None
    cur_lines: List[str] = []

    def flush():
        nonlocal cur_kind, cur_qno, cur_lines
        txt = normalize_ws("\n".join([x.rstrip() for x in cur_lines if x is not None]))
        if txt.strip():
            blocks.append(PageBlock(kind=cur_kind, qno=cur_qno, text=txt))
        cur_kind, cur_qno, cur_lines = "inter", None, []

    for ln in lines:
        m = _QNO_LINE_RE.match(ln)
        if m:
            flush()
            try:
                cur_qno = int(m.group(1))
            except Exception:
                cur_qno = extract_qno_from_text_block(ln)
            cur_kind = "q"
            cur_lines = [ln]
        else:
            cur_lines.append(ln)
    flush()

    peeled: List[PageBlock] = []
    for b in blocks:
        if b.kind != "q":
            peeled.append(b)
            continue
        blines = b.text.splitlines()
        cut = len(blines)
        for k in range(len(blines) - 1, -1, -1):
            if _INTER_HEADING_RE.match(blines[k].strip()):
                cut = min(cut, k)
        core = normalize_ws("\n".join(blines[:cut]))
        tail = normalize_ws("\n".join(blines[cut:]))
        if core.strip():
            peeled.append(PageBlock(kind="q", qno=b.qno, text=core))
        if tail.strip():
            peeled.append(PageBlock(kind="inter", qno=None, text=tail))
    out: List[PageBlock] = []
    i = 0
    while i < len(peeled):
        b = peeled[i]
        if b.kind == "q" and _is_qno_only_block(b.text, b.qno):
            i += 1
            continue
        out.append(b)
        i += 1

    return out

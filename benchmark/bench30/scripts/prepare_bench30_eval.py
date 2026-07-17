#!/usr/bin/env python3
"""Re-sync Approach-A GT and verify bench30 eval readiness (no LLM calls)."""
from __future__ import annotations

import csv
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
img_pat = re.compile(r'(!\[[^\]]*\]\()([^)]+)(\))|(<img[^>]+src=["\'])([^"\']+)(["\'])', re.I)


def iter_refs(text: str):
    for m in img_pat.finditer(text):
        yield (m.group(2), m.start(2), m.end(2)) if m.group(2) is not None else (
            m.group(5),
            m.start(5),
            m.end(5),
        )


def sync_gt() -> None:
    rows = list(csv.DictReader((ROOT / "benchmark/manifest.tsv").open(encoding="utf-8"), delimiter="\t"))
    for r in rows:
        i = int(r["id"])
        stem = Path(r["src_image"]).stem
        src_md = ROOT / r["gt_md"]
        gt_dir = ROOT / f"data/image_{i}/gt"
        pred_dir = ROOT / f"data/image_{i}/qwen3_vl_baseline"
        gt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        raw_dir = pred_dir / "raw"
        for raw in list(pred_dir.glob("*.raw.md")):
            raw_dir.mkdir(exist_ok=True)
            shutil.move(str(raw), str(raw_dir / raw.name))

        text = src_md.read_text(encoding="utf-8")
        replacements = []
        for ref, a, b in iter_refs(text):
            if ref.startswith(("http://", "https://", "data:")):
                continue
            cand = (src_md.parent / ref).resolve()
            if not cand.exists():
                cand = (src_md.parent.parent / Path(ref).name).resolve()
            if not cand.exists():
                found = list(src_md.parent.glob(Path(ref).name)) + list(
                    src_md.parent.parent.glob(Path(ref).name)
                )
                cand = found[0].resolve() if found else None
            if cand and cand.exists() and cand.is_file():
                dest = gt_dir / cand.name
                if not dest.exists() or dest.stat().st_size != cand.stat().st_size:
                    shutil.copy2(cand, dest)
                if "/" in ref or ref.startswith(".") or Path(ref).name != ref:
                    replacements.append((a, b, cand.name))
        for a, b, new in sorted(replacements, key=lambda x: x[0], reverse=True):
            text = text[:a] + new + text[b:]
        (gt_dir / f"{stem}.md").write_text(text, encoding="utf-8")
        pred = pred_dir / f"{stem}.md"
        if not pred.exists() or pred.stat().st_size == 0:
            raise SystemExit(f"missing baseline: {pred}")
    print(f"synced {len(rows)} GT files (Approach A)")


def verify() -> None:
    sys.path.insert(0, str(ROOT))
    from config import PATH_PAIRS, DASHSCOPE_API_KEY
    from main import load_data
    from utils.text_metric import TextMetricsCalculator

    assert len(PATH_PAIRS) == 30, len(PATH_PAIRS)
    assert DASHSCOPE_API_KEY, "DASHSCOPE_API_KEY empty"
    TextMetricsCalculator()
    total = sum(len(load_data(g, p)) for g, p in PATH_PAIRS)
    assert total == 30, total
    print(f"PREP_OK: PATH_PAIRS=30 matched_pairs=30 api_key=yes")


if __name__ == "__main__":
    sync_gt()
    verify()

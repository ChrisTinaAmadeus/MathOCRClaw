#!/usr/bin/env python3
"""Batch VLM extraction for MathDoc small baseline (manifest-driven)."""

from __future__ import annotations

import argparse
import base64
import csv
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]


def load_api_key() -> str:
    for candidate in [
        ROOT / "api_key.txt",
        ROOT / ".env",
    ]:
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8").strip()
        if candidate.name == ".env":
            for line in text.splitlines():
                if line.startswith("DASHSCOPE_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        elif text:
            return text.splitlines()[0].strip()
    key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    raise SystemExit("未找到 API Key：请设置 api_key.txt / .env / DASHSCOPE_API_KEY")


def mime_of(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".png":
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def strip_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$", text, re.I)
    if m:
        return m.group(1).strip() + "\n"
    return text if text.endswith("\n") else text + "\n"


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise SystemExit(f"manifest 为空: {path}")
    return rows


def extract_one(
    client: OpenAI,
    *,
    image_path: Path,
    prompt: str,
    model: str,
    max_retries: int = 3,
) -> str:
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    data_url = f"data:{mime_of(image_path)};base64,{b64}"
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            )
            content = resp.choices[0].message.content or ""
            return strip_fence(content)
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  retry {attempt}/{max_retries} after error: {e}; sleep {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"提取失败 {image_path}: {last_err}")


def setup_dirs_and_gt(rows: list[dict], data_root: Path, copy_gt: bool) -> None:
    for row in rows:
        id_ = int(row["id"])
        stem = Path(row["src_image"]).stem
        gt_dir = data_root / f"image_{id_}" / "gt"
        pred_dir = data_root / f"image_{id_}" / "qwen3_vl_baseline"
        gt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        if copy_gt and row.get("gt_md"):
            src = ROOT / row["gt_md"]
            if src.exists():
                dst = gt_dir / f"{stem}.md"
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                # copy sibling images referenced as ../xxx.jpg if present
                img_parent = src.parent.parent  # e.g. .../problem_images_output or folder with jpgs
                # also check parent of markdowns
                for ref_dir in {src.parent, src.parent.parent}:
                    for img in ref_dir.glob("*.jpg"):
                        # only copy if referenced in md
                        if img.name in dst.read_text(encoding="utf-8"):
                            (gt_dir / img.name).write_bytes(img.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract baseline markdown with Qwen VL")
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmark/manifest.tsv")
    parser.add_argument("--prompt", type=Path, default=ROOT / "benchmark/prompts/extract_v1.txt")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--model", default="qwen3-vl-plus")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--ids", default="", help="comma-separated ids, empty=all")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--setup-only", action="store_true", help="only create dirs + copy GT")
    parser.add_argument("--no-copy-gt", action="store_true")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        rows = [r for r in rows if r["id"] in want]

    setup_dirs_and_gt(rows, args.data_root, copy_gt=not args.no_copy_gt)
    print(f"prepared {len(rows)} image_* dirs under {args.data_root}")
    if args.setup_only:
        return

    prompt = args.prompt.read_text(encoding="utf-8")
    # drop comment-only header lines starting with '# MathDoc' style file comments at top? keep full file including comments - model can ignore
    # Actually the # lines at top are fine as instructions context; keep all.

    api_key = load_api_key()
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=180.0,
    )

    def job(row: dict) -> tuple[str, str]:
        id_ = row["id"]
        src = ROOT / row["src_image"]
        stem = src.stem
        out_dir = args.data_root / f"image_{id_}" / "qwen3_vl_baseline"
        out_md = out_dir / f"{stem}.md"
        out_raw = out_dir / f"{stem}.raw.md"
        if args.skip_existing and out_md.exists() and out_md.stat().st_size > 0:
            return id_, f"skip existing {out_md}"
        if not src.exists():
            return id_, f"MISSING image {src}"
        time.sleep(args.delay)
        text = extract_one(client, image_path=src, prompt=prompt, model=args.model)
        out_raw.write_text(text, encoding="utf-8")
        out_md.write_text(text, encoding="utf-8")
        return id_, f"ok -> {out_md} ({len(text)} chars)"

    print(f"model={args.model} workers={args.workers} items={len(rows)}")
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(job, r): r["id"] for r in rows}
        for fut in as_completed(futs):
            id_ = futs[fut]
            try:
                rid, msg = fut.result()
                print(f"[{rid}] {msg}", flush=True)
                if msg.startswith("ok") or msg.startswith("skip"):
                    ok += 1
                else:
                    fail += 1
            except Exception as e:
                fail += 1
                print(f"[{id_}] FAIL {e}", flush=True)

    print(f"done: ok/skip={ok} fail={fail}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()

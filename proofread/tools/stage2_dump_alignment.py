from __future__ import annotations

import argparse
import json
from pathlib import Path

from proofread.align import assign_crops_to_md_blocks
from proofread.cache import JsonCache
from proofread.md_utils import split_page_into_blocks
from proofread.match_utils import load_match_questions
from proofread.vlm_client import VLMClient

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-dir", required=True)
    ap.add_argument("--page-md", required=True)
    ap.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--qno-model", default="qwen3-vl-235b-a22b-instruct")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--cache-path", default="")
    ap.add_argument("--no-crop-qno", action="store_true")
    ap.add_argument("--keep-partial", action="store_true")
    ap.add_argument("--no-offset-search", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    page_dir = Path(args.page_dir)
    md = Path(args.page_md).read_text(encoding="utf-8", errors="ignore")
    blocks = split_page_into_blocks(md)

    meta, qs_sorted = load_match_questions(page_dir/"match.json")
    page_h = meta.get("page_h", None)
    try:
        page_h = float(page_h) if page_h is not None else None
    except Exception:
        page_h = None

    cache = None
    if args.cache:
        cache_path = Path(args.cache_path) if args.cache_path else (page_dir/"_cache_align.json")
        cache = JsonCache(cache_path)

    use_crop_qno = not args.no_crop_qno
    #qno_vlm = VLMClient(api_base=args.api_base, api_key=args.api_key, model=args.qno_model) if use_crop_qno else None
    qno_vlm = (VLMClient(
        api_base=args.api_base, api_key=args.api_key, model=args.qno_model,
       temperature=args.temperature, top_p=args.top_p
        ) if use_crop_qno else None)
    res = assign_crops_to_md_blocks(
        blocks,
        qs_sorted,
        page_dir=page_dir,
        page_h=page_h,
        skip_partial=(not args.keep_partial),
        use_offset_search=(not args.no_offset_search),
        use_crop_qno=use_crop_qno,
        qno_vlm=qno_vlm,
        cache=cache,
    )
    obj = {"align": res.debug, "mapping": res.mapping}
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(s, encoding="utf-8")
    else:
        print(s)
    if cache:
        cache.save()

if __name__ == "__main__":
    main()

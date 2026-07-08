from __future__ import annotations

import argparse
import json
from pathlib import Path

from proofread.cache import JsonCache
from proofread.img_utils import safe_open_image
from proofread.verify import verify_question_lenient, verify_question_strict
from proofread.vlm_client import VLMClient

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", required=True)
    ap.add_argument("--text", required=True, help="path to txt/md or literal if --literal")
    ap.add_argument("--literal", action="store_true")
    ap.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--ver-model", default="qwen3-vl-235b-a22b-instruct")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--cache-path", default="")
    ap.add_argument("--max-qchars", type=int, default=900)
    args = ap.parse_args()

    img = safe_open_image(args.crop)
    if img is None:
        raise SystemExit("failed to open crop image.")

    if args.literal:
        text = args.text
    else:
        text = Path(args.text).read_text(encoding="utf-8", errors="ignore")

    cache = None
    if args.cache:
        cache_path = Path(args.cache_path) if args.cache_path else (Path(args.crop).parent/"_cache_verify.json")
        cache = JsonCache(cache_path)

    #ver = VLMClient(api_base=args.api_base, api_key=args.api_key, model=args.ver_model)
    ver = VLMClient(
            api_base=args.api_base, api_key=args.api_key, model=args.ver_model,
            temperature=args.temperature, top_p=args.top_p
        )
    r1 = verify_question_strict(ver, img, text, cache=cache, max_chars=args.max_qchars)
    r2 = verify_question_lenient(ver, img, text, cache=cache, max_chars=args.max_qchars)

    obj = {"strict": {"verdict": r1.verdict, "chunks": r1.chunks}, "lenient": {"verdict": r2.verdict, "chunks": r2.chunks}}
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    if cache:
        cache.save()

if __name__ == "__main__":
    main()


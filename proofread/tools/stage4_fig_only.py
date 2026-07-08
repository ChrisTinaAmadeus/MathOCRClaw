from __future__ import annotations

import argparse
import json
from pathlib import Path

from proofread.cache import JsonCache
from proofread.figures import FigureFilterCfg, select_figures_for_question
from proofread.img_utils import safe_open_image
from proofread.vlm_client import VLMClient

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q-crop", required=True)
    ap.add_argument("--fig-dir", default="", help="directory containing figure_*.png")
    ap.add_argument("--figs", nargs="*", default=None, help="explicit figure image paths")
    ap.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--fig-model", default="qwen3-vl-235b-a22b-instruct")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--cache-path", default="")
    ap.add_argument("--no-cls", action="store_true")
    ap.add_argument("--no-rel", action="store_true")
    args = ap.parse_args()

    q_img = safe_open_image(args.q_crop)
    if q_img is None:
        raise SystemExit("failed to open q crop.")

    fig_paths = []
    if args.figs:
        fig_paths = [Path(p) for p in args.figs]
    elif args.fig_dir:
        d = Path(args.fig_dir)
        fig_paths = sorted(list(d.glob("figure_*.png")) + list(d.glob("figure_*.jpg")) + list(d.glob("figure_*.jpeg")))
    else:
        raise SystemExit("provide --fig-dir or --figs")

    cache = None
    if args.cache:
        cache_path = Path(args.cache_path) if args.cache_path else (Path(args.q_crop).parent/"_cache_fig.json")
        cache = JsonCache(cache_path)

    #fig_vlm = VLMClient(api_base=args.api_base, api_key=args.api_key, model=args.fig_model)
    fig_vlm = VLMClient(
        api_base=args.api_base, api_key=args.api_key, model=args.fig_model,
        temperature=args.temperature, top_p=args.top_p
    )
    cfg = FigureFilterCfg(do_vlm_cls=(not args.no_cls), do_vlm_rel=(not args.no_rel))
    kept, dbg = select_figures_for_question(q_img, fig_paths, fig_vlm, cache=cache, cfg=cfg)

    obj = {"kept": [str(p) for p in kept], "debug": dbg}
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    if cache:
        cache.save()

if __name__ == "__main__":
    main()


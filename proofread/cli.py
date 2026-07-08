from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Optional

from .cache import JsonCache
from .figures import FigureFilterCfg
from .pipeline import process_one_page
from .vlm_client import VLMClient

DEFAULT_VLM_MODEL = os.environ.get("MTC_VLM_MODEL", "qwen3.7-plus")


def _find_page_dirs(pages_root: Path, find_pattern: str) -> List[Path]:
    if find_pattern:
        hits = [Path(p) for p in glob.glob(str(pages_root / find_pattern))]
    else:
        hits = [p for p in pages_root.iterdir() if p.is_dir()]
    hits = [p for p in hits if (p / "match.json").exists()]
    return sorted(hits)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("proofread_page_v6_6")

    # input
    g = p.add_argument_group("Input")
    g.add_argument("--page-dir", type=str, default=None, help="single page directory containing match.json")
    g.add_argument("--page-md", type=str, default=None, help="whole-page markdown path for this page")
    g.add_argument("--pages-root", type=str, default=None, help="batch: root containing many page dirs")
    g.add_argument("--page-md-root", type=str, default=None, help="batch: root containing <page_dir.name>.md")
    g.add_argument("--find-pattern", type=str, default="", help="batch: glob pattern under pages-root")
    g.add_argument("--inputs", nargs="*", default=None, help="batch: specific page_dir names")

    # output
    g = p.add_argument_group("Output")
    g.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="single: output directory (default: <page_dir>/proofread_v6_6_out)",
    )
    g.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="batch: output root (default: <pages_root>/proofread_v6_6_out)",
    )

    # safety / weak evidence
    g = p.add_argument_group("Safety / Weak evidence")

    g.add_argument(
        "--weak-mask-on-weak",
        dest="weak_mask_on_weak",
        action="store_true",
        default=True,
        help="on weak evidence, if v_after!=Y then output qno + mask token (default: enabled)",
    )
    g.add_argument(
        "--no-weak-mask-on-weak",
        dest="weak_mask_on_weak",
        action="store_false",
        help="disable weak masking (keep md text on weak evidence)",
    )

    g.add_argument(
        "--weak-keep-figures",
        dest="weak_keep_figures",
        action="store_true",
        default=True,
        help="attach figures in weak evidence path (default: enabled)",
    )
    g.add_argument(
        "--no-weak-keep-figures",
        dest="weak_keep_figures",
        action="store_false",
        help="disable attaching figures in weak evidence path",
    )

    # VLM endpoints/models
    g = p.add_argument_group("VLM")
    g.add_argument("--temperature", type=float, default=0.7, help="sampling temperature for all VLM calls")
    g.add_argument("--top-p", type=float, default=0.8, help="sampling top_p for all VLM calls")
    g.add_argument("--api-base", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    g.add_argument("--api-key", type=str, default="", help="DashScope/OpenAI key (or env DASHSCOPE_API_KEY)")
    g.add_argument("--ver-model", type=str, default=DEFAULT_VLM_MODEL, help="verifier model")
    g.add_argument("--gen-model", type=str, default=DEFAULT_VLM_MODEL, help="generator (OCR) model")
    g.add_argument("--qno-model", type=str, default=DEFAULT_VLM_MODEL, help="crop-qno reader model")
    g.add_argument("--fig-model", type=str, default=DEFAULT_VLM_MODEL, help="figure filter model")

    # alignment / robustness (default: conservative)
    g = p.add_argument_group("Alignment (conservative defaults)")
    g.add_argument("--keep-partial", action="store_true", help="do NOT skip partial_question (default: skip)")
    g.add_argument("--skip-partial", action="store_true", help="force skip partial_question (default: skip anyway)")
    g.add_argument(
        "--partial-main-min-hfrac",
        type=float,
        default=0.0,
        help="if >0, keep tall partials even when skipping partial",
    )
    g.add_argument("--no-offset-search", action="store_true", help="disable offset-search fallback (default: enabled)")
    g.add_argument("--use-offset-search", action="store_true", help="alias of default; kept for CLI compatibility")
    g.add_argument("--no-crop-qno", action="store_true", help="disable crop-qno anchors (default: enabled; use with --cache)")
    g.add_argument("--use-crop-qno", action="store_true", help="alias of default; kept for CLI compatibility")

    # repair
    g = p.add_argument_group("Repair")
    g.add_argument("--max-qchars", type=int, default=900)
    g.add_argument("--mask-u-token", type=str, default="[UNREADABLE]", help="token for U (occluded/unclear)")
    g.add_argument("--mask-n-token", type=str, default="[HALLUCINATION]", help="token for N (not in image / hallucination)")

    # ablations
    g = p.add_argument_group("Ablation")
    g.add_argument(
        "--no-patcher",
        dest="ablation_no_patcher",
        action="store_true",
        default=False,
        help=(
            "Ablation: disable all text patching/repair (no regen, no tail-prune edits). "
            "Only run verifier and directly output keep/mask conclusions."
        ),
    )
    g.add_argument(
        "--verdict-comment",
        dest="verdict_comment",
        action="store_true",
        default=False,
        help="When enabled, append an HTML comment with final verdict (Y/N/U) to each question block.",
    )

    # figures
    g = p.add_argument_group("Figures")
    g.add_argument("--no-fig", action="store_true", help="disable figure attach")
    g.add_argument("--fig-min-edge", type=int, default=28)
    g.add_argument("--fig-max-aspect", type=float, default=8.0)
    g.add_argument("--fig-max-blank-frac", type=float, default=0.97)
    g.add_argument("--fig-no-cls", action="store_true", help="disable VLM printed_figure classification")
    g.add_argument("--fig-no-rel", action="store_true", help="disable VLM relevance check")

    # cache
    g = p.add_argument_group("Cache")
    g.add_argument("--cache", action="store_true", help="enable persistent cache (HIGHLY recommended)")
    g.add_argument("--cache-path", type=str, default=None, help="cache json path (default under out dir/root)")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    skip_partial = (not args.keep_partial) or bool(args.skip_partial)

    use_offset_search = bool(args.use_offset_search or (not args.no_offset_search))
    use_crop_qno = bool(args.use_crop_qno or (not args.no_crop_qno))

    ver_vlm = VLMClient(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.ver_model,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    gen_vlm = None
    if not args.ablation_no_patcher:
        gen_vlm = VLMClient(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.gen_model,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    qno_vlm = (
        VLMClient(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.qno_model,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if use_crop_qno
        else None
    )
    fig_vlm = (
        None
        if args.no_fig
        else VLMClient(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.fig_model,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    )

    fig_cfg = FigureFilterCfg(
        min_edge=args.fig_min_edge,
        max_aspect=args.fig_max_aspect,
        max_blank_frac=args.fig_max_blank_frac,
        do_vlm_cls=(not args.fig_no_cls),
        do_vlm_rel=(not args.fig_no_rel),
    )

    weak_mask_on_weak = bool(args.weak_mask_on_weak)
    weak_keep_figures = bool(args.weak_keep_figures)

    # single
    if args.page_dir:
        page_dir = Path(args.page_dir)
        if not args.page_md:
            raise SystemExit("--page-md is required in single-page mode.")
        page_md = Path(args.page_md)
        out_dir = Path(args.out_dir) if args.out_dir else (page_dir / "proofread_v6_6_out")

        cache = None
        if args.cache:
            cache_path = Path(args.cache_path) if args.cache_path else (out_dir / "_cache.json")
            cache = JsonCache(cache_path)

        process_one_page(
            page_dir,
            page_md,
            out_dir=out_dir,
            ver_vlm=ver_vlm,
            gen_vlm=gen_vlm,
            fig_vlm=fig_vlm,
            cache=cache,
            skip_partial=skip_partial,
            partial_main_min_hfrac=args.partial_main_min_hfrac,
            use_offset_search=use_offset_search,
            use_crop_qno=use_crop_qno,
            qno_vlm=qno_vlm,
            max_qchars=args.max_qchars,
            mask_u_token=args.mask_u_token,
            mask_n_token=args.mask_n_token,
            fig_cfg=fig_cfg,
            weak_mask_on_weak=weak_mask_on_weak,
            weak_keep_figures=weak_keep_figures,
            ablation_no_patcher=bool(args.ablation_no_patcher),
            verdict_comment=bool(args.verdict_comment),
        )
        return 0

    # batch
    if not args.pages_root:
        raise SystemExit("Provide either --page-dir (single) or --pages-root (batch).")
    pages_root = Path(args.pages_root)
    md_root = Path(args.page_md_root) if args.page_md_root else None
    if md_root is None:
        raise SystemExit("--page-md-root is required in batch mode.")

    out_root = Path(args.out_root) if args.out_root else (pages_root / "proofread_v6_6_out")
    out_root.mkdir(parents=True, exist_ok=True)

    cache = None
    if args.cache:
        cache_path = Path(args.cache_path) if args.cache_path else (out_root / "_cache.json")
        cache = JsonCache(cache_path)

    page_dirs = _find_page_dirs(pages_root, args.find_pattern)
    if args.inputs:
        wanted = set(args.inputs)
        page_dirs = [p for p in page_dirs if p.name in wanted]

    for pd in page_dirs:
        md_path = md_root / f"{pd.name}.md"
        if not md_path.exists():
            continue

        out_dir = out_root / pd.name
        process_one_page(
            pd,
            md_path,
            out_dir=out_dir,
            ver_vlm=ver_vlm,
            gen_vlm=gen_vlm,
            fig_vlm=fig_vlm,
            cache=cache,
            skip_partial=skip_partial,
            partial_main_min_hfrac=args.partial_main_min_hfrac,
            use_offset_search=use_offset_search,
            use_crop_qno=use_crop_qno,
            qno_vlm=qno_vlm,
            max_qchars=args.max_qchars,
            mask_u_token=args.mask_u_token,
            mask_n_token=args.mask_n_token,
            fig_cfg=fig_cfg,
            weak_mask_on_weak=weak_mask_on_weak,
            weak_keep_figures=weak_keep_figures,
            ablation_no_patcher=bool(args.ablation_no_patcher),
            verdict_comment=bool(args.verdict_comment),
        )

    if cache:
        cache.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


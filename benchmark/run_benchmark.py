from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.workflow import build_argparser as build_workflow_argparser  # noqa: E402
from agent.workflow import run_agent  # noqa: E402
from benchmark.scoring import score_files  # noqa: E402


API2_RUN_COUNT = 3


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _discover_pages(dataset_root: Path, selected: Sequence[str], limit: int) -> List[str]:
    images = {path.stem: path for path in (dataset_root / "images").glob("page*.*")}
    gold = {path.stem: path for path in (dataset_root / "baseline").glob("page*.md")}
    pages = sorted(images.keys() & gold.keys())
    if selected:
        requested = [Path(value).stem for value in selected]
        missing = [page for page in requested if page not in pages]
        if missing:
            raise FileNotFoundError(f"benchmark pages not found: {', '.join(missing)}")
        pages = requested
    return pages[:limit] if limit > 0 else pages


def _find_image(dataset_root: Path, page: str) -> Path:
    matches = sorted((dataset_root / "images").glob(f"{page}.*"))
    if not matches:
        raise FileNotFoundError(f"image not found for {page}")
    return matches[0]


def _workflow_args(cli: argparse.Namespace, image: Path, page_work_root: Path) -> argparse.Namespace:
    argv = [
        "--image",
        str(image),
        "--work-root",
        str(page_work_root),
        "--checkpoint",
        cli.checkpoint,
        "--doclayout-device",
        cli.doclayout_device,
        "--api-base",
        cli.api_base,
        "--model",
        cli.model,
        "--baseline-max-tokens",
        str(cli.baseline_max_tokens),
        "--review-max-tokens",
        str(cli.review_max_tokens),
        "--baseline-timeout",
        str(cli.baseline_timeout),
        "--review-timeout",
        str(cli.review_timeout),
        "--answer-detail-views",
        str(cli.answer_detail_views),
        "--review-runs",
        str(API2_RUN_COUNT),
    ]
    if cli.api_key:
        argv.extend(["--api-key", cli.api_key])
    if cli.skip_layout:
        argv.append("--skip-layout")
    if cli.no_cache:
        argv.append("--no-cache")
    return build_workflow_argparser().parse_args(argv)


def _aggregate_api_metrics(pages: Sequence[Dict[str, Any]], cli: argparse.Namespace) -> Dict[str, Any]:
    page_metrics = [page.get("api_metrics") or {} for page in pages if page.get("status") == "ok"]

    def total(name: str) -> Optional[int]:
        values = [metrics.get(name) for metrics in page_metrics if metrics.get(name) is not None]
        return sum(int(value) for value in values) if values else None

    prompt_tokens = total("prompt_tokens")
    completion_tokens = total("completion_tokens")
    cost = None
    pricing_configured = cli.input_price_per_million > 0 or cli.output_price_per_million > 0
    if prompt_tokens is not None and completion_tokens is not None and pricing_configured:
        cost = (
            prompt_tokens * cli.input_price_per_million
            + completion_tokens * cli.output_price_per_million
        ) / 1_000_000
    return {
        "logical_call_count": sum(int(metrics.get("logical_call_count") or 0) for metrics in page_metrics),
        "network_request_count": sum(int(metrics.get("network_request_count") or 0) for metrics in page_metrics),
        "successful_request_count": sum(
            int(metrics.get("successful_request_count") or 0) for metrics in page_metrics
        ),
        "failed_request_count": sum(int(metrics.get("failed_request_count") or 0) for metrics in page_metrics),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total("total_tokens"),
        "api_latency_s": round(sum(float(metrics.get("latency_s") or 0.0) for metrics in page_metrics), 6),
        "estimated_cost": round(cost, 8) if cost is not None else None,
        "currency": cli.currency if cost is not None else None,
    }


def _score_summary(pages: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    scores = [page[key] for page in pages if page.get("status") == "ok"]
    names = [
        "score",
        "raw_score",
        "structure_score",
        "stem_score",
        "answer_score",
        "state_macro_f1",
        "omission_rate",
        "hallucination_rate",
    ]
    return {
        name: round(sum(float(score[name]) for score in scores) / len(scores), 6)
        if scores
        else None
        for name in names
    }


def _average_score_results(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not scores:
        raise ValueError("at least one API2 score is required")

    averaged = copy.deepcopy(scores[0])
    for name in (
        "score",
        "raw_score",
        "structure_score",
        "stem_score",
        "answer_score",
        "state_macro_f1",
        "gold_question_count",
        "candidate_question_count",
        "matched_question_count",
        "omission_count",
        "omission_rate",
        "hallucination_count",
        "hallucination_rate",
    ):
        values = [score.get(name) for score in scores if score.get(name) is not None]
        averaged[name] = round(sum(float(value) for value in values) / len(values), 6)

    answer_types = set().union(
        *((score.get("answer_by_type") or {}).keys() for score in scores)
    )
    averaged["answer_by_type"] = {
        name: (
            round(sum(values) / len(values), 6)
            if (
                values := [
                    float((score.get("answer_by_type") or {}).get(name))
                    for score in scores
                    if (score.get("answer_by_type") or {}).get(name) is not None
                ]
            )
            else None
        )
        for name in sorted(answer_types)
    }

    diagnostic_names = set().union(
        *((score.get("format_diagnostics") or {}).keys() for score in scores)
    )
    averaged["format_diagnostics"] = {
        name: round(
            sum(float((score.get("format_diagnostics") or {}).get(name, 0)) for score in scores)
            / len(scores),
            6,
        )
        for name in sorted(diagnostic_names)
    }
    averaged.pop("details", None)
    averaged.pop("missing_gold_indices", None)
    averaged.pop("extra_candidate_indices", None)
    averaged["aggregation"] = {
        "method": "arithmetic_mean",
        "api2_run_count": len(scores),
    }
    return averaged


def _api2_result_paths(final_payload: Dict[str, Any], fallback: Path) -> List[Path]:
    configured = (final_payload.get("outputs") or {}).get("api2_results") or []
    paths = [Path(value).resolve() for value in configured]
    return paths or [fallback]


def _request_timed_out(request: Dict[str, Any]) -> bool:
    if request.get("success"):
        return False
    text = str(request.get("error") or "").casefold()
    return "timed out" in text or "timeout" in text


def _sum_request_elapsed_s(requests: Sequence[Dict[str, Any]]) -> Optional[float]:
    if not requests:
        return None
    return round(sum(float(request.get("elapsed_s") or 0.0) for request in requests), 6)


def _api_latency_fields(api_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract API1 elapsed and API2 mean-run elapsed from workflow api_metrics."""
    passes = ((api_metrics or {}).get("passes") or {}) if isinstance(api_metrics, dict) else {}
    call_1 = passes.get("call_1") if isinstance(passes.get("call_1"), list) else []
    call_2 = passes.get("call_2") if isinstance(passes.get("call_2"), list) else []
    call_2_runs = (
        passes.get("call_2_runs") if isinstance(passes.get("call_2_runs"), list) else []
    )

    api1_timeout = any(_request_timed_out(request) for request in call_1)
    api2_timeout = any(_request_timed_out(request) for request in call_2)
    if not api2_timeout:
        api2_timeout = any(
            _request_timed_out(request)
            for run in call_2_runs
            for request in (run if isinstance(run, list) else [])
        )

    api1_s = None if api1_timeout else _sum_request_elapsed_s(call_1)
    api2_s: Optional[float] = None
    if not api2_timeout:
        if call_2_runs:
            per_run = [
                _sum_request_elapsed_s(run if isinstance(run, list) else [])
                for run in call_2_runs
            ]
            values = [value for value in per_run if value is not None]
            api2_s = round(sum(values) / len(values), 6) if values else None
        else:
            api2_s = _sum_request_elapsed_s(call_2)

    return {
        "api1_latency_s": api1_s,
        "api2_latency_s": api2_s,
        "api1_timeout": api1_timeout,
        "api2_timeout": api2_timeout,
    }


def _format_latency_cell(
    latency_s: Optional[float],
    *,
    timed_out: bool = False,
) -> str:
    if timed_out:
        return "timeout"
    if latency_s is None:
        return "—"
    return f"{latency_s:.1f}s"


def _error_latency_cells(page: Dict[str, Any]) -> tuple[str, str]:
    fields = _api_latency_fields(page.get("api_metrics"))
    err = str(page.get("error") or "").casefold()
    error_timeout = "timed out" in err or "timeout" in err
    api1 = _format_latency_cell(
        fields["api1_latency_s"],
        timed_out=bool(fields["api1_timeout"]),
    )
    api2 = _format_latency_cell(
        fields["api2_latency_s"],
        timed_out=bool(fields["api2_timeout"]) or (error_timeout and not fields["api1_timeout"]),
    )
    if error_timeout and api1 == "—" and api2 == "—":
        api2 = "timeout"
    return api1, api2


def _render_report(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# MathOCRClaw Benchmark Report",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Model: `{report['model']}`",
        f"- API1 timeout: {report['timeouts']['baseline_s']}s",
        f"- API2 timeout: {report['timeouts']['review_s']}s",
        f"- API2 runs per page: {summary['api2_runs_per_page']}",
        f"- Pages: {summary['completed_pages']}/{summary['requested_pages']}",
        f"- Baseline (API1): **{summary['baseline']['score'] or 0:.2f}**",
        f"- Workflow (API2 mean): **{summary['workflow']['score'] or 0:.2f}**",
        f"- Gain: **{summary['gain'] or 0:+.2f}**",
        "",
        "| Page | API1 baseline | API2 mean | Gain | API1 time | API2 time | Omission Δ | Hallucination Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for page in report["pages"]:
        if page.get("status") != "ok":
            api1_time, api2_time = _error_latency_cells(page)
            lines.append(
                f"| {page['page']} | ERROR | ERROR | — | {api1_time} | {api2_time} | — | — |"
            )
            continue
        baseline = page["baseline"]
        workflow = page["workflow"]
        fields = _api_latency_fields(page.get("api_metrics"))
        api1_time = _format_latency_cell(
            page.get("api1_latency_s", fields["api1_latency_s"]),
            timed_out=bool(page.get("api1_timeout", fields["api1_timeout"])),
        )
        api2_time = _format_latency_cell(
            page.get("api2_latency_s", fields["api2_latency_s"]),
            timed_out=bool(page.get("api2_timeout", fields["api2_timeout"])),
        )
        lines.append(
            f"| {page['page']} | {baseline['score']:.2f} | {workflow['score']:.2f} | "
            f"{page['gain']:+.2f} | {api1_time} | {api2_time} | "
            f"{workflow['omission_rate'] - baseline['omission_rate']:+.3f} | "
            f"{workflow['hallucination_rate'] - baseline['hallucination_rate']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## API2 run scores",
            "",
            "| Page | Run 1 | Run 2 | Run 3 | Mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for page in report["pages"]:
        if page.get("status") != "ok":
            lines.append(f"| {page['page']} | ERROR | ERROR | ERROR | — |")
            continue
        run_scores = [run["score"] for run in page["api2_runs"]]
        displayed = [f"{score:.2f}" for score in run_scores[:API2_RUN_COUNT]]
        displayed.extend(["—"] * (API2_RUN_COUNT - len(displayed)))
        lines.append(
            f"| {page['page']} | {' | '.join(displayed)} | {page['workflow']['score']:.2f} |"
        )
    api = report["api"]
    lines.extend(
        [
            "",
            "## Format diagnostics",
            "",
            "| Page | Gold LaTeX spans | API1 LaTeX spans | API2 LaTeX spans |",
            "|---|---:|---:|---:|",
        ]
    )
    for page in report["pages"]:
        if page.get("status") != "ok":
            continue
        baseline_format = page["baseline"].get("format_diagnostics") or {}
        workflow_format = page["workflow"].get("format_diagnostics") or {}
        lines.append(
            f"| {page['page']} | {baseline_format.get('gold_latex_formula_spans', 0)} | "
            f"{baseline_format.get('candidate_latex_formula_spans', 0)} | "
            f"{workflow_format.get('candidate_latex_formula_spans', 0)} |"
        )
    lines.extend(
        [
            "",
            "## API statistics",
            "",
            f"- Logical workflow calls: {api['logical_call_count']}",
            f"- Actual network requests: {api['network_request_count']}",
            f"- Successful / failed requests: {api['successful_request_count']} / {api['failed_request_count']}",
            f"- Prompt / completion / total tokens: {api['prompt_tokens']} / {api['completion_tokens']} / {api['total_tokens']}",
            f"- API latency: {api['api_latency_s']:.2f}s",
            (
                f"- Estimated cost: {api['estimated_cost']} {api['currency']}"
                if api["estimated_cost"] is not None
                else "- Estimated cost: not configured"
            ),
            "",
            "## Evaluator",
            "",
            "The default scorer is deterministic and local: question alignment, CER/bigram/critical-span text scoring, normalized LaTeX soft alignment, answer-type scoring, state Macro-F1, and hallucination penalty. It makes no additional judge-model calls.",
        ]
    )
    failures = [page for page in report["pages"] if page.get("status") != "ok"]
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- `{page['page']}`: {page.get('error')}" for page in failures)
    return "\n".join(lines).strip() + "\n"


def run(cli: argparse.Namespace) -> Dict[str, Any]:
    dataset_root = Path(cli.dataset_root).resolve()
    output_dir = Path(cli.output_dir).resolve()
    work_root = Path(cli.work_root).resolve() if cli.work_root else output_dir / "workflow"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    pages = _discover_pages(dataset_root, cli.pages, cli.limit)
    started = time.perf_counter()
    page_reports: List[Dict[str, Any]] = []

    for ordinal, page in enumerate(pages, start=1):
        print(f"[{ordinal}/{len(pages)}] {page}", flush=True)
        image = _find_image(dataset_root, page)
        gold_path = dataset_root / "baseline" / f"{page}.md"
        page_work_root = work_root / page
        first_pass_path = page_work_root / "api_markdown" / f"{page}.json"
        final_path = page_work_root / "agent_outputs" / "result.json"
        page_started = time.perf_counter()
        try:
            final_payload: Dict[str, Any]
            if cli.score_only:
                final_payload = json.loads(final_path.read_text(encoding="utf-8"))
            elif cli.resume and first_pass_path.exists() and final_path.exists():
                resumed_payload = json.loads(final_path.read_text(encoding="utf-8"))
                resumed_paths = _api2_result_paths(resumed_payload, final_path)
                if len(resumed_paths) == API2_RUN_COUNT and all(
                    path.exists() for path in resumed_paths
                ):
                    final_payload = resumed_payload
                else:
                    final_payload = run_agent(_workflow_args(cli, image, work_root))
            else:
                final_payload = run_agent(_workflow_args(cli, image, work_root))
            api2_paths = _api2_result_paths(final_payload, final_path)
            if not cli.score_only and len(api2_paths) != API2_RUN_COUNT:
                raise RuntimeError(
                    f"expected {API2_RUN_COUNT} API2 results for {page}, got {len(api2_paths)}"
                )
            scored_runs = [
                score_files(gold_path, first_pass_path, api2_path) for api2_path in api2_paths
            ]
            baseline_score = scored_runs[0]["baseline"]
            workflow_runs = [scored["workflow"] for scored in scored_runs]
            workflow_score = _average_score_results(workflow_runs)
            page_report = {
                "page": page,
                "status": "ok",
                "elapsed_s": round(time.perf_counter() - page_started, 6),
                "gold": str(gold_path),
                "first_pass": str(first_pass_path),
                "final_result": str(final_path),
                "final_results": [str(path) for path in api2_paths],
                "baseline": baseline_score,
                "workflow": workflow_score,
                "api2_runs": [
                    {
                        "run": run_index,
                        "result": str(api2_path),
                        "score": scored["workflow"]["score"],
                        "metrics": scored["workflow"],
                    }
                    for run_index, (api2_path, scored) in enumerate(
                        zip(api2_paths, scored_runs), start=1
                    )
                ],
                "gain": round(workflow_score["score"] - baseline_score["score"], 6),
                "evaluator": scored_runs[0]["evaluator"],
                "api_metrics": final_payload.get("api_metrics") or {},
            }
            page_report.update(_api_latency_fields(page_report.get("api_metrics")))
        except Exception as exc:
            prior_path = output_dir / "pages" / f"{page}.json"
            prior_error = ""
            if prior_path.exists():
                try:
                    prior_payload = json.loads(prior_path.read_text(encoding="utf-8"))
                    if prior_payload.get("status") == "error":
                        prior_error = str(prior_payload.get("error") or "")
                except Exception:
                    prior_error = ""
            error_text = f"{type(exc).__name__}: {exc}"
            # Keep a prior timeout/error message when score-only cannot load missing outputs.
            if (
                cli.score_only
                and prior_error
                and (
                    "timed out" in prior_error.casefold()
                    or "timeout" in prior_error.casefold()
                    or isinstance(exc, FileNotFoundError)
                )
            ):
                error_text = prior_error
            page_report = {
                "page": page,
                "status": "error",
                "elapsed_s": round(time.perf_counter() - page_started, 6),
                "error": error_text,
            }
            if cli.fail_fast:
                page_reports.append(page_report)
                _json_dump(output_dir / "report.partial.json", {"pages": page_reports})
                raise
        page_reports.append(page_report)
        _json_dump(output_dir / "pages" / f"{page}.json", page_report)

    baseline = _score_summary(page_reports, "baseline")
    workflow = _score_summary(page_reports, "workflow")
    completed = sum(page.get("status") == "ok" for page in page_reports)
    api2_run_counts = sorted(
        {len(page["api2_runs"]) for page in page_reports if page.get("status") == "ok"}
    )
    api2_runs_per_page: Any = (
        api2_run_counts[0] if len(api2_run_counts) == 1 else api2_run_counts
    )
    report = {
        "schema_version": 2,
        "run_id": cli.run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "model": cli.model,
        "dataset_root": str(dataset_root),
        "work_root": str(work_root),
        "timeouts": {
            "baseline_s": cli.baseline_timeout,
            "review_s": cli.review_timeout,
        },
        "evaluator": "mathocrclaw_deterministic_v3_local",
        "summary": {
            "requested_pages": len(pages),
            "completed_pages": completed,
            "failed_pages": len(pages) - completed,
            "api2_runs_per_page": api2_runs_per_page,
            "baseline": baseline,
            "workflow": workflow,
            "gain": round(float(workflow["score"] or 0) - float(baseline["score"] or 0), 6),
            "elapsed_s": round(time.perf_counter() - started, 6),
        },
        "api": _aggregate_api_metrics(page_reports, cli),
        "pages": page_reports,
    }
    _json_dump(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(_render_report(report), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_model = os.environ.get("MTC_VLM_MODEL", "qwen3.7-plus")
    parser = argparse.ArgumentParser(
        "mathocrclaw_benchmark",
        description="Score API1 draft OCR and API2 workflow output against gold Markdown.",
    )
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "benchmark"))
    parser.add_argument("--pages", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-id", default=timestamp)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "benchmark" / "runs" / timestamp))
    parser.add_argument("--work-root", default="")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-layout", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoint_best_total.pth")
    parser.add_argument("--doclayout-device", default="cpu")
    parser.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--baseline-max-tokens", type=int, default=5000)
    parser.add_argument("--review-max-tokens", type=int, default=7000)
    parser.add_argument(
        "--baseline-timeout",
        type=int,
        default=360,
        help="API1 upload/read timeout in seconds",
    )
    parser.add_argument(
        "--review-timeout",
        type=int,
        default=360,
        help="API2 upload/read timeout in seconds",
    )
    parser.add_argument("--answer-detail-views", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--input-price-per-million", type=float, default=0.0)
    parser.add_argument("--output-price-per-million", type=float, default=0.0)
    parser.add_argument("--currency", default="CNY")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    cli = build_argparser().parse_args(argv)
    report = run(cli)
    summary = report["summary"]
    print(
        f"API1 baseline={summary['baseline']['score'] or 0:.2f} "
        f"API2 workflow mean={summary['workflow']['score'] or 0:.2f} "
        f"gain={summary['gain'] or 0:+.2f}"
    )
    print(Path(cli.output_dir).resolve() / "report.md")
    return 0 if summary["failed_pages"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line interface (design doc 3.2, 11).

Exit codes:
    0  review completed (findings may exist; completion is not a pass/fail verdict)
    1  unexpected internal error
    2  invalid input (bad path, bad config)
    3  review failed (LLM failure, missing required tool, budget stop, no coverage)
    4  report could not be written
    130 interrupted
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from . import __version__
from .config.settings import AgentConfig, load_config
from .errors import CodeReviewError, InputError
from .workflow.orchestrator import (
    EXIT_INPUT_ERROR,
    EXIT_INTERRUPTED,
    EXIT_OK,
    ReviewWorkflow,
)

logger = logging.getLogger("code_review_agent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="code-review",
        description=(
            "Evidence-based AI code review agent. Scans a project, runs static analysis, "
            "asks an LLM for evidence-backed findings, validates every finding, and writes "
            "Markdown + JSON reports. The reviewed project is never modified."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  code-review ./my-react-app\n"
            "  code-review ./app --provider fake\n"
            "  code-review ./app --model gpt-4o-mini --max-files 20\n"
            "  code-review ./app --config ./code-review.toml --output-dir /tmp/review\n"
        ),
    )
    parser.add_argument("project", help="path to the project directory to review")
    parser.add_argument("--config", metavar="PATH", help="path to a TOML config file")
    parser.add_argument("--output-dir", metavar="PATH", help="where to write the reports")
    parser.add_argument(
        "--provider",
        choices=["openai", "openai-compatible", "deepseek", "anthropic", "fake"],
        help="LLM provider; 'fake' runs fully offline with no model calls",
    )
    parser.add_argument("--model", help="model name to use")
    parser.add_argument("--base-url", help="override the provider API base URL")
    parser.add_argument("--api-key-env", help="environment variable holding the API key")
    parser.add_argument("--max-files", type=int, help="maximum files to review")
    parser.add_argument("--max-requests", type=int, help="maximum LLM requests")
    parser.add_argument("--max-tokens", type=int, help="maximum total tokens")
    parser.add_argument("--max-cost-usd", type=float, help="maximum estimated model cost")
    parser.add_argument(
        "--eslint-command",
        metavar="CMD",
        help="pin the ESLint executable, or 'auto' to probe, or 'off' to disable ESLint",
    )
    parser.add_argument(
        "--skip-static-analysis",
        action="store_true",
        help="disable the ESLint phase (built-in rules still run)",
    )
    parser.add_argument(
        "--fail-on-missing-tool",
        action="store_true",
        help="treat a missing static-analysis tool as fatal instead of skipping it",
    )
    parser.add_argument(
        "--no-tool-calls",
        action="store_true",
        help="disable the agent's read-only tool-use phase",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="stdout summary format (reports are always written to disk)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("-q", "--quiet", action="store_true", help="errors only")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def apply_overrides(config: AgentConfig, args: argparse.Namespace) -> AgentConfig:
    """Fold CLI flags into the loaded config (CLI wins over file and env)."""
    llm = config.llm
    budget = config.budget
    static = config.static_analysis
    review = config.review
    report = config.report

    if args.provider:
        llm = replace(llm, provider=args.provider)
    if args.model:
        llm = replace(llm, model=args.model)
    if args.base_url:
        llm = replace(llm, base_url=args.base_url)
    if args.api_key_env:
        llm = replace(llm, api_key_env=args.api_key_env)

    if args.max_files is not None:
        budget = replace(budget, max_files=args.max_files)
    if args.max_requests is not None:
        budget = replace(budget, max_requests=args.max_requests)
    if args.max_tokens is not None:
        budget = replace(budget, max_total_tokens=args.max_tokens)
    if args.max_cost_usd is not None:
        budget = replace(budget, max_cost_usd=args.max_cost_usd)

    if args.eslint_command:
        if args.eslint_command == "off":
            static = replace(static, enabled=False)
        else:
            static = replace(static, eslint_command=args.eslint_command)
    if args.skip_static_analysis:
        static = replace(static, enabled=False)
    if args.fail_on_missing_tool:
        static = replace(static, fail_on_missing_tool=True)

    if args.no_tool_calls:
        review = replace(review, enable_tool_calls=False)

    if args.output_dir:
        report = replace(report, output_dir=args.output_dir)

    log_level = "DEBUG" if args.verbose else ("ERROR" if args.quiet else config.log_level)
    return replace(
        config,
        llm=llm,
        budget=budget,
        static_analysis=static,
        review=review,
        report=report,
        log_level=log_level,
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def render_text_summary(result, *, stream=None) -> None:
    # Resolve sys.stdout *inside* the call: a default of sys.stdout binds at
    # import time, which silently defeats redirection and output capture.
    stream = stream if stream is not None else sys.stdout
    run = result.run
    findings = result.report.findings if result.report is not None else []
    counts = result.report.severity_counts if result.report is not None else {}

    lines: list[str] = []
    if run.status.value == "completed":
        lines.append("Review completed.")
    else:
        lines.append(f"Review did not complete (status: {run.status.value}).")
    lines.append("")
    lines.append(f"Files scanned: {run.files_scanned}")
    lines.append(f"Files reviewed: {run.files_reviewed}")
    lines.append(f"Findings: {run.findings_count}")
    for severity in ("critical", "high", "medium", "low", "info"):
        lines.append(f"{severity.capitalize()}: {counts.get(severity, 0)}")

    if findings:
        static_only = [f for f in findings if f.source.value == "static_analysis"]
        if static_only:
            lines.append("")
            lines.append(
                f"Source: {len(static_only)} from built-in static analysis, "
                f"{len(findings) - len(static_only)} from model review."
            )
        if run.files_reviewed == 0:
            # Without this, "0 files reviewed / N findings" reads as a
            # contradiction. It is not: lint results are real evidence.
            lines.append(
                "Note: no file received an AI review; the findings above come only from "
                "deterministic static analysis."
            )

    lines.append("")
    if run.skipped_analyzers:
        names = ", ".join(name for name, _ in run.skipped_analyzers)
        lines.append(f"Static analysis skipped: {names}")
    if result.failure_reason:
        lines.append(f"Reason: {result.failure_reason}")
    if result.markdown_path:
        lines.append(f"Report: {result.markdown_path}")
    if result.json_path:
        lines.append(f"Report (JSON): {result.json_path}")
    if run.status.value != "completed":
        lines.append("")
        lines.append(
            "Note: an incomplete review is not a passing review. See the report for details."
        )
    stream.write("\n".join(lines) + "\n")


def render_json_summary(result, *, stream=None) -> None:
    stream = stream if stream is not None else sys.stdout
    run = result.run
    counts = result.report.severity_counts if result.report is not None else {}
    payload = {
        "status": run.status.value,
        "run_id": run.run_id,
        "files_scanned": run.files_scanned,
        "files_reviewed": run.files_reviewed,
        "findings_count": run.findings_count,
        "severity_counts": counts,
        "exit_code": result.exit_code,
        "failure_reason": result.failure_reason,
        "markdown_report": str(result.markdown_path) if result.markdown_path else None,
        "json_report": str(result.json_path) if result.json_path else None,
        "skipped_analyzers": [name for name, _ in run.skipped_analyzers],
        "errors": [error.to_dict() for error in run.errors],
        "notes": run.notes,
    }
    stream.write(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        config = apply_overrides(config, args)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    configure_logging(config.log_level)

    project_root = Path(args.project).expanduser()
    if not project_root.exists():
        print(f"error: project directory does not exist: {project_root}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if not project_root.is_dir():
        print(f"error: not a directory: {project_root}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    workflow = ReviewWorkflow(config, output_dir=output_dir)

    try:
        result = workflow.execute(project_root)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except CodeReviewError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except Exception as exc:  # noqa: BLE001 - last-resort guard, never a silent success
        logger.exception("unexpected failure")
        print(f"error: unexpected internal failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        render_json_summary(result)
    else:
        render_text_summary(result)

    if result.exit_code != EXIT_OK:
        # Make the failure visible on stderr as well as in the summary.
        print(
            f"error: review did not complete successfully ({result.failure_reason or 'unknown reason'})",
            file=sys.stderr,
        )
    return result.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

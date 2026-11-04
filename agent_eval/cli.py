"""Small CLI with machine-readable artifacts and CI-friendly exit codes."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .adapters import ChatAdapter, DemoAdapter, SYSTEM_PROMPT
from .compare import compare_runs
from .dataset import load_dataset
from .report import render_report, write_artifacts
from .runner import RunConfig, run_suite
from .tools import ToolBox

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"


def provenance(knowledge: Path, extra: dict | None = None) -> dict:
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=3, cwd=ROOT).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        revision = None
    return {"git_commit": revision, "knowledge_sha256": hashlib.sha256(knowledge.read_bytes()).hexdigest(), **(extra or {})}


async def execute(args) -> int:
    if args.command == "compare":
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
        comparison = compare_runs(baseline, candidate, args.min_pass_rate, args.max_drop)
        write_artifacts(candidate, args.out, comparison)
        print(json.dumps(comparison, indent=2))
        return 0 if comparison["gate_passed"] else 1
    cases, fingerprint = load_dataset(args.dataset)
    knowledge = Path(args.knowledge)
    tools = ToolBox(knowledge)
    config = RunConfig(args.concurrency, args.timeout, args.retries, args.repetitions)
    if args.command == "demo":
        baseline = await run_suite(cases, fingerprint, DemoAdapter(tools, "baseline"), config, provenance(knowledge, {"mode": "scripted_demo"}))
        candidate = await run_suite(cases, fingerprint, DemoAdapter(tools), config, provenance(knowledge, {"mode": "scripted_demo"}))
        comparison = compare_runs(baseline, candidate)
        output = Path(args.out)
        write_artifacts(baseline, output / "baseline")
        write_artifacts(candidate, output / "candidate", comparison)
        (output / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        (output / "report.html").write_text(render_report(candidate, comparison), encoding="utf-8")
        print(f"SCRIPTED DEMO (no model inference): baseline {baseline['summary']['pass_rate']:.1%} -> candidate {candidate['summary']['pass_rate']:.1%}")
        print(f"Gate: {'PASS' if comparison['gate_passed'] else 'FAIL'} | report: {output.resolve() / 'report.html'}")
        return 0 if comparison["gate_passed"] else 1
    if not args.base_url or not args.model:
        raise ValueError("run requires --base-url and --model (or EVAL_BASE_URL / EVAL_MODEL)")
    adapter = ChatAdapter(tools, args.base_url, args.model, os.environ.get("EVAL_API_KEY", ""), args.timeout,
                          args.max_steps, args.max_tool_calls, args.input_price, args.output_price)
    if not 0 <= args.min_pass_rate <= 1:
        raise ValueError("min-pass-rate must be in [0,1]")
    extra = {"mode": "live", "base_url": args.base_url, "model": args.model,
             "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(), "temperature": 0,
             "max_steps": args.max_steps, "max_tool_calls": args.max_tool_calls,
             "input_price_per_million": args.input_price, "output_price_per_million": args.output_price}
    run = await run_suite(cases, fingerprint, adapter, config, provenance(knowledge, extra))
    write_artifacts(run, args.out)
    summary = run["summary"]
    safety_failed = any("safety" in r["tags"] and not r["passed"] for r in run["results"])
    passed = summary["pass_rate"] >= args.min_pass_rate and summary["errors"] == 0 and not safety_failed
    print(f"{adapter.name}: {summary['passed']}/{summary['count']} passed; {summary['errors']} execution errors; gate {'PASS' if passed else 'FAIL'}")
    print(f"Report: {Path(args.out).resolve() / 'report.html'}")
    return 0 if passed else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Trace-aware LLM and agent evaluation")
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("demo", "run"):
        p = commands.add_parser(command, help="Offline scripted demonstration" if command == "demo" else "Evaluate a live chat-completions endpoint")
        p.add_argument("--dataset", default=str(DATA / "support_v1.jsonl"))
        p.add_argument("--knowledge", default=str(DATA / "knowledge.json"))
        p.add_argument("--out", default=f"runs/{command}")
        p.add_argument("--concurrency", type=int, default=4)
        p.add_argument("--timeout", type=float, default=30)
        p.add_argument("--retries", type=int, default=0)
        p.add_argument("--repetitions", type=int, default=1)
        if command == "run":
            p.add_argument("--base-url", default=os.environ.get("EVAL_BASE_URL"))
            p.add_argument("--model", default=os.environ.get("EVAL_MODEL"))
            p.add_argument("--max-steps", type=int, default=6)
            p.add_argument("--max-tool-calls", type=int, default=12)
            p.add_argument("--input-price", type=float, help="USD per million input tokens; set both prices for cost checks")
            p.add_argument("--output-price", type=float, help="USD per million output tokens")
            p.add_argument("--min-pass-rate", type=float, default=.9)
    p = commands.add_parser("compare", help="Compare runs on identical evaluation contracts")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", default="runs/comparison")
    p.add_argument("--min-pass-rate", type=float, default=.9)
    p.add_argument("--max-drop", type=float, default=0)
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        return asyncio.run(execute(args))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        # CLI input errors are local; provider/adapter error messages never reach here.
        print(f"Configuration/artifact error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

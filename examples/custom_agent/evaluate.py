"""Run from a clone: python -m examples.custom_agent --out runs/custom."""

import argparse
import asyncio
import hashlib
from pathlib import Path
import sys

from agent_eval.dataset import load_dataset
from agent_eval.report import write_artifacts
from agent_eval.runner import RunConfig, run_suite
from examples.custom_agent.adapter import CustomAgentAdapter

EXAMPLE_DIR = Path(__file__).resolve().parent


async def evaluate(args) -> int:
    if not 0 <= args.min_pass_rate <= 1:
        raise ValueError("min-pass-rate must be in [0,1]")
    config = RunConfig(concurrency=args.concurrency, timeout_s=args.timeout, repetitions=args.repetitions)
    cases, fingerprint = load_dataset(args.dataset)
    adapter = CustomAgentAdapter()
    metadata = {
        "mode": "scripted_demo",
        "example": "custom_order_agent",
        "agent_sha256": hashlib.sha256((EXAMPLE_DIR / "agent.py").read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256((EXAMPLE_DIR / "adapter.py").read_bytes()).hexdigest(),
    }
    run = await run_suite(cases, fingerprint, adapter, config, provenance=metadata)
    write_artifacts(run, args.out)
    summary = run["summary"]
    safety_failed = any("safety" in row["tags"] and not row["passed"] for row in run["results"])
    passed = summary["pass_rate"] >= args.min_pass_rate and summary["errors"] == 0 and not safety_failed
    print(f"SCRIPTED CUSTOM EXAMPLE (no model inference): {summary['passed']}/{summary['count']} passed; gate {'PASS' if passed else 'FAIL'}")
    print(f"Report: {Path(args.out).resolve() / 'report.html'}")
    return 0 if passed else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate an independent application through a custom adapter")
    parser.add_argument("--dataset", type=Path, default=EXAMPLE_DIR / "cases.jsonl")
    parser.add_argument("--out", type=Path, default=Path("runs/custom"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--min-pass-rate", type=float, default=.9)
    args = parser.parse_args(argv)
    try:
        return asyncio.run(evaluate(args))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Configuration/artifact error: {type(exc).__name__}", file=sys.stderr)
        return 2

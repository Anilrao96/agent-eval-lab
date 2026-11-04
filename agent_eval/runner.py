"""Bounded execution, failure isolation and provenance-aware run artifacts."""

import asyncio
import hashlib
import json
import math
import platform
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from . import __version__
from .metrics import evaluate
from .models import Adapter, AgentInput, Case, CaseResult, TransientError


@dataclass(frozen=True)
class RunConfig:
    concurrency: int = 4
    timeout_s: float = 30
    retries: int = 0
    repetitions: int = 1

    def __post_init__(self):
        if type(self.concurrency) is not int or not 1 <= self.concurrency <= 64:
            raise ValueError("concurrency must be in [1,64]")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout must be positive and finite")
        if type(self.retries) is not int or not 0 <= self.retries <= 5:
            raise ValueError("retries must be in [0,5]")
        if type(self.repetitions) is not int or not 1 <= self.repetitions <= 100:
            raise ValueError("repetitions must be in [1,100]")


def summarize(results: list[dict]) -> dict:
    def group(rows):
        return {"count": len(rows), "passed": sum(r["passed"] for r in rows),
                "pass_rate": sum(r["passed"] for r in rows) / len(rows) if rows else 0}
    latency = sorted(r["latency_ms"] for r in results)
    tags = sorted({t for r in results for t in r["tags"]})
    known_costs = [r["prediction"]["cost_usd"] for r in results if r["prediction"] and r["prediction"]["cost_usd"] is not None]
    names = sorted({m["name"] for r in results for m in r["metrics"]})
    return {**group(results), "errors": sum(r["status"] != "ok" for r in results),
            "latency_p50_ms": latency[(len(latency) - 1) // 2] if latency else 0,
            "latency_p95_ms": latency[max(0, math.ceil(len(latency) * .95) - 1)] if latency else 0,
            "known_success_cost_usd": sum(known_costs) if known_costs else None,
            "cost_coverage": len(known_costs) / len(results) if results else 0,
            "slices": {t: group([r for r in results if t in r["tags"]]) for t in tags},
            "metrics": {n: {"count": len(ms := [m for r in results for m in r["metrics"] if m["name"] == n]),
                             "mean_score": sum(m["score"] for m in ms) / len(ms),
                             "pass_rate": sum(m["passed"] for m in ms) / len(ms)} for n in names}}


async def run_suite(cases: list[Case], fingerprint: str, adapter: Adapter, config: RunConfig,
                    provenance: dict | None = None) -> dict:
    semaphore = asyncio.Semaphore(config.concurrency)

    async def one(case, repetition):
        async with semaphore:
            started = time.perf_counter()
            prediction, status, error, attempts = None, "error", None, 0
            deadline = time.monotonic() + config.timeout_s
            for attempt in range(config.retries + 1):
                attempts += 1
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError()
                    prediction = await asyncio.wait_for(adapter.predict(AgentInput(case.prompt)), timeout=remaining)
                    status = "ok"
                    break
                except (asyncio.TimeoutError, TimeoutError):
                    status, error = "timeout", "TimeoutError"
                    break
                except TransientError:
                    error = "TransientError"
                    if attempt < config.retries:
                        delay = min(.1 * (2 ** attempt), max(0, deadline - time.monotonic()))
                        await asyncio.sleep(delay)
                except Exception as exc:
                    # Exception messages can contain credentials or user data. Persist only types.
                    error = type(exc).__name__
                    break
            latency = round((time.perf_counter() - started) * 1000, 3)
            metrics = []
            if prediction is not None:
                if attempts > 1:
                    # Earlier failed attempts may have been billed; total usage is unknown.
                    prediction.cost_usd = prediction.input_tokens = prediction.output_tokens = None
                try:
                    metrics = evaluate(case, prediction, latency)
                except Exception as exc:
                    status, error = "evaluator_error", type(exc).__name__
            if status == "ok":
                error = None
            return CaseResult(case.id, repetition, case.tags, status, latency, attempts, prediction, metrics, error).to_dict()

    results = await asyncio.gather(*(one(case, repetition) for case in cases for repetition in range(config.repetitions)))
    contract = [{"id": c.id, "tags": c.tags, "checks": c.checks} for c in cases]
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return {"schema_version": 1, "run_id": uuid.uuid4().hex[:12], "created_at": datetime.now(timezone.utc).isoformat(),
            "adapter": adapter.name, "framework_version": __version__, "python": platform.python_version(),
            "dataset_sha256": fingerprint, "evaluation_sha256": contract_hash, "config": asdict(config),
            "provenance": provenance or {}, "summary": summarize(results), "results": results}

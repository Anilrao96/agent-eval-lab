"""Paired regression gates; bootstrap resamples cases, not repeated draws."""

import random
from collections import defaultdict


def compare_runs(baseline: dict, candidate: dict, min_pass_rate: float = .9, max_drop: float = 0,
                 bootstrap_samples: int = 2000, seed: int = 17) -> dict:
    if not 0 <= min_pass_rate <= 1 or not 0 <= max_drop <= 1 or bootstrap_samples < 100:
        raise ValueError("invalid comparison thresholds")
    for field in ("schema_version", "dataset_sha256", "evaluation_sha256"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"incomparable runs: different {field}")
    if baseline["config"]["repetitions"] != candidate["config"]["repetitions"]:
        raise ValueError("incomparable runs: repetition count differs")

    def index(run):
        rows = {(r["case_id"], r["repetition"]): r for r in run["results"]}
        if len(rows) != len(run["results"]) or not rows:
            raise ValueError("empty run or duplicate case/repetition keys")
        return rows

    before, after = index(baseline), index(candidate)
    if before.keys() != after.keys():
        raise ValueError("incomparable runs: case/repetition keys differ")
    improvements, regressions, per_case = [], [], defaultdict(list)
    for key, b in before.items():
        c = after[key]
        if b["tags"] != c["tags"]:
            raise ValueError("incomparable runs: case tags differ")
        delta = int(c["passed"]) - int(b["passed"])
        per_case[key[0]].append(delta)
        if delta:
            (improvements if delta > 0 else regressions).append({"case_id": key[0], "repetition": key[1]})
    deltas = [sum(values) / len(values) for values in per_case.values()]
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(bootstrap_samples))
    baseline_rate = sum(r["passed"] for r in before.values()) / len(before)
    candidate_rate = sum(r["passed"] for r in after.values()) / len(after)
    delta = candidate_rate - baseline_rate
    reasons = []
    if candidate_rate < min_pass_rate:
        reasons.append("candidate_below_minimum_pass_rate")
    if delta < -max_drop - 1e-12:
        reasons.append("aggregate_pass_rate_regression")
    if regressions:
        reasons.append("new_case_failures")
    if any(r["status"] != "ok" for r in after.values()):
        reasons.append("candidate_execution_errors")
    if any("safety" in r["tags"] and not r["passed"] for r in after.values()):
        reasons.append("candidate_safety_failure")
    return {"schema_version": 1, "baseline_run_id": baseline["run_id"], "candidate_run_id": candidate["run_id"],
            "baseline_pass_rate": baseline_rate, "candidate_pass_rate": candidate_rate, "delta": delta,
            "delta_ci95": [samples[int(.025 * (len(samples) - 1))], samples[int(.975 * (len(samples) - 1))]],
            "bootstrap": {"unit": "case", "samples": bootstrap_samples, "seed": seed, "cases": len(deltas)},
            "gate_passed": not reasons, "gate_reasons": reasons,
            "thresholds": {"min_pass_rate": min_pass_rate, "max_drop": max_drop, "new_case_failures_allowed": False},
            "improvements": improvements, "regressions": regressions,
            "limitations": "Synthetic/small curated sets do not estimate production performance; percentile intervals describe this case mix only."}

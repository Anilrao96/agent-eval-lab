"""Deterministic evaluators; all required checks must pass (no safety averaging)."""

import json
import math
import re
from collections import Counter

from .models import Case, Metric, Prediction


def normalize(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def token_f1(answer: str, reference: str) -> float:
    a, b = Counter(normalize(answer).split()), Counter(normalize(reference).split())
    if not a or not b:
        return float(a == b)
    overlap = sum((a & b).values())
    return 2 * overlap / (sum(a.values()) + sum(b.values()))


def _json_matches(answer: str, contract: dict) -> bool:
    try:
        value = json.loads(answer, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, TypeError):
        return False
    if not isinstance(value, dict):
        return False
    types = {"string": (str,), "number": (int, float), "integer": (int,), "boolean": (bool,), "array": (list,), "object": (dict,), "null": (type(None),)}
    for key, expected in contract["types"].items():
        if key not in value or type(value[key]) not in types[expected]:
            return False
        if expected == "number" and not math.isfinite(value[key]):
            return False
    if not contract.get("allow_extra", False) and set(value) != set(contract["types"]):
        return False
    return all(value.get(k) == v for k, v in contract.get("values", {}).items())


def evaluate(case: Case, prediction: Prediction, latency_ms: float) -> list[Metric]:
    checks, answer, metrics = case.checks, prediction.answer, []

    def add(name: str, passed: bool, reason: str):
        metrics.append(Metric(name, float(passed), 1.0, reason))

    for name, spec in checks.items():
        if name == "exact":
            add(name, normalize(answer) == normalize(spec), "Normalized answer equality")
        elif name == "numeric":
            try:
                actual = float(answer.strip())
                passed = math.isfinite(actual) and math.isclose(actual, spec["value"], rel_tol=0, abs_tol=spec["abs_tol"])
            except (ValueError, TypeError):
                passed = False
            add(name, passed, f"Numeric answer within absolute tolerance {spec['abs_tol']}")
        elif name == "contains_all":
            add(name, all(s.casefold() in answer.casefold() for s in spec), "All required answer fragments")
        elif name == "contains_any":
            add(name, any(s.casefold() in answer.casefold() for s in spec), "At least one accepted answer fragment")
        elif name == "contains_none":
            add(name, all(s.casefold() not in answer.casefold() for s in spec), "No disallowed answer fragments")
        elif name == "token_f1":
            metrics.append(Metric(name, token_f1(answer, spec["reference"]), spec["threshold"], "Multiset token overlap; lexical proxy only"))
        elif name == "json_contract":
            add(name, _json_matches(answer, spec), "Strict JSON object, types, values and extra-key policy")
        elif name == "tool_path":
            cursor = 0
            for event in prediction.trace:
                target = spec[cursor]
                if event.error is None and event.name == target["name"] and all(event.arguments.get(k) == v for k, v in target.get("arguments", {}).items()):
                    cursor += 1
                    if cursor == len(spec):
                        break
            add(name, cursor == len(spec), f"Successful ordered subsequence {cursor}/{len(spec)}; arguments matched")
        elif name == "forbidden_tools":
            violations = sorted({e.name for e in prediction.trace} & set(spec))
            add(name, not violations, "Forbidden attempts: " + (", ".join(violations) or "none"))
        elif name == "citations":
            cited = set(re.findall(r"\[policy:([\w-]+)\]", answer))
            retrieved = set()
            for e in prediction.trace:
                if e.name == "search_policy" and e.error is None and isinstance(e.output, dict):
                    retrieved.update(d["id"] for d in e.output.get("documents", []) if isinstance(d, dict) and isinstance(d.get("id"), str))
            add(name, set(spec) <= cited and cited <= retrieved, "Required citations present; every cited ID was retrieved (not semantic entailment)")
        elif name.startswith("max_"):
            actual = {"max_latency_ms": latency_ms, "max_cost_usd": prediction.cost_usd,
                      "max_tokens": None if prediction.input_tokens is None or prediction.output_tokens is None else prediction.input_tokens + prediction.output_tokens,
                      "max_tool_calls": len(prediction.trace)}[name]
            known = actual is not None and math.isfinite(actual) and actual >= 0
            add(name, known and actual <= spec, f"Observed {actual!r}; limit {spec}; unknown usage fails closed")
    return metrics

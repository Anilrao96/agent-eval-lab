"""Strict, versioned JSONL loading and stable content fingerprints."""

import hashlib
import json
import math
from pathlib import Path

from .models import Case

SUPPORTED = {
    "exact", "numeric", "contains_all", "contains_any", "contains_none", "token_f1", "json_contract",
    "tool_path", "forbidden_tools", "citations", "max_latency_ms",
    "max_cost_usd", "max_tokens", "max_tool_calls",
}


def load_dataset(path: str | Path) -> tuple[list[Case], str]:
    cases, seen, canonical = [], set(), []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if set(row) != {"id", "prompt", "tags", "checks"}:
                raise ValueError("expected exactly id, prompt, tags, checks")
            if not isinstance(row["id"], str) or not row["id"].strip() or row["id"] in seen:
                raise ValueError("case IDs must be non-empty and unique")
            if not isinstance(row["prompt"], str) or not row["prompt"].strip():
                raise ValueError("prompt must be non-empty text")
            if not isinstance(row["tags"], list) or not row["tags"] or not all(isinstance(t, str) and t for t in row["tags"]):
                raise ValueError("tags must be a non-empty string list")
            checks = row["checks"]
            if not isinstance(checks, dict) or not checks or set(checks) - SUPPORTED:
                raise ValueError("empty checks or unknown check names")
            for key, value in checks.items():
                if key in {"max_latency_ms", "max_cost_usd", "max_tokens", "max_tool_calls"}:
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"{key} must be finite and non-negative")
                elif key == "exact" and not isinstance(value, str):
                    raise ValueError("exact must be text")
                elif key == "numeric":
                    if not isinstance(value, dict) or set(value) != {"value", "abs_tol"} or any(type(value[k]) not in (int, float) or not math.isfinite(value[k]) for k in value) or value["abs_tol"] < 0:
                        raise ValueError("numeric requires finite value and non-negative abs_tol")
                elif key in {"contains_all", "contains_any", "contains_none", "forbidden_tools", "citations"}:
                    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
                        raise ValueError(f"{key} must be a non-empty string list")
                elif key == "tool_path":
                    if not isinstance(value, list) or not value:
                        raise ValueError("tool_path must be non-empty")
                    for step in value:
                        if not isinstance(step, dict) or set(step) - {"name", "arguments"} or not isinstance(step.get("name"), str) or not isinstance(step.get("arguments", {}), dict):
                            raise ValueError("invalid tool_path step")
                elif key == "token_f1":
                    if not isinstance(value, dict) or set(value) != {"reference", "threshold"} or not isinstance(value["reference"], str) or type(value["threshold"]) not in (int, float) or not 0 <= value["threshold"] <= 1:
                        raise ValueError("token_f1 requires reference and threshold in [0,1]")
                elif key == "json_contract":
                    if not isinstance(value, dict) or set(value) - {"types", "values", "allow_extra"}:
                        raise ValueError("invalid JSON contract")
                    if not isinstance(value.get("types"), dict) or not value["types"] or any(t not in {"string", "number", "integer", "boolean", "array", "object", "null"} for t in value["types"].values()):
                        raise ValueError("JSON contract requires valid property types")
                    if not isinstance(value.get("values", {}), dict) or set(value.get("values", {})) - set(value["types"]):
                        raise ValueError("JSON contract values must refer to typed properties")
                    if type(value.get("allow_extra", False)) is not bool:
                        raise ValueError("allow_extra must be boolean")
            seen.add(row["id"])
            canonical.append(row)
            cases.append(Case(row["id"], row["prompt"], tuple(row["tags"]), checks))
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not cases:
        raise ValueError("dataset is empty")
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return cases, hashlib.sha256(payload.encode()).hexdigest()

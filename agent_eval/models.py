"""Contracts separating gold labels from the system under evaluation."""

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentInput:
    prompt: str


@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    tags: tuple[str, ...]
    checks: dict[str, Any]


@dataclass
class ToolEvent:
    name: str
    arguments: dict[str, Any]
    output: Any = None
    error: str | None = None


@dataclass
class Prediction:
    answer: str
    trace: list[ToolEvent] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class Adapter(Protocol):
    name: str

    async def predict(self, item: AgentInput) -> Prediction: ...


class TransientError(Exception):
    """Only explicitly transient adapter failures are eligible for retries."""


@dataclass(frozen=True)
class Metric:
    name: str
    score: float
    threshold: float
    reason: str

    @property
    def passed(self) -> bool:
        return self.score >= self.threshold

    def to_dict(self) -> dict:
        return {**asdict(self), "passed": self.passed}


@dataclass
class CaseResult:
    case_id: str
    repetition: int
    tags: tuple[str, ...]
    status: str
    latency_ms: float
    attempts: int
    prediction: Prediction | None
    metrics: list[Metric]
    error_type: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "ok" and bool(self.metrics) and all(m.passed for m in self.metrics)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["metrics"] = [m.to_dict() for m in self.metrics]
        result["passed"] = self.passed
        return result

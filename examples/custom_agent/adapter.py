"""Translate an application's native result into framework contracts."""

from agent_eval.models import AgentInput, Prediction, ToolEvent
from examples.custom_agent.agent import run_agent


class CustomAgentAdapter:
    name = "scripted-custom-orders-v1"

    async def predict(self, item: AgentInput) -> Prediction:
        raw = await run_agent(item.prompt)
        usage = raw.get("usage") or {}
        return Prediction(
            answer=raw["text"],
            trace=[
                ToolEvent(
                    name=event["tool"],
                    arguments=event["input"],
                    output=event.get("result"),
                    error=event.get("error"),
                )
                for event in raw["events"]
            ],
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cost_usd=usage.get("cost_usd"),
        )

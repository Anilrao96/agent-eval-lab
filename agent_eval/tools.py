"""Read-only fictional support tools and a bounded arithmetic sandbox."""

import ast
import math
import operator
from pathlib import Path
import json

from .models import ToolEvent

SCHEMAS = [
    {"type": "function", "function": {"name": "search_policy", "description": "Retrieve support policy documents by topic", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_order", "description": "Read an order's status", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate basic arithmetic (+ - * /) without executing code", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"], "additionalProperties": False}}},
]


def calculate(expression: str) -> float:
    if not isinstance(expression, str) or len(expression) > 120:
        raise ValueError("expression too long")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 40:
        raise ValueError("expression too complex")
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            value = operations[type(node.op)](visit(node.left), visit(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        else:
            raise ValueError("unsupported arithmetic syntax")
        if not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("result out of range")
        return value

    return visit(tree.body)


class ToolBox:
    def __init__(self, knowledge_path: str | Path):
        self.data = json.loads(Path(knowledge_path).read_text(encoding="utf-8"))

    def call(self, name: str, arguments: dict) -> ToolEvent:
        try:
            if name == "search_policy" and set(arguments) == {"query"} and isinstance(arguments["query"], str):
                query = arguments["query"].casefold()
                documents = [p for p in self.data["policies"] if any(w in query for w in p["keywords"])]
                documents.sort(key=lambda p: max(len(w) for w in p["keywords"] if w in query), reverse=True)
                output = {"documents": documents}
            elif name == "get_order" and set(arguments) == {"order_id"} and isinstance(arguments["order_id"], str):
                output = self.data["orders"].get(arguments["order_id"].upper(), {"found": False})
            elif name == "calculator" and set(arguments) == {"expression"}:
                output = {"value": calculate(arguments["expression"])}
            else:
                raise ValueError("unknown tool or invalid arguments; write actions are unavailable")
            return ToolEvent(name, arguments, output)
        except (ValueError, TypeError, SyntaxError, ArithmeticError):
            return ToolEvent(name, arguments, error="tool_rejected")

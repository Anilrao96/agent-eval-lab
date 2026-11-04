"""A fictional read-only order agent; no framework imports or evaluation labels.

This is a scripted application used to demonstrate adapter integration. Replace
it with your real application without changing the evaluator's contracts.
"""

import json
import re

ORDERS = {
    "A100": {"status": "shipped", "shipment_id": "S100"},
    "A200": {"status": "processing", "shipment_id": None},
    "A300": {"status": "delivered", "shipment_id": "S300"},
}
SHIPMENTS = {
    "S100": {"delivery_estimate": "2 business days"},
    "S300": {"delivery_estimate": "delivered yesterday"},
}


async def run_agent(prompt: str) -> dict:
    """Return an application's own answer/event format, not a Prediction."""
    events = []

    def lookup(tool: str, arguments: dict, database: dict, key: str) -> dict:
        record = database.get(key)
        result = {"found": record is not None, **(record or {})}
        # Capture the tool invocation where it actually happens, before scoring.
        events.append({"tool": tool, "input": arguments, "result": result})
        return result

    if re.search(r"\b(cancel|delete)\b", prompt, re.IGNORECASE):
        return {"text": "I cannot modify or cancel orders.", "events": events}

    match = re.search(r"\bA\d{3}\b", prompt, re.IGNORECASE)
    if not match:
        return {"text": "Please include an order ID such as A100.", "events": events}

    order_id = match[0].upper()
    order = lookup("get_order", {"order_id": order_id}, ORDERS, order_id)
    if not order["found"]:
        return {"text": "I don't know; order not found.", "events": events}

    estimate = None
    if order["shipment_id"]:
        shipment_id = order["shipment_id"]
        shipment = lookup("get_shipment", {"shipment_id": shipment_id}, SHIPMENTS, shipment_id)
        estimate = shipment.get("delivery_estimate")

    if re.search(r"\bjson\b", prompt, re.IGNORECASE):
        answer = json.dumps({"order_id": order_id, "status": order["status"], "delivery_estimate": estimate})
    else:
        answer = f"Order {order_id} is {order['status']}."
        if estimate is not None:
            answer += f" Delivery estimate: {estimate}."
    # Ordinary Python code has no provider token usage; do not fabricate zeros.
    return {"text": answer, "events": events}

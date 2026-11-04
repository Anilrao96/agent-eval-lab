import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from agent_eval.dataset import load_dataset
from agent_eval.metrics import evaluate
from agent_eval.models import AgentInput
from examples.custom_agent.adapter import CustomAgentAdapter
from examples.custom_agent.evaluate import EXAMPLE_DIR, main


class CustomAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_dependent_lookups_capture_actual_outputs(self):
        prediction = await CustomAgentAdapter().predict(AgentInput("Track order A100."))
        self.assertIn("2 business days", prediction.answer)
        self.assertEqual([event.name for event in prediction.trace], ["get_order", "get_shipment"])
        self.assertEqual(prediction.trace[1].arguments["shipment_id"], prediction.trace[0].output["shipment_id"])
        self.assertIsNone(prediction.cost_usd)
        self.assertIsNone(prediction.input_tokens)

    async def test_adapter_passes_only_prompt_and_preserves_failed_events_and_usage(self):
        raw = {
            "text": "Cannot complete request.",
            "events": [{"tool": "cancel_order", "input": {"order_id": "A100"}, "error": "denied"}],
            "usage": {"input_tokens": 12, "output_tokens": 4, "cost_usd": .001},
        }
        with patch("examples.custom_agent.adapter.run_agent", new_callable=AsyncMock, return_value=raw) as application:
            prediction = await CustomAgentAdapter().predict(AgentInput("Cancel A100."))
        application.assert_awaited_once_with("Cancel A100.")
        self.assertEqual(prediction.trace[0].error, "denied")
        self.assertIsNone(prediction.trace[0].output)
        self.assertEqual((prediction.input_tokens, prediction.output_tokens, prediction.cost_usd), (12, 4, .001))

    async def test_correct_answer_does_not_hide_wrong_tool_order(self):
        cases, _ = load_dataset(EXAMPLE_DIR / "cases.jsonl")
        prediction = await CustomAgentAdapter().predict(AgentInput(cases[0].prompt))
        self.assertTrue(all(metric.passed for metric in evaluate(cases[0], prediction, 1)))
        prediction.trace.reverse()
        metrics = {metric.name: metric for metric in evaluate(cases[0], prediction, 1)}
        self.assertTrue(metrics["contains_all"].passed)
        self.assertFalse(metrics["tool_path"].passed)


class CustomCliTests(unittest.TestCase):
    def test_example_generates_repeated_results_and_all_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main(["--out", directory, "--repetitions", "3"]), 0)
            output = Path(directory)
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["summary"]["count"], 21)
            self.assertEqual(run["summary"]["pass_rate"], 1)
            self.assertEqual(len((output / "results.jsonl").read_text(encoding="utf-8").splitlines()), 21)
            self.assertIn("SCRIPTED DEMO", (output / "report.html").read_text(encoding="utf-8"))
            self.assertIn("21/21 passed; gate PASS", stdout.getvalue())
            self.assertTrue(all(row["prediction"]["input_tokens"] is None for row in run["results"]))

    def test_bad_answer_fails_gate_but_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            rows = [json.loads(line) for line in (EXAMPLE_DIR / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
            rows[0]["checks"]["contains_all"] = ["impossible expected answer"]
            dataset = Path(directory) / "cases.jsonl"
            dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            self.assertEqual(main(["--dataset", str(dataset), "--out", str(output)]), 1)
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertFalse(run["results"][0]["passed"])
            self.assertTrue((output / "report.html").is_file())

    def test_safety_failure_blocks_gate_even_with_minimum_zero(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            rows = [json.loads(line) for line in (EXAMPLE_DIR / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
            rows[-1]["checks"]["contains_all"] = ["unexpected refusal wording"]
            dataset = Path(directory) / "cases.jsonl"
            dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            self.assertEqual(main(["--dataset", str(dataset), "--out", str(Path(directory) / "run"), "--min-pass-rate", "0"]), 1)

    def test_invalid_config_fails_before_writing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            output = Path(directory) / "run"
            self.assertEqual(main(["--out", str(output), "--repetitions", "0"]), 2)
            self.assertEqual(main(["--out", str(output), "--min-pass-rate", "1.1"]), 2)
            self.assertFalse(output.exists())

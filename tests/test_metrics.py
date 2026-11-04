import unittest

from agent_eval.metrics import evaluate, token_f1
from agent_eval.models import Case, Prediction, ToolEvent
from agent_eval.tools import calculate


class MetricTests(unittest.TestCase):
    def score(self, checks, answer="", trace=None, **usage):
        return evaluate(Case("x", "prompt", ("unit",), checks), Prediction(answer, trace or [], **usage), 25)

    def test_exact_normalizes_punctuation_and_case(self):
        self.assertTrue(self.score({"exact": "Hello, World!"}, "hello world")[0].passed)

    def test_token_f1_counts_duplicates(self):
        self.assertAlmostEqual(token_f1("a a b", "a b"), .8)
        self.assertEqual(token_f1("", ""), 1)
        self.assertEqual(token_f1("", "answer"), 0)

    def test_f1_threshold_applies(self):
        self.assertFalse(self.score({"token_f1": {"reference": "a b c", "threshold": .9}}, "a b")[0].passed)

    def test_numeric_tolerance_handles_float_arithmetic(self):
        self.assertTrue(self.score({"numeric": {"value": 21.6, "abs_tol": 1e-6}}, str(120 * .18))[0].passed)
        for bad in ("NaN", "Infinity", "The answer is 21.6", "22"):
            with self.subTest(bad=bad):
                self.assertFalse(self.score({"numeric": {"value": 21.6, "abs_tol": 1e-6}}, bad)[0].passed)

    def test_negation_is_not_accepted_as_eligibility(self):
        checks = {"contains_all": ["eligible"], "contains_none": ["not eligible"]}
        self.assertFalse(all(m.passed for m in self.score(checks, "Not eligible")))

    def test_json_rejects_extra_properties_markdown_and_wrong_types(self):
        contract = {"json_contract": {"types": {"amount": "integer"}, "values": {"amount": 3}}}
        self.assertTrue(self.score(contract, '{"amount":3}')[0].passed)
        for bad in ('{"amount":true}', '{"amount":3.0}', '{"amount":3,"x":1}', '```json\n{"amount":3}\n```', '[3]', '{"amount":NaN}'):
            with self.subTest(bad=bad):
                self.assertFalse(self.score(contract, bad)[0].passed)

    def test_tool_order_and_arguments_are_required(self):
        checks = {"tool_path": [{"name": "get_order", "arguments": {"order_id": "A200"}}, {"name": "search_policy"}]}
        good = [ToolEvent("get_order", {"order_id": "A200"}), ToolEvent("search_policy", {"query": "returns"})]
        self.assertTrue(self.score(checks, trace=good)[0].passed)
        self.assertFalse(self.score(checks, trace=list(reversed(good)))[0].passed)
        good[0].arguments["order_id"] = "A300"
        self.assertFalse(self.score(checks, trace=good)[0].passed)

    def test_failed_tool_does_not_satisfy_required_step(self):
        self.assertFalse(self.score({"tool_path": [{"name": "get_order"}]}, trace=[ToolEvent("get_order", {}, error="rejected")])[0].passed)

    def test_forbidden_attempt_fails_even_when_tool_rejects_it(self):
        self.assertFalse(self.score({"forbidden_tools": ["cancel_order"]}, trace=[ToolEvent("cancel_order", {}, error="blocked")])[0].passed)

    def test_citation_must_be_present_and_retrieved(self):
        checks = {"citations": ["returns"]}
        trace = [ToolEvent("search_policy", {}, {"documents": [{"id": "returns"}]})]
        self.assertTrue(self.score(checks, "30 days [policy:returns]", trace)[0].passed)
        self.assertFalse(self.score(checks, "30 days [policy:returns]")[0].passed)
        self.assertFalse(self.score(checks, "[policy:returns] [policy:invented]", trace)[0].passed)

    def test_unknown_cost_and_tokens_fail_closed(self):
        self.assertFalse(all(m.passed for m in self.score({"max_cost_usd": .01, "max_tokens": 100})))
        self.assertTrue(all(m.passed for m in self.score({"max_cost_usd": .01, "max_tokens": 100}, input_tokens=60, output_tokens=40, cost_usd=.01)))

    def test_latency_and_trace_budgets_use_boundaries(self):
        self.assertTrue(self.score({"max_latency_ms": 25})[0].passed)
        self.assertFalse(self.score({"max_latency_ms": 24})[0].passed)
        self.assertFalse(self.score({"max_tool_calls": 0}, trace=[ToolEvent("read", {})])[0].passed)

    def test_calculator_rejects_code_and_unbounded_operations(self):
        self.assertEqual(calculate("(35 + 15) * 2"), 100)
        for bad in ("__import__('os').system('echo unsafe')", "2 ** 100000", "True + 1", "1 / 0", "9" * 121, "[1,2]"):
            with self.subTest(bad=bad):
                with self.assertRaises((ValueError, ArithmeticError)):
                    calculate(bad)

import asyncio
import unittest

from agent_eval.models import Case, Prediction, TransientError
from agent_eval.runner import RunConfig, run_suite


class Stub:
    name = "stub"

    async def predict(self, item):
        return Prediction("ok")


CASES = [Case(str(i), f"task {i}", ("unit",), {"exact": "ok"}) for i in range(8)]


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_expected_answers_never_reach_adapter(self):
        class Inspect(Stub):
            async def predict(self, item):
                self.keys = set(vars(item))
                return Prediction("ok")
        adapter = Inspect()
        run = await run_suite(CASES, "datahash", adapter, RunConfig(repetitions=2))
        self.assertEqual(adapter.keys, {"prompt"})
        self.assertEqual(run["summary"]["count"], 16)
        self.assertEqual(run["summary"]["pass_rate"], 1)
        self.assertEqual(len({(r['case_id'], r['repetition']) for r in run['results']}), 16)

    async def test_concurrency_is_bounded(self):
        class Active(Stub):
            current, maximum = 0, 0
            async def predict(self, item):
                self.current += 1
                self.maximum = max(self.maximum, self.current)
                await asyncio.sleep(.01)
                self.current -= 1
                return Prediction("ok")
        adapter = Active()
        await run_suite(CASES, "x", adapter, RunConfig(concurrency=2))
        self.assertEqual(adapter.maximum, 2)

    async def test_timeout_is_a_failed_case(self):
        class Slow(Stub):
            async def predict(self, item):
                await asyncio.sleep(.1)
                return Prediction("ok")
        run = await run_suite(CASES[:1], "x", Slow(), RunConfig(timeout_s=.01))
        self.assertEqual(run['results'][0]['status'], 'timeout')
        self.assertEqual(run['summary']['errors'], 1)
        self.assertFalse(run['results'][0]['passed'])

    async def test_only_transient_errors_retry_and_usage_becomes_unknown(self):
        class Flaky(Stub):
            calls = 0
            async def predict(self, item):
                self.calls += 1
                if self.calls == 1:
                    raise TransientError("sensitive provider body")
                return Prediction("ok", input_tokens=10, output_tokens=5, cost_usd=.01)
        run = await run_suite(CASES[:1], "x", Flaky(), RunConfig(retries=1))
        result = run['results'][0]
        self.assertTrue(result['passed'])
        self.assertIsNone(result['error_type'])
        self.assertEqual(result['attempts'], 2)
        self.assertIsNone(result['prediction']['cost_usd'])
        self.assertIsNone(result['prediction']['input_tokens'])

    async def test_nontransient_error_is_isolated_without_message_leak(self):
        class Broken(Stub):
            async def predict(self, item):
                if item.prompt == "task 0":
                    raise ValueError("private_token_123")
                return Prediction("ok")
        run = await run_suite(CASES, "x", Broken(), RunConfig(retries=2))
        self.assertEqual(run['summary']['passed'], 7)
        self.assertEqual(run['results'][0]['attempts'], 1)
        self.assertNotIn('private_token_123', str(run))

    async def test_retry_exhaustion_fails(self):
        class Unavailable(Stub):
            async def predict(self, item):
                raise TransientError("unavailable")
        run = await run_suite(CASES[:1], "x", Unavailable(), RunConfig(retries=1))
        self.assertEqual(run['results'][0]['attempts'], 2)
        self.assertFalse(run['results'][0]['passed'])


class ConfigTests(unittest.TestCase):
    def test_invalid_execution_limits_rejected(self):
        for kw in ({"concurrency": 0}, {"timeout_s": float('nan')}, {"timeout_s": 0}, {"retries": 6}, {"repetitions": 0}):
            with self.subTest(kw=kw):
                with self.assertRaises(ValueError):
                    RunConfig(**kw)

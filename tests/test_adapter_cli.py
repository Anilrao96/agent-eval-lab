import asyncio
import contextlib
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
from unittest.mock import patch

from agent_eval.adapters import ChatAdapter
from agent_eval.cli import main
from agent_eval.models import AgentInput, TransientError
from agent_eval.report import render_report
from agent_eval.tools import ToolBox

ROOT = Path(__file__).resolve().parent.parent


class MockHandler(BaseHTTPRequestHandler):
    bodies = []
    mode = 'normal'

    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.bodies.append(payload)
        if self.mode == 'rate_limit':
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b'private provider body')
            return
        if not any(m['role'] == 'tool' for m in payload['messages']):
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {'name': 'get_order', 'arguments': '{"order_id":"A100"}'}}]}
        else:
            message = {'role': 'assistant', 'content': 'Order A100 is shipped.'}
        response = {'choices': [{'message': message}], 'usage': {'prompt_tokens': 20, 'completion_tokens': 10}}
        if self.mode == 'no_usage':
            response.pop('usage')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())


class LiveAdapterTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), MockHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        MockHandler.bodies = []
        MockHandler.mode = 'normal'
        self.tools = ToolBox(ROOT / 'agent_eval/data/knowledge.json')
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1'

    async def test_real_http_tool_loop_and_usage_aggregation(self):
        adapter = ChatAdapter(self.tools, self.url, 'mock', input_per_million=1, output_per_million=2)
        result = await adapter.predict(AgentInput('Check A100'))
        self.assertEqual(result.answer, 'Order A100 is shipped.')
        self.assertEqual(result.trace[0].arguments, {'order_id': 'A100'})
        self.assertEqual((result.input_tokens, result.output_tokens), (40, 20))
        self.assertAlmostEqual(result.cost_usd, .00008)
        self.assertEqual(len(MockHandler.bodies), 2)
        self.assertEqual(MockHandler.bodies[1]['messages'][-1]['role'], 'tool')
        self.assertNotIn('checks', json.dumps(MockHandler.bodies))

    async def test_missing_usage_is_unknown_not_zero(self):
        MockHandler.mode = 'no_usage'
        result = await ChatAdapter(self.tools, self.url, 'mock').predict(AgentInput('Check A100'))
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.cost_usd)

    async def test_rate_limit_is_transient_and_body_is_not_exposed(self):
        MockHandler.mode = 'rate_limit'
        with self.assertRaises(TransientError) as caught:
            await ChatAdapter(self.tools, self.url, 'mock').predict(AgentInput('Check A100'))
        self.assertNotIn('private', str(caught.exception))

    async def test_step_and_tool_limits_fail_instead_of_fabricating_answer(self):
        with self.assertRaisesRegex(RuntimeError, 'step_limit'):
            await ChatAdapter(self.tools, self.url, 'mock', max_steps=1).predict(AgentInput('Check A100'))
        with self.assertRaisesRegex(RuntimeError, 'tool_limit'):
            await ChatAdapter(self.tools, self.url, 'mock', max_tool_calls=0).predict(AgentInput('Check A100'))

    async def test_redirects_are_not_followed_with_credentials(self):
        error = urllib.error.HTTPError(self.url, 302, 'redirect', {}, None)
        with patch('urllib.request.OpenerDirector.open', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'provider_http_302'):
                await ChatAdapter(self.tools, self.url, 'mock', api_key='private-value').predict(AgentInput('x'))

    def test_insecure_remote_and_embedded_credentials_rejected(self):
        for url in ('http://example.com/v1', 'https://key@example.com/v1', 'https://example.com?key=x', 'file:///tmp/model'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ChatAdapter(self.tools, url, 'mock')

    def test_invalid_prices_and_limits_are_rejected(self):
        for kw in ({'input_per_million': float('nan'), 'output_per_million': 1}, {'input_per_million': 1}, {'max_steps': 0}, {'timeout': float('inf')}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                ChatAdapter(self.tools, self.url, 'mock', **kw)


class CliTests(unittest.TestCase):
    def test_demo_compare_exit_codes_and_escaped_html(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            self.assertEqual(main(['demo', '--out', str(output)]), 0)
            before, after = output / 'baseline/run.json', output / 'candidate/run.json'
            self.assertEqual(main(['compare', '--baseline', str(before), '--candidate', str(after), '--out', str(output / 'compare')]), 0)
            self.assertEqual(main(['compare', '--baseline', str(after), '--candidate', str(before), '--out', str(output / 'reverse')]), 1)
            run = json.loads(after.read_text(encoding='utf-8'))
            self.assertEqual(run['summary']['pass_rate'], 1)
            run['results'][0]['prediction']['answer'] = '<script>alert("injected")</script>'
            report = render_report(run)
            self.assertNotIn('<script>alert', report)
            self.assertIn('&lt;script&gt;alert', report)
            self.assertIn('SCRIPTED DEMO', report)
            self.assertEqual(len((output / 'candidate/results.jsonl').read_text(encoding='utf-8').splitlines()), 24)

    def test_invalid_configuration_exits_two_without_network(self):
        with patch.dict('os.environ', {}, clear=True), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['run']), 2)
            self.assertEqual(main(['demo', '--concurrency', '0']), 2)

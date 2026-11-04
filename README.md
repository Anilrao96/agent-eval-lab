# Agent Eval Lab

**A framework built from scratch for evaluating LLM answers and multi-step tool-using agents.**

[![Evaluate](https://github.com/Anilrao96/agent-eval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Anilrao96/agent-eval-lab/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-245c71)
![Runtime dependencies: 0](https://img.shields.io/badge/runtime_dependencies-0-075e47)

An agent can produce a plausible answer after calling the wrong tool, looking up the wrong order, or attempting a forbidden action. Agent Eval Lab evaluates the **answer, execution trace and operational budget together**, and turns that evidence into a repeatable regression gate.

This is an **AI-assisted portfolio project**, not a production deployment or a claim of prior professional evaluation ownership. The included results use scripted agents and fictional data; no live model benchmark is claimed.

## Try it in 30 seconds

Requires Python 3.10+. No package installation, API key or model download is needed for the demo.

```bash
git clone https://github.com/Anilrao96/agent-eval-lab.git
cd agent-eval-lab
python -m unittest discover -s tests -v
python -m agent_eval demo --out runs/demo --repetitions 3
```

Open `runs/demo/report.html`. On Windows, if `python` is unavailable and `uv` is installed:

```powershell
$pythonPath = (uv python find).Trim()
& $pythonPath -m agent_eval demo --out runs/demo --repetitions 3
Start-Process .\runs\demo\report.html
```

The demonstration compares a deliberately brittle support agent with a more careful scripted agent on **24 authored cases**. The baseline passes **4/24** and the candidate passes **24/24**. With three repetitions, these become 12/72 and 72/72 executions. The agents use a separate knowledge store and never receive expected answers or evaluation rules. These numbers demonstrate scorer behavior, not LLM quality or generalization.

[View the sample report source/download](examples/demo/report.html) · [Architecture and decisions](docs/architecture.md) · [Evaluation methodology](docs/evaluation.md) · [Interview preparation](docs/interview.md)

## What is implemented

| Layer | Behavior |
|---|---|
| Dataset contracts | Strict JSONL validation, unique IDs, named slices, SHA-256 dataset and evaluation fingerprints |
| Answer checks | Normalized equality, numeric tolerance, required/forbidden phrases, multiset token F1, strict JSON object contracts |
| Agent checks | Successful ordered tool subsequences, argument matching, forbidden tool attempts, tool-call limits |
| Grounding checks | Required citation IDs must appear and must have been retrieved; this does not establish entailment |
| Operations | Bounded case concurrency, per-case logical deadlines, opt-in transient retries, repeated trials, latency/token/cost checks |
| Regression gate | Key-aligned comparisons, no new case failures, minimum candidate pass rate, execution and safety failure gates |
| Uncertainty | Seeded paired bootstrap over cases; repeated executions stay in the same resampling cluster |
| Evidence | JSON run manifest, per-execution JSONL, slice summaries and a self-contained HTML report |
| Real model integration | Chat-completions HTTP adapter with a bounded read-only tool loop and provider usage aggregation |
| CI | Python 3.10/3.13 unit and mock HTTP integration tests, offline regression gate and uploaded reports |

**No evaluator libraries wrap the core logic.** The runner, contracts, metrics, comparison and reports use Python's standard library. There is no mandatory LLM judge: lexical and trace checks are cheap and inspectable. Semantic judgments still require a separately calibrated judge or human review.

## Evaluate a live model

Use a running chat-completions compatible endpoint and a model that supports tool calling. The adapter is designed against the documented [Ollama compatibility API](https://docs.ollama.com/api/openai-compatibility). Compatibility is not a guarantee across every provider or model; the HTTP protocol is integration-tested with a local mock server.

```bash
python -m agent_eval run \
  --base-url http://localhost:11434/v1 \
  --model YOUR_TOOL_CAPABLE_MODEL \
  --out runs/local-model \
  --concurrency 2 --timeout 60 --repetitions 3
```

The command above uses Bash line continuations; in PowerShell run it on one line. Alternatively set `EVAL_BASE_URL` and `EVAL_MODEL`. For an authenticated HTTPS provider, set `EVAL_API_KEY` in your environment; credentials are never stored in run metadata. `.env.example` is a template only; the CLI does not automatically load `.env` files.

Token usage comes from the provider. Cost is **unknown** unless both `--input-price` and `--output-price` are supplied, in USD per million tokens. Supply prices for your actual model rather than assuming a default. A missing usage field or a retried execution makes usage/cost unknown; explicit budget checks then fail closed. Reported totals cover successful calls with known usage, not a billing ledger.

The bundled data deliberately tests multiple capabilities. Customize checks and tools for your own application before interpreting live scores. Only fictional read-only tools are available; no actual shop or customer account is connected.

## Compare two versions

```bash
python -m agent_eval compare \
  --baseline runs/demo/baseline/run.json \
  --candidate runs/demo/candidate/run.json \
  --out runs/comparison --min-pass-rate 0.90
```

Comparison requires the same dataset fingerprint, evaluation fingerprint, case/repetition keys and repetition count. A higher average cannot conceal a newly failing case. Safety failures or execution errors fail the gate. `--max-drop` controls the aggregate drop tolerance; it never permits new individual case failures.

Exit codes: **0** gate passed, **1** evaluation/regression gate failed, **2** invalid configuration or artifact.

## Add your application

Implement `async predict(AgentInput) -> Prediction` and a `name` property. The adapter receives only a prompt; it returns answer text, structured tool events, and usage when known. Use `run_suite()` from Python:

```python
import asyncio
from agent_eval.dataset import load_dataset
from agent_eval.models import Prediction
from agent_eval.runner import RunConfig, run_suite
from agent_eval.report import write_artifacts

class MyApplication:
    name = "my-app-v1"

    async def predict(self, item):
        # Replace this with your application call and capture its actual tool events.
        return Prediction(answer="Your application's answer")

cases, fingerprint = load_dataset("agent_eval/data/support_v1.jsonl")
run = asyncio.run(run_suite(cases, fingerprint, MyApplication(), RunConfig()))
write_artifacts(run, "runs/custom")
```

An example case:

```json
{"id":"order-check","prompt":"Check order A100","tags":["agent"],"checks":{"contains_all":["shipped"],"tool_path":[{"name":"get_order","arguments":{"order_id":"A100"}}],"max_tool_calls":1}}
```

Use `max_latency_ms`, `max_tokens` and `max_cost_usd` when those budgets are part of the task contract. Every configured check must pass; scores are not averaged to hide a critical failure. See [the methodology](docs/evaluation.md) for exact semantics and limitations.

## Repository map

```text
agent_eval/    Contracts, loader, adapters, tools, runner, metrics, gates, CLI, report
agent_eval/data/ Fictional knowledge store and 24-case JSONL evaluation set
tests/         Scorer, runner, comparison, CLI and local HTTP integration tests
docs/          Architecture, evaluation methodology and interview walkthrough
examples/demo/ Generated offline evidence, labeled as scripted
.github/       CI that runs without credentials or external model calls
```

Do not commit real prompts or traces without reviewing them. Run artifacts include raw answers and tool outputs and can contain sensitive information. By default, `runs/`, `.env` and local caches are ignored.

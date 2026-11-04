# Evaluation methodology

## Dataset

`support_v1.jsonl` contains 24 manually authored synthetic tasks for a fictional support shop: policy QA, missing information, order lookup, multi-step return eligibility, a boundary condition, strict JSON, arithmetic and unsafe requests. Tags overlap. It is a small regression suite, not a representative benchmark, blinded holdout or dataset with production-derived labels.

`knowledge.json` contains the agent-accessible policies and orders. The scripted adapters parse prompts and use those tools; they never see checks. The baseline deliberately omits citations, mishandles unknowns and JSON, returns incorrect arithmetic answers and attempts a blocked write. The candidate repairs those known behaviors. Its perfect score is expected on these fixtures and gives no evidence of unseen-task quality.

## Scorer semantics

| Check | Exact behavior | Important limit |
|---|---|---|
| `exact` | Case-fold, tokenize word characters, collapse whitespace, compare | Punctuation-sensitive domains should use another evaluator |
| `numeric` | Parse entire answer as finite float; compare with explicit absolute tolerance | Does not extract a number from prose; not decimal accounting |
| `contains_all/any/none` | Case-insensitive substring checks | Negation, evasion and paraphrases need stronger contracts/human review |
| `token_f1` | Multiset token overlap: `2*overlap/(answer_tokens+reference_tokens)` | Lexical proxy; high overlap can still be factually wrong |
| `json_contract` | Parse strict JSON object; required property types/values; reject extra keys by default | Small top-level contract, not full JSON Schema; nested validation not implemented |
| `tool_path` | Successful ordered subsequence, matching named argument subsets | Extra calls are allowed unless independently limited |
| `forbidden_tools` | Fail on any attempted named tool, including rejected attempts | Only detects actions captured in the trace |
| `citations` | Required IDs present; every cited policy ID was successfully retrieved | Evidence availability, not claim-level entailment or faithfulness |
| `max_tool_calls` | Count every captured attempt | Does not inspect hidden work outside instrumentation |
| `max_latency_ms` | Measured execution wall time including retries, excluding queue wait | Scripted timing is not model latency; coroutine timeouts are logical |
| `max_tokens` | Sum known input/output tokens | Unknown or retried usage fails the check |
| `max_cost_usd` | Provider token counts times both supplied per-million prices | Not billed cost; discounts/caching/failure charges are not modeled |

Each execution passes only if its status is `ok`, at least one metric exists, and **every configured metric passes**. Missing predictions, scorer errors and unknown required resource usage fail closed. Error types are recorded; raw provider error messages are not saved. Raw successful answers/traces are saved and need a data handling policy before real use.

## Comparing runs

Comparison aligns by `(case_id, repetition)` and rejects mismatched dataset/evaluation fingerprints, repetition counts, key sets, duplicate keys or differing tags. It recomputes pass rates from result rows rather than trusting summary fields.

The release gate fails for a candidate below the configured minimum pass rate (default 90%), an aggregate drop beyond tolerance, **any newly failing case/repetition**, any execution error, or any failing task tagged `safety`. The aggregate tolerance never disables the individual-case rule. A candidate can fail a task that already failed in baseline and still pass if all other thresholds hold; inspect all failures before deployment.

The paired delta is the candidate pass fraction minus baseline pass fraction. Each bootstrap draw resamples **case IDs with replacement**, keeping all repetitions for that case together, and averages case-level deltas. Default: 2,000 draws, seed 17, 2.5/97.5 percentile bounds. Those intervals describe this task mixture, not production uncertainty or a significance guarantee. Repeating a deterministic fixture does not add independent evidence. The gate does not use the interval as a hypothesis test.

## Validation strategy

The automated suite tests scorer counterexamples and execution behavior, including numeric rounding, eligibility negation, boolean/integer JSON confusion, fabricated citations, rejected forbidden actions, tool order and entity mismatch, unknown usage, input leakage, bounded concurrency, timeouts, retry exhaustion, sanitization, comparison mismatches, CLI exit codes and HTML escaping.

A local HTTP server tests the actual network adapter's request/response loop, tool replay, multi-turn usage aggregation, missing usage, HTTP 429 handling and agent step/call limits. These integration tests **do not validate a real provider's tool behavior or a real LLM's quality**. The supplied offline report is separate from those transport tests. No paid external API or model download is required in CI.

For a real evaluation: build representative tasks, separate development and holdout sets, have domain experts review gold labels, predeclare thresholds, capture real tool spans, run repeated trials against pinned model/prompt versions, review slice errors, and calibrate semantic scorers against human ratings before promotion.

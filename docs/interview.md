# Evaluation ownership: an honest walkthrough

The original application question is: “Have you built an evaluation framework for LLMs or multi-step AI agents from scratch? What did you personally architect?”

This repository is a newly built **AI-assisted portfolio project**. Do not present it as past employment work, a production deployment, independently written code, or evidence that you personally made every architecture decision. Use the distinction between project capabilities and your own contribution clearly.

## Application draft for the current stage

> I recently built an AI-assisted portfolio project, Agent Eval Lab, to evaluate LLM applications and multi-step tool-using agents. It includes versioned test cases, answer and JSON checks, tool-order and argument validation, citation evidence, latency/token/cost budgets, paired regression gates, and HTML/JSON reports. The core uses Python's standard library and has automated tests, including a mock HTTP tool-calling integration. The demo uses fictional data and scripted agents; it is not a production LLM benchmark. I can share the repository and walk through the architecture and its limitations.

Repository: https://github.com/Anilrao96/agent-eval-lab

For “what did you personally architect?”, add only decisions you actually made and can defend. The initial implementation was generated with coding assistance from the project brief. After reviewing and extending it yourself, describe those specific choices and changes rather than adopting a fictional ownership history.

## Ten-minute demo

1. Run the tests and the three-repetition demo from README.
2. Open the HTML report. Explain why the demo's 100% result is unsurprising on synthetic fixtures.
3. Open a baseline failure. Show the answer, failing check and captured trace.
4. Explain why a correct answer after a forbidden action must still fail.
5. Compare runs in reverse; the command exits 1 and records regression reasons.
6. Explain why missing cost is unknown, why retry cost cannot be assumed zero, and why tool citations do not establish entailment.
7. Show how the real HTTP adapter replays tool results and how its protocol is tested without calling a paid model.

## Questions to be ready for

- **How did you prevent label leakage?** The adapter receives an `AgentInput` with only a prompt. Gold checks remain in the scorer. This is an interface boundary, not a security sandbox against malicious local code.
- **Why no LLM judge?** Deterministic task contracts are inspectable and sufficient for this first version. Semantic judgment would need human calibration, judge provenance and injection testing.
- **Why evaluate the trajectory?** Wrong-entity lookups, unnecessary calls and forbidden attempts can be invisible in the final answer.
- **What makes the comparison fair?** Same dataset/check fingerprints and trial keys. Candidate prompt/model changes are recorded separately. A matching dataset alone does not establish production comparability.
- **How do you handle repeated runs?** Record every trial; resample cases as clusters for intervals. A deterministic demo's repeated scores are not independent new evidence.
- **What would you change for production?** Representative held-out data, human-reviewed labels, real span ingestion, hard cancellation, robust billing capture, redaction, resumable execution and calibrated semantic evaluation.

## Make the project personally defensible

Choose one genuine extension, implement it, and record your reasoning and tests. Examples: add application-specific held-out cases, integrate your own agent and traces, implement Decimal scoring for financial amounts, or add human-reviewed semantic labels and measure scorer disagreement. That creates concrete personal architecture work you can point to in an interview.

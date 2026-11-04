import copy
import json
from pathlib import Path
import tempfile
import unittest

from agent_eval.compare import compare_runs
from agent_eval.dataset import load_dataset

ROOT = Path(__file__).resolve().parent.parent


def fixture(passes, tags=None):
    return {"schema_version": 1, "run_id": "fixture", "dataset_sha256": "data", "evaluation_sha256": "eval", "config": {"repetitions": 1},
            "results": [{"case_id": str(i), "repetition": 0, "passed": p, "status": "ok", "tags": tags or ["qa"]} for i, p in enumerate(passes)]}


class DatasetTests(unittest.TestCase):
    def test_bundled_dataset_has_24_unique_cases(self):
        cases, digest = load_dataset(ROOT / 'agent_eval/data/support_v1.jsonl')
        self.assertEqual(len(cases), 24)
        self.assertEqual(len(digest), 64)

    def test_hash_is_stable_across_json_whitespace(self):
        row = {"id": "x", "prompt": "Q", "tags": ["unit"], "checks": {"exact": "A"}}
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'data.jsonl'
            p.write_text(json.dumps(row), encoding='utf-8')
            _, first = load_dataset(p)
            p.write_text(json.dumps(row, separators=(',', ':')) + '\n\n', encoding='utf-8')
            self.assertEqual(load_dataset(p)[1], first)

    def test_duplicate_ids_empty_and_invalid_contracts_rejected(self):
        row = {"id": "x", "prompt": "Q", "tags": ["unit"], "checks": {"exact": "A"}}
        bad_rows = ["", json.dumps(row) + '\n' + json.dumps(row)]
        for checks in ({}, {"typo": 1}, {"max_tokens": -1}, {"max_cost_usd": float('nan')}, {"tool_path": []}, {"json_contract": {"types": {"x": "madeup"}}}, {"token_f1": {"reference": "x", "threshold": 2}}):
            bad_rows.append(json.dumps({**row, "checks": checks}))
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'bad.jsonl'
            for content in bad_rows:
                with self.subTest(content=content):
                    p.write_text(content, encoding='utf-8')
                    with self.assertRaises(ValueError):
                        load_dataset(p)


class CompareTests(unittest.TestCase):
    def test_improvements_pass_and_bootstrap_is_deterministic(self):
        before, after = fixture([True, False, False]), fixture([True, True, True])
        comparison = compare_runs(before, after)
        self.assertTrue(comparison['gate_passed'])
        self.assertAlmostEqual(comparison['delta'], 2/3)
        self.assertEqual(comparison, compare_runs(before, after))
        self.assertEqual(comparison['bootstrap']['unit'], 'case')

    def test_new_failure_blocks_even_if_average_improves(self):
        result = compare_runs(fixture([True, False, False]), fixture([False, True, True]), min_pass_rate=.5)
        self.assertFalse(result['gate_passed'])
        self.assertIn('new_case_failures', result['gate_reasons'])

    def test_safety_failures_cannot_be_averaged_away(self):
        result = compare_runs(fixture([False], ['safety']), fixture([False], ['safety']), min_pass_rate=0)
        self.assertIn('candidate_safety_failure', result['gate_reasons'])

    def test_fingerprint_and_contract_mismatch_rejected(self):
        for key in ('dataset_sha256', 'evaluation_sha256', 'schema_version'):
            candidate = fixture([True])
            candidate[key] = 'changed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare_runs(fixture([True]), candidate)

    def test_missing_duplicate_and_repetition_mismatch_rejected(self):
        base = fixture([True, True])
        bad = [fixture([True])]
        duplicate = copy.deepcopy(base)
        duplicate['results'][1] = duplicate['results'][0]
        bad.append(duplicate)
        repeated = copy.deepcopy(base)
        repeated['config']['repetitions'] = 2
        bad.append(repeated)
        for candidate in bad:
            with self.assertRaises(ValueError):
                compare_runs(base, candidate)

    def test_results_align_by_keys_not_array_order(self):
        base = fixture([True, False])
        candidate = fixture([True, True])
        candidate['results'].reverse()
        self.assertTrue(compare_runs(base, candidate)['gate_passed'])

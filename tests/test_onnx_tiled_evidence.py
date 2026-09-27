"""Protect frozen decisions, source provenance and retained negative evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
from compare_onnx_tiled import check_protocol, validate as validate_parity
from benchmark_onnx_tiled import validate as validate_benchmark, summarize


class EvidenceTests(unittest.TestCase):
    def test_frozen_parity_and_checkpoint_identity(self):
        check_protocol()
        result = json.loads((ROOT/'results/onnx-tiled-comparison.json').read_text())
        validate_parity(result)
        for case, recorded in zip(result['protocol']['matrix'], result['random_cases']):
            self.assertEqual(case, recorded['case'])
        original = json.loads((ROOT/'results/synthetic.json').read_text())
        trained = result['held_out']
        self.assertEqual(trained['weights_sha256'], original['final_weights_sha256'])
        self.assertEqual(trained['full_logits_sha256'], original['held_out_logits_sha256'])
        self.assertEqual(trained['evaluation_sha256'], original['data']['evaluation_sha256'])
        self.assertEqual(trained['full_metrics'], original['final_held_out'])
        self.assertEqual(trained['checkpoint_sha256'], hashlib.sha256((ROOT/'results/synthetic-weights.pt').read_bytes()).hexdigest())
        changed = copy.deepcopy(result)
        changed['random_cases'][0]['within_tolerance'] = False
        with self.assertRaises(AssertionError):
            validate_parity(changed)
        changed = copy.deepcopy(result)
        changed['tolerance_failures'] = ['random:0']
        with self.assertRaises(AssertionError):
            validate_parity(changed)

    def test_benchmark_links_artifacts_and_numerics(self):
        result = json.loads((ROOT/'results/onnx-tiled-benchmark.json').read_text())
        validate_benchmark(result)
        for key in ('output_sha256', 'onnx_sha256', 'input_file_sha256'):
            changed = copy.deepcopy(result)
            changed['samples'][0][key] = '0'*64
            with self.assertRaises(AssertionError):
                validate_benchmark(changed)
        changed = copy.deepcopy(result)
        changed['artifacts']['small']['onnx_tiled.py']['sha256'] = '0'*64
        with self.assertRaisesRegex(AssertionError, 'source drift'):
            validate_benchmark(changed)
        # Slow observations are valid evidence, never a reason to loosen a gate.
        changed = copy.deepcopy(result)
        for record in changed['samples']:
            if record['method'] == 'onnx':
                record['seconds'] = [min(3599, t*1000) for t in record['seconds']]
        changed['summary'] = summarize(changed['samples'])
        validate_benchmark(changed)
        self.assertTrue(all(s['slower_onnx_pairs'] == 6 for s in changed['summary']))

    def test_initial_run_and_historical_tolerance_failure_preserved(self):
        initial = json.loads((ROOT/'results/onnx-tiled-benchmark-initial.json').read_text())
        source = (ROOT/'results/onnx-tiled-benchmark-initial-source.txt').read_bytes()
        helper = (ROOT/'results/onnx-tiled-benchmark-initial-runtime.txt').read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), initial['source_hashes']['experiments/benchmark_onnx_tiled.py'])
        self.assertEqual(initial['summary'], summarize(initial['samples']))
        self.assertEqual(len(initial['samples']), 36)
        for workload in ('small', 'medium'):
            self.assertEqual(hashlib.sha256(helper).hexdigest(), initial['artifacts'][workload]['onnx_tiled.py']['sha256'])
        self.assertNotEqual(initial['artifacts']['small']['onnx_tiled.py']['sha256'],
                            initial['source_hashes']['src/unet_budget/onnx_tiled.py'])
        with self.assertRaisesRegex(AssertionError, 'source drift'):
            validate_benchmark(initial)
        historical = json.loads((ROOT/'results/onnx-numerical-investigation.json').read_text())
        self.assertFalse(historical['comparisons_to_torch_float32']['onnx_default']['within_tolerance'])
        self.assertEqual(historical['comparisons_to_torch_float32']['onnx_default']['elementwise_violation_count'], 1)


if __name__ == '__main__':
    unittest.main()

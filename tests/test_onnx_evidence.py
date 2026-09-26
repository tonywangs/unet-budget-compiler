"""Guard numerical decisions and preserve the superseded measurement evidence."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
from compare_onnx import comparison
from benchmark_onnx import summarize


class EvidenceTests(unittest.TestCase):
    def test_elementwise_failure_is_not_hidden_by_aggregate_limits(self):
        reference = np.array([[[[-0.10544267296791077, 0], [0, 0]]]], dtype=np.float32)
        actual = np.array([[[[-0.10545620322227478, 0], [0, 0]]]], dtype=np.float32)
        with self.assertRaises(AssertionError):
            comparison(reference, actual)
        result = comparison(reference, actual, require_close=False)
        self.assertFalse(result['within_tolerance'])
        self.assertEqual(result['elementwise_violation_count'], 1)
        self.assertGreater(result['maximum_error_to_elementwise_bound'], 1)
        self.assertEqual(result['label_disagreement_fraction'], 0)
        self.assertLess(result['mean_absolute_error'], 1e-5)
        self.assertLess(result['maximum_absolute_error'], 1e-4)

    def test_binary_and_multiclass_labels_and_finite_contract(self):
        for reference, actual in ((np.zeros((1, 1, 2, 2), np.float32), np.full((1, 1, 2, 2), 1e-7, np.float32)),
                                  (np.zeros((1, 2, 2, 2), np.float32), np.array([[[[0, 0], [0, 0]], [[1e-7, 1e-7], [1e-7, 1e-7]]]], np.float32))):
            result = comparison(reference, actual, require_close=False)
            self.assertTrue(result['elementwise_close'])
            self.assertFalse(result['within_tolerance'])
            self.assertEqual(result['label_disagreement_fraction'], 1)
            with self.assertRaises(AssertionError):
                comparison(reference, actual)
            self.assertTrue(comparison(reference, reference)['within_tolerance'])
            with self.assertRaises(AssertionError):
                comparison(reference, actual * np.nan, require_close=False)

    def test_initial_measurements_and_source_are_preserved(self):
        initial = json.loads((ROOT/'results/onnx-benchmark-initial.json').read_text())
        source = (ROOT/'results/onnx-benchmark-initial-source.txt').read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), initial['source_hashes']['experiments/benchmark_onnx.py'])
        self.assertEqual(initial['summary'], summarize(initial['samples']))
        self.assertEqual(len(initial['samples']), 36)
        self.assertEqual(initial['protocol'], json.loads((ROOT/'experiments/onnx_protocol.json').read_text()))
        # The identical per-pair watermark is the observed measurement defect.
        for workload in initial['protocol']['benchmark']['workloads']:
            for pair in range(6):
                samples = [s for s in initial['samples'] if s['workload'] == workload['name'] and s['pair'] == pair]
                self.assertEqual(len(samples), 2)
                self.assertEqual(samples[0]['peak_rss_bytes'], samples[1]['peak_rss_bytes'])


if __name__ == '__main__':
    unittest.main()

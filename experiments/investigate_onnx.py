"""Investigate the frozen-tolerance failure without changing thresholds or weights.

This is a diagnostic, not a replacement model, benchmark configuration or accuracy
claim. Float64 PyTorch supplies a higher-precision reference on the same inputs.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile

import numpy as np
import onnxruntime as ort
import torch

from compare_onnx import PROTOCOL, PROTOCOL_PATH, comparison, model_from_spec, sha, array_sha
from unet_budget.export import export_model, read_checkpoint
from unet_budget.onnx_inference import load_bundle, predict

ROOT = Path(__file__).resolve().parents[1]


def against_double(actual, reference):
    error = np.abs(actual.astype(np.float64) - reference)
    return dict(maximum_absolute_error=float(error.max()), mean_absolute_error=float(error.mean()),
                logits_sha256=array_sha(actual))


def experiment():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    workload = next(w for w in PROTOCOL['benchmark']['workloads'] if w['name'] == 'larger')
    image = np.random.default_rng(PROTOCOL['benchmark']['input_seed']).standard_normal(workload['shape']).astype('float32')
    spec = json.loads((ROOT/'examples/circles.json').read_text())
    model, _ = model_from_spec(spec)
    checkpoint_hash = read_checkpoint(ROOT/'results/synthetic-weights.pt', model, torch)
    with torch.inference_mode():
        baseline = model(torch.from_numpy(image)).numpy()
        with torch.backends.mkldnn.flags(enabled=False):
            torch_unoptimized = model(torch.from_numpy(image)).numpy()
        double = model.double()(torch.from_numpy(image).double()).numpy()
    outputs = dict(torch_default=baseline, torch_mkldnn_disabled=torch_unoptimized)
    with tempfile.TemporaryDirectory(prefix='unet-onnx-diagnostic-') as temporary:
        bundle = Path(temporary)/'bundle'
        manifest = export_model(spec, ROOT/'results/synthetic-weights.pt', workload['shape'], bundle)
        session, contract = load_bundle(bundle)
        outputs['onnx_default'] = predict(session, contract, image)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(bundle/'model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        outputs['onnx_graph_optimizations_disabled'] = session.run(['logits'], {'image': image})[0]
    comparisons = {name: comparison(baseline, actual, require_close=False) for name, actual in outputs.items()}
    # Preserve the failing case as a negative result, not an adjusted tolerance.
    assert not comparisons['onnx_default']['within_tolerance'], 'recorded failure changed; investigate before refreshing evidence'
    result = dict(schema_version=1, protocol_sha256=sha(PROTOCOL_PATH), shape=workload['shape'],
                  tolerances=PROTOCOL['tolerances'], input_sha256=array_sha(image), checkpoint_sha256=checkpoint_hash,
                  onnx_sha256=manifest['artifacts']['model.onnx']['sha256'],
                  versions={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'onnxruntime', 'onnx')},
                  comparisons_to_torch_float32=comparisons,
                  errors_to_torch_float64={name: against_double(value, double) for name, value in outputs.items()},
                  torch_float64_logits_sha256=array_sha(double),
                  source_hashes={name: sha(ROOT/name) for name in ('experiments/investigate_onnx.py', 'experiments/compare_onnx.py',
                                  'src/unet_budget/export.py', 'src/unet_budget/onnx_inference.py')},
                  limitations=['Float64 is a diagnostic reference, not a mathematical exact oracle.',
                               'Disabling optimizations changes computation paths; this alone does not identify a specific native kernel.',
                               'No weights, tolerances, inputs or default runtime settings were changed after the failure.'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/onnx-numerical-investigation.json')
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    result = experiment()
    if args.check:
        assert json.loads(args.check.read_text()) == result, 'numerical investigation did not reproduce'
        print('Known tolerance failure and higher-precision/optimization diagnostics reproduced exactly.')
    else:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: dict(max_error=v['maximum_absolute_error'], violation_count=v['elementwise_violation_count'],
                              within_tolerance=v['within_tolerance'])
                      for k, v in result['comparisons_to_torch_float32'].items()}, sort_keys=True))
    print(json.dumps(result['errors_to_torch_float64'], sort_keys=True))


if __name__ == '__main__':
    main()

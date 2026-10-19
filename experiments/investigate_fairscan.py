"""Diagnose the retained no-overlap tolerance failure without altering defaults."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from fairscan import ROOT, compare, dataset, make_model, sha
from unet_budget.inference import tiled_logits as torch_tiled
from unet_budget.onnx_tiled import load_bundle, tiled_logits


def experiment():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = ROOT / 'results/fairscan'
    model, _ = make_model()
    model.load_state_dict(torch.load(output / 'weights.pt', weights_only=True))
    model.eval()
    identity, image, _ = next(row for row in dataset()['val'] if row[0] == 'IMG_20250614_210835')
    image = image.unsqueeze(0)
    settings = dict(tile_size=(65, 81), overlap=(0, 0), blend='constant', tile_batch_size=1)
    with torch.inference_mode():
        reference = torch_tiled(model, image, **settings).numpy()
        with torch.backends.mkldnn.flags(enabled=False):
            unoptimized = torch_tiled(model, image, **settings).numpy()
        double = torch_tiled(model.double(), image.double(), **settings).numpy()
    session, contract = load_bundle(output / 'bundle')
    actual = tiled_logits(session, contract, image.numpy(), overlap=(0, 0), blend='constant')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(output / 'bundle/model.onnx'), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    disabled = tiled_logits(session, contract, image.numpy(), overlap=(0, 0), blend='constant')
    outputs = dict(torch_default=reference, torch_mkldnn_disabled=unoptimized,
                   onnx_default=actual, onnx_optimizations_disabled=disabled)
    result = dict(image=identity, comparison_to_torch={k: compare(reference, v) for k, v in outputs.items()},
                  errors_to_float64={k: dict(max_absolute_error=float(np.abs(v-double).max()),
                                             mean_absolute_error=float(np.abs(v-double).mean())) for k, v in outputs.items()},
                  provenance={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
                              ROOT / 'experiments/fairscan.py', output / 'weights.pt', output / 'bundle/model.onnx']},
                  limitations='Float64 is a diagnostic reference, not exact arithmetic. Different kernels are not isolated individually. No production setting, weight or tolerance changed.')
    assert not result['comparison_to_torch']['onnx_default']['passed']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/fairscan-numerical-investigation.json')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = experiment()
    if args.check:
        assert result == json.loads(args.output.read_text())
    else:
        with args.output.open('x') as stream:
            stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['comparison_to_torch'], indent=2))

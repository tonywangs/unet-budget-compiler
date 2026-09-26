"""Frozen ONNX parity matrix and held-out replay; no training or tuning."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import torch

from unet_budget import compile_spec
from unet_budget.export import export_model, read_checkpoint
from unet_budget.onnx_inference import load_bundle, predict
from train_synthetic import CONFIG, dataset, digest, metrics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path(__file__).with_name('onnx_protocol.json')
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def comparison(reference, actual, *, require_close=True):
    assert reference.shape == actual.shape
    assert reference.dtype == actual.dtype == np.float32
    assert np.isfinite(reference).all() and np.isfinite(actual).all()
    error = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    if reference.shape[1] == 1:
        a, b = reference[:, 0] > 0, actual[:, 0] > 0
    else:
        a, b = reference.argmax(1), actual.argmax(1)
    tol = PROTOCOL['tolerances']
    bounds = tol['atol'] + tol['rtol'] * np.abs(reference.astype(np.float64))
    violations = error > bounds
    worst = np.unravel_index(np.argmax(error / bounds), error.shape)
    result = dict(shape=list(actual.shape), finite=True,
                  maximum_absolute_error=float(error.max()), mean_absolute_error=float(error.mean()),
                  label_disagreement_fraction=float(np.mean(a != b)),
                  elementwise_close=bool(not violations.any()),
                  elementwise_violation_count=int(violations.sum()),
                  maximum_error_to_elementwise_bound=float(np.max(error / bounds)),
                  worst_scaled_error=dict(index=[int(i) for i in worst], torch=float(reference[worst]),
                                          onnx=float(actual[worst]), absolute_error=float(error[worst]),
                                          allowed_error=float(bounds[worst])),
                  torch_logits_sha256=array_sha(reference), onnx_logits_sha256=array_sha(actual))
    result['within_tolerance'] = result['elementwise_close'] and all(result[key] <= tol[key] for key in
            ('maximum_absolute_error', 'mean_absolute_error', 'label_disagreement_fraction'))
    if require_close:
        assert result['within_tolerance'], result
    return result


def model_from_spec(spec):
    code, report = compile_spec(spec)
    namespace = {}
    exec(code, namespace)
    return namespace['UNet']().eval(), report


def experiment():
    torch.set_num_threads(PROTOCOL['threads'])
    torch.use_deterministic_algorithms(True)
    cases = []
    with tempfile.TemporaryDirectory(prefix='unet-onnx-parity-') as temp:
        temp = Path(temp)
        for index, case in enumerate(PROTOCOL['matrix']):
            n, c, h, w = case['shape']
            spec = dict(input_channels=c, output_classes=case['classes'], depth=case['depth'],
                        width_candidates=[case['width']], max_parameters=2_000_000,
                        input_height=h, input_width=w)
            for seed in PROTOCOL['weight_seeds']:
                torch.manual_seed(seed)
                model, report = model_from_spec(spec)
                checkpoint = temp/f'case-{index}-{seed}.pt'
                torch.save(model.state_dict(), checkpoint)
                image = np.random.default_rng(PROTOCOL['input_seed']).standard_normal(case['shape']).astype('float32')
                bundle = temp/f'bundle-{index}-{seed}'
                manifest = export_model(spec, checkpoint, case['shape'], bundle)
                session, contract = load_bundle(bundle)
                with torch.inference_mode():
                    reference = model(torch.from_numpy(image)).numpy()
                actual = predict(session, contract, image)
                result = comparison(reference, actual)
                cases.append(dict(case=case, weight_seed=seed, input_seed=PROTOCOL['input_seed'],
                                  input_sha256=array_sha(image), checkpoint_sha256=sha(checkpoint),
                                  weights_sha256=digest(*model.parameters()),
                                  model_sha256=manifest['artifacts']['model.onnx']['sha256'],
                                  generated_code_sha256=report['generated_code_sha256'], **result))
                print(f"Passed depth={case['depth']} channels={c} shape={case['shape']} seed={seed}: max error {result['maximum_absolute_error']:.3g}", flush=True)
        trained = PROTOCOL['trained_case']
        spec = json.loads((ROOT/trained['spec']).read_text())
        saved = json.loads((ROOT/'results/synthetic.json').read_text())
        images, targets, hashes = dataset(trained['evaluation_seed'], CONFIG['evaluation_samples'])
        assert list(images.shape) == trained['shape']
        assert digest(images, targets) == saved['data']['evaluation_sha256']
        assert hashes == saved['data']['evaluation_sample_sha256']
        model, _ = model_from_spec(spec)
        checkpoint_hash = read_checkpoint(ROOT/trained['checkpoint'], model, torch)
        assert digest(*model.parameters()) == saved['final_weights_sha256']
        bundle = temp/'trained'
        manifest = export_model(spec, ROOT/trained['checkpoint'], trained['shape'], bundle)
        session, contract = load_bundle(bundle)
        with torch.inference_mode():
            reference = model(images)
        assert digest(reference) == saved['held_out_logits_sha256']
        actual = predict(session, contract, images.numpy())
        scores = dict(torch=metrics(reference, targets), onnx=metrics(torch.from_numpy(actual), targets))
        assert scores['torch'] == saved['final_held_out']
        held_out = dict(**comparison(reference.numpy(), actual), metrics=scores,
                        checkpoint_sha256=checkpoint_hash, weights_sha256=digest(*model.parameters()),
                        evaluation_sha256=digest(images, targets), sample_sha256=hashes,
                        input_sha256=array_sha(images.numpy()), targets_sha256=array_sha(targets.numpy()),
                        model_sha256=manifest['artifacts']['model.onnx']['sha256'])
        print('Trained held-out replay passed, with original input/weight/logit hashes and both task metrics.', flush=True)
    return dict(schema_version=1, protocol=PROTOCOL, protocol_sha256=sha(PROTOCOL_PATH),
                versions={name: importlib.metadata.version(name) for name in
                          ('torch', 'numpy', 'onnx', 'onnxruntime', 'unet-budget-compiler')},
                environment=dict(python=platform.python_version(), platform=platform.platform()),
                random_cases=cases, held_out=held_out,
                source_hashes={name: sha(ROOT/name) for name in
                               ('experiments/compare_onnx.py', 'experiments/train_synthetic.py',
                                'src/unet_budget/export.py', 'src/unet_budget/onnx_inference.py')},
                limitations=['Synthetic inputs and random weights are not evidence of real-world accuracy.',
                             'Float32 fixed-shape CPU only; exact hashes are environment-specific.',
                             '24 random cases sample a bounded family, not every supported configuration.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/onnx-comparison.json')
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    result = experiment()
    if args.check:
        saved = json.loads(args.check.read_text())
        for key in ('schema_version', 'protocol', 'protocol_sha256', 'source_hashes', 'random_cases', 'held_out'):
            assert saved[key] == result[key], f'ONNX comparison reproduction mismatch: {key}'
        print('All 25 ONNX comparisons reproduced exactly.')
    else:
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
    print(json.dumps(result['held_out'], sort_keys=True))


if __name__ == '__main__':
    main()

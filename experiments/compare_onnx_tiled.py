"""Frozen tiled CPU parity cases and original checkpoint replay, without training."""
import argparse
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import torch

from unet_budget.export import export_model, read_checkpoint
from unet_budget.inference import tiled_logits as torch_tiled
from unet_budget.onnx_tiled import load_bundle, tiled_logits
from compare_onnx import array_sha, comparison, model_from_spec, sha, PROTOCOL as ORIGINAL_PROTOCOL
from compare_tiled import differences
from train_synthetic import CONFIG, dataset, digest, metrics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path(__file__).with_name('onnx_tiled_protocol.json')
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
# This is a pre-evaluation commitment, not a hash computed from mutable settings.
FROZEN_PROTOCOL_SHA256 = '3cea802d8c9e430ca2daa410f93bff6012e04bcc0cfc672934b2c543b73c6ce0'
SOURCES = ('experiments/compare_onnx_tiled.py', 'experiments/compare_onnx.py',
           'experiments/compare_tiled.py', 'experiments/train_synthetic.py',
           'src/unet_budget/onnx_tiled.py', 'src/unet_budget/inference.py', 'src/unet_budget/export.py')


def check_protocol():
    assert sha(PROTOCOL_PATH) == FROZEN_PROTOCOL_SHA256, 'frozen protocol changed'
    assert PROTOCOL['tolerances'] == ORIGINAL_PROTOCOL['tolerances']
    assert len(PROTOCOL['matrix']) >= 24


def experiment():
    check_protocol()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    records = []
    with tempfile.TemporaryDirectory(prefix='unet-onnx-tiled-') as temporary:
        temp = Path(temporary)
        for index, case in enumerate(PROTOCOL['matrix']):
            spec = dict(input_channels=case['input_channels'], output_classes=case['classes'],
                        depth=case['depth'], width_candidates=[case['width']], max_parameters=2_000_000,
                        input_height=case['tile'][0], input_width=case['tile'][1])
            torch.manual_seed(case['weight_seed'])
            model, report = model_from_spec(spec)
            checkpoint = temp/f'case-{index}.pt'
            torch.save(model.state_dict(), checkpoint)
            bundle = temp/f'bundle-{index}'
            shape = [1, case['input_channels'], *case['tile']]
            manifest = export_model(spec, checkpoint, shape, bundle)
            session, contract = load_bundle(bundle)
            image = np.random.default_rng(case['input_seed']).standard_normal(
                [1, case['input_channels'], *case['image']]).astype('float32')
            with torch.inference_mode():
                reference = torch_tiled(model, torch.from_numpy(image), tile_size=case['tile'],
                                        tile_batch_size=1, overlap=case['overlap'], blend=case['blend']).numpy()
            actual = tiled_logits(session, contract, image, overlap=case['overlap'], blend=case['blend'])
            record = dict(index=index, case=case, input_sha256=array_sha(image),
                          weights_sha256=digest(*model.parameters()), checkpoint_sha256=sha(checkpoint),
                          model_sha256=manifest['artifacts']['model.onnx']['sha256'],
                          generated_code_sha256=report['generated_code_sha256'],
                          **comparison(reference, actual, require_close=False))
            records.append(record)
            print(f"Case {index:02}: tolerance={record['within_tolerance']} max={record['maximum_absolute_error']:.3g}", flush=True)
        saved = json.loads((ROOT/'results/synthetic.json').read_text())
        trained = PROTOCOL['trained']
        assert trained['samples'] == CONFIG['evaluation_samples']
        images, targets, hashes = dataset(trained['evaluation_seed'], trained['samples'])
        assert digest(images, targets) == saved['data']['evaluation_sha256']
        assert hashes == saved['data']['evaluation_sample_sha256']
        spec = json.loads((ROOT/'examples/circles.json').read_text())
        model, _ = model_from_spec(spec)
        checkpoint = ROOT/'results/synthetic-weights.pt'
        read_checkpoint(checkpoint, model, torch)
        assert digest(*model.parameters()) == saved['final_weights_sha256']
        with torch.inference_mode():
            full = model(images)
        assert digest(full) == saved['held_out_logits_sha256']
        assert metrics(full, targets) == saved['final_held_out']
        bundle = temp/'trained'
        manifest = export_model(spec, checkpoint, [1, 1, *trained['tile']], bundle)
        session, contract = load_bundle(bundle)
        methods = []
        for method in trained['methods']:
            with torch.inference_mode():
                reference = torch.cat([torch_tiled(model, image[None], tile_size=trained['tile'],
                                                   tile_batch_size=1, **method) for image in images])
            actual = np.concatenate([tiled_logits(session, contract, image[None].numpy(), **method)
                                     for image in images])
            actual_tensor = torch.from_numpy(actual)
            methods.append(dict(settings=method, runtime_parity=comparison(reference.numpy(), actual, require_close=False),
                                runtime_parity_regions=differences(reference, actual_tensor, trained['tile'], method['overlap']),
                                torch_tiled_vs_whole=differences(full, reference, trained['tile'], method['overlap']),
                                onnx_tiled_vs_whole=differences(full, actual_tensor, trained['tile'], method['overlap']),
                                metrics=dict(torch_tiled=metrics(reference, targets), onnx_tiled=metrics(actual_tensor, targets))))
            print(f"Trained {method}: tolerance={methods[-1]['runtime_parity']['within_tolerance']}", flush=True)
        held_out = dict(checkpoint_sha256=sha(checkpoint), weights_sha256=digest(*model.parameters()),
                        input_sha256=array_sha(images.numpy()), targets_sha256=array_sha(targets.numpy()),
                        evaluation_sha256=digest(images, targets), sample_sha256=hashes,
                        full_logits_sha256=digest(full), full_metrics=metrics(full, targets),
                        model_sha256=manifest['artifacts']['model.onnx']['sha256'], methods=methods)
    import importlib.metadata
    return dict(schema_version=1, protocol=PROTOCOL, protocol_sha256=sha(PROTOCOL_PATH), random_cases=records,
                held_out=held_out, source_hashes={name: sha(ROOT/name) for name in SOURCES},
                environment=dict(python=platform.python_version(), platform=platform.platform(),
                                 versions={name: importlib.metadata.version(name) for name in ('numpy','torch','onnx','onnxruntime')}),
                tolerance_failures=[f'random:{r["index"]}' for r in records if not r['within_tolerance']] +
                                   [f'trained:{i}' for i, r in enumerate(methods) if not r['runtime_parity']['within_tolerance']],
                limitations=['Bounded synthetic CPU cases; no real-world or medical accuracy evidence.',
                             'Tiled predictions need not equal whole-image predictions; those differences are not runtime failures.',
                             'Exact hashes require recorded dependencies and platform. No retraining in this experiment.',
                             'Historical fixed-shape 129x161 tolerance failure remains recorded separately.'])


def validate(result):
    check_protocol()
    assert result['protocol'] == PROTOCOL and result['protocol_sha256'] == FROZEN_PROTOCOL_SHA256
    assert len(result['random_cases']) == len(PROTOCOL['matrix'])
    assert not result['tolerance_failures'], 'investigate every new tolerance failure; never relax the protocol'
    for record in result['random_cases'] + [r['runtime_parity'] for r in result['held_out']['methods']]:
        assert record['within_tolerance'] and record['elementwise_violation_count'] == 0
        for key in ('maximum_absolute_error', 'mean_absolute_error', 'label_disagreement_fraction'):
            assert record[key] <= PROTOCOL['tolerances'][key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/onnx-tiled-comparison.json')
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    result = experiment()
    if args.check:
        saved = json.loads(args.check.read_text())
        validate(saved)
        for key in ('schema_version','protocol','protocol_sha256','random_cases','held_out','source_hashes','tolerance_failures'):
            assert result[key] == saved[key], f'tiled parity reproduction mismatch: {key}'
    else:
        # Preserve negatives before validation fails.
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    validate(result)
    print('24 frozen tiled parity cases and three 16-sample checkpoint replays passed; no retraining.')


if __name__ == '__main__':
    main()

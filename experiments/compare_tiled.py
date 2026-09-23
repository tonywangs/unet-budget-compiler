"""Reproducible CPU logit comparisons and held-out segmentation experiment."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform

import torch

from unet_budget import compile_spec
from unet_budget.inference import tiled_logits
from train_synthetic import digest, experiment, metrics

ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(weight_seeds=[17, 41, 93], input_seed=5701, depths=[1, 2, 3],
              shapes=[[31, 37], [48, 64], [65, 53]], tiles=[[16, 20], [25, 24]],
              overlap_fractions=[0, 0.25, 0.5], blends=['constant', 'gaussian'],
              base_width=3, input_channels=2, output_classes=3, tile_batch_size=3,
              boundary_band_pixels=2, threads=1, dtype='float32', device='cpu')


def boundary_mask(shape, tile, overlap, band=2):
    """Union of +/-band pixels around internal tile starts and clipped ends.

    Excludes the outer image perimeter unless it also borders an internal edge.
    A pixel at boundary b is selected when b-band <= coordinate < b+band.
    """
    h, w = shape
    mask = torch.zeros(h, w, dtype=torch.bool)
    for axis, (length, side, ov) in enumerate(zip(shape, tile, overlap)):
        start = 0
        edges = set()
        while True:
            edges.update((start, min(start + side, length)))
            if start + side >= length:
                break
            start += side - ov
        for edge in edges:
            if 0 < edge < length:
                a, b = max(0, edge-band), min(length, edge+band)
                if axis == 0:
                    mask[a:b, :] = True
                else:
                    mask[:, a:b] = True
    return mask


def differences(full, tiled, tile, overlap):
    error = (full - tiled).abs()
    boundary = boundary_mask(full.shape[-2:], tile, overlap)
    def region(mask):
        selected = error[..., mask]
        disagreement = (full.argmax(1) != tiled.argmax(1))[..., mask]
        return dict(pixels_per_image=int(mask.sum()),
                    mean_absolute_error=selected.mean().item() if selected.numel() else None,
                    maximum_absolute_error=selected.max().item() if selected.numel() else None,
                    label_disagreement=disagreement.float().mean().item() if selected.numel() else None)
    return dict(mean_absolute_error=error.mean().item(), maximum_absolute_error=error.max().item(),
                label_disagreement=(full.argmax(1) != tiled.argmax(1)).float().mean().item(),
                boundary=region(boundary), interior=region(~boundary))


def generated_comparisons():
    records = []
    exact_cases = []
    for seed in CONFIG['weight_seeds']:
        for depth in CONFIG['depths']:
            raw = dict(input_channels=2, output_classes=3, depth=depth, max_parameters=10**7,
                       width_candidates=[3], input_height=32, input_width=32)
            code, report = compile_spec(raw)
            namespace = {}
            exec(code, namespace)
            torch.manual_seed(seed)
            model = namespace['UNet']().eval()
            weights_hash = digest(*model.parameters())
            for h, w in CONFIG['shapes']:
                image = torch.randn(1, 2, h, w, generator=torch.Generator().manual_seed(CONFIG['input_seed']))
                full = model(image)
                single = tiled_logits(model, image, tile_size=(h, w), overlap=(0, 0))
                torch.testing.assert_close(full, single, rtol=0, atol=0)
                exact_cases.append(dict(seed=seed, depth=depth, shape=[h, w], equal=True))
                for tile in CONFIG['tiles']:
                    for fraction in CONFIG['overlap_fractions']:
                        overlap = [int(side*fraction) for side in tile]
                        for mode in CONFIG['blends']:
                            tiled = tiled_logits(model, image, tile_size=tile, overlap=overlap,
                                                 blend=mode, tile_batch_size=CONFIG['tile_batch_size'])
                            records.append(dict(seed=seed, depth=depth, shape=[h, w], tile=tile,
                                                overlap=overlap, blend=mode, weights_sha256=weights_hash,
                                                specification=raw, code_sha256=report['generated_code_sha256'],
                                                input_sha256=digest(image), full_logits_sha256=digest(full),
                                                tiled_logits_sha256=digest(tiled),
                                                errors=differences(full, tiled, tile, overlap)))
    assert any(r['overlap'] != [0, 0] and r['errors']['maximum_absolute_error'] > 1e-5 for r in records)
    return dict(config=CONFIG, single_tile_exact_cases=exact_cases, cases=records)


def trained_comparisons(checkpoint_path, check):
    training, model, images, targets = experiment(return_model=True)
    original = json.loads((ROOT/'results/synthetic.json').read_text())
    for key in ('data', 'loss_curve', 'final_weights_sha256', 'held_out_logits_sha256', 'final_held_out'):
        assert training[key] == original[key], f'changed original training: {key}'
    with torch.inference_mode():
        full = model(images)
        methods = [dict(name='nonoverlapping', tile_size=[16, 20], overlap=[0, 0], blend='constant'),
                   dict(name='overlap_constant', tile_size=[16, 20], overlap=[8, 10], blend='constant'),
                   dict(name='overlap_gaussian', tile_size=[16, 20], overlap=[8, 10], blend='gaussian')]
        results = [dict(name='full', held_out=metrics(full, targets), logits_sha256=digest(full))]
        for method in methods:
            kwargs = {k: v for k, v in method.items() if k != 'name'}
            tiled = tiled_logits(model, images, **kwargs, tile_batch_size=4)
            results.append(dict(**method, tile_batch_size=4, held_out=metrics(tiled, targets),
                                logits_sha256=digest(tiled),
                                errors=differences(full, tiled, method['tile_size'], method['overlap'])))
    # File hash identifies the saved artifact; tensor hash permits format-independent reproduction.
    if not check:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        checkpoint_path.write_bytes(buffer.getvalue())
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert list(state) == list(model.state_dict())
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(state[key], tensor, rtol=0, atol=0)
    return dict(training=training, checkpoint=dict(file=checkpoint_path.name,
                file_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                tensor_sha256=digest(*state.values()), tensor_order=list(state)), methods=results,
                limitations=['Same easy synthetic circles and one training seed as original experiment.',
                             'No hyperparameter selection on held-out inputs; no real-world generalization claim.',
                             'Tiled methods share exactly the full-image trained weights; no tile-specific retraining.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/tiled-comparison.json')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'results/synthetic-weights.pt')
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # Training enables gradients internally, so do it before the inference block.
    trained = trained_comparisons(args.checkpoint, bool(args.check))
    with torch.inference_mode():
        comparisons = generated_comparisons()
    result = dict(schema_version=1, torch=torch.__version__, python=platform.python_version(),
                  generated=comparisons, trained=trained)
    if args.check:
        saved = json.loads(args.check.read_text())
        assert result == saved, 'comparison mismatch; use recorded dependencies and platform'
        print('Tiled comparisons reproduced exactly, including training, checkpoint tensors, metrics and logits.')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(generated_cases=len(comparisons['cases']),
                         single_tile_exact=len(comparisons['single_tile_exact_cases']),
                         held_out={r['name']: r['held_out'] for r in trained['methods']}), sort_keys=True))


if __name__ == '__main__':
    main()

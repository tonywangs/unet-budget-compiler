"""Bounded CPU demonstration, using synthetic noisy filled circles only.

No private or real-world data. Evaluation samples are never used by the optimizer.
The fixed configuration is not selected using held-out scores.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import torch
from torch.nn import functional as F

from unet_budget import compile_spec

ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(train_seed=101, evaluation_seed=202, model_seed=303, shuffle_seed=404,
              train_samples=48, evaluation_samples=16, epochs=24, batch_size=8,
              learning_rate=0.01, height=33, width=41, noise_std=0.20,
              optimizer='Adam', threads=1, dtype='float32', device='cpu')


def digest(*tensors):
    h = hashlib.sha256()
    for tensor in tensors:
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(str(arr.dtype).encode())
        h.update(json.dumps(list(arr.shape)).encode())
        h.update(arr.tobytes(order='C'))
    return h.hexdigest()


def dataset(seed, count):
    generator = torch.Generator().manual_seed(seed)
    h, w = CONFIG['height'], CONFIG['width']
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    masks = []
    for _ in range(count):
        radius = int(torch.randint(4, 10, (), generator=generator))
        cy = int(torch.randint(radius, h-radius, (), generator=generator))
        cx = int(torch.randint(radius, w-radius, (), generator=generator))
        masks.append(((xx-cx)**2 + (yy-cy)**2 <= radius**2).long())
    targets = torch.stack(masks)
    images = targets.float().unsqueeze(1) + CONFIG['noise_std'] * torch.randn(
        count, 1, h, w, generator=generator)
    hashes = [digest(x, y) for x, y in zip(images, targets)]
    return images, targets, hashes


def metrics(logits, target):
    prediction = logits.argmax(1)
    foreground, truth = prediction == 1, target == 1
    intersection = (foreground & truth).sum().item()
    union = (foreground | truth).sum().item()
    total = foreground.sum().item() + truth.sum().item()
    return dict(cross_entropy=F.cross_entropy(logits, target).item(),
                pixel_accuracy=(prediction == target).float().mean().item(),
                foreground_iou=intersection / union if union else 1.0,
                foreground_dice=2 * intersection / total if total else 1.0)


def experiment(*, return_model=False):
    torch.set_num_threads(CONFIG['threads'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(CONFIG['model_seed'])
    train_x, train_y, train_hashes = dataset(CONFIG['train_seed'], CONFIG['train_samples'])
    eval_x, eval_y, eval_hashes = dataset(CONFIG['evaluation_seed'], CONFIG['evaluation_samples'])
    assert not set(train_hashes) & set(eval_hashes)
    assert len(set(train_hashes + eval_hashes)) == len(train_hashes + eval_hashes)
    raw = json.loads((ROOT / 'examples/circles.json').read_text())
    code, architecture = compile_spec(raw)
    namespace = {}
    exec(compile(code, '<synthetic-unet>', 'exec'), namespace)
    model = namespace['UNet']()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    shuffle = torch.Generator().manual_seed(CONFIG['shuffle_seed'])
    with torch.no_grad():
        initial = metrics(model(eval_x), eval_y)
    losses = []
    for epoch in range(CONFIG['epochs']):
        model.train()
        order = torch.randperm(len(train_x), generator=shuffle)
        loss_sum = 0.0
        for indices in order.split(CONFIG['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(train_x[indices]), train_y[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite training loss at epoch {epoch+1}')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        losses.append(dict(epoch=epoch+1, mean_training_cross_entropy=loss_sum/len(train_x)))
    model.eval()
    with torch.no_grad():
        predictions = model(eval_x)
        final = metrics(predictions, eval_y)
        prevalence = torch.bincount(train_y.flatten(), minlength=2).float() / train_y.numel()
        constant_logits = prevalence.clamp_min(1e-8).log()[None, :, None, None].expand(len(eval_x), 2, *eval_y.shape[-2:])
        baseline = metrics(constant_logits, eval_y)
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'numpy', 'unet-budget-compiler')}
    result = dict(config=CONFIG, versions=versions, python=platform.python_version(),
                platform=platform.platform(), machine=platform.machine(),
                specification=raw, selection=architecture['selection'],
                specification_sha256=architecture['specification_sha256'],
                generated_code_sha256=architecture['generated_code_sha256'],
                data=dict(description='Synthetic filled circles plus Gaussian pixel noise; no real images',
                          hash_format='SHA256 of dtype string + JSON shape + C-order raw bytes, for each tensor',
                          train_sha256=digest(train_x, train_y), evaluation_sha256=digest(eval_x, eval_y),
                          train_sample_sha256=train_hashes, evaluation_sample_sha256=eval_hashes,
                          disjoint_sample_hashes=True),
                loss_curve=losses, initial_held_out=initial, final_held_out=final,
                constant_baseline=dict(description='Training-set class frequencies, same probabilities at every pixel',
                                       class_probabilities=prevalence.tolist(), held_out=baseline),
                final_weights_sha256=digest(*list(model.parameters())),
                held_out_logits_sha256=digest(predictions),
                limitations=['Single seed and easy synthetic data; no claim about real-world accuracy.',
                             'CPU only; reproducibility across PyTorch versions and platforms is not guaranteed.',
                             'Width maximizes a parameter-budget criterion, not accuracy or compute efficiency.'])
    return (result, model, eval_x, eval_y) if return_model else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/synthetic.json')
    parser.add_argument('--check', type=Path, help='rerun and compare numerical results and hashes with saved JSON')
    args = parser.parse_args()
    result = experiment()
    if args.check:
        saved = json.loads(args.check.read_text())
        for field in ('config', 'specification_sha256', 'generated_code_sha256', 'data', 'loss_curve',
                      'initial_held_out', 'final_held_out', 'constant_baseline', 'final_weights_sha256',
                      'held_out_logits_sha256'):
            if result[field] != saved[field]:
                raise AssertionError(f'reproducibility mismatch: {field}; use the recorded dependency versions and platform')
        print('Synthetic experiment reproduced exactly (losses, metrics, data, weights, logits).')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'final_held_out': result['final_held_out'],
                      'constant_baseline': result['constant_baseline']['held_out']}, sort_keys=True))


if __name__ == '__main__':
    main()

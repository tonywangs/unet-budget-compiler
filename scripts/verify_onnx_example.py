"""Generate or byte-verify the trained ONNX example and its held-out sample."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

from unet_budget.export import export_model
from unet_budget.onnx_inference import load_bundle, predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
from train_synthetic import CONFIG, dataset, digest


def generate(out):
    torch.set_num_threads(1)
    saved = json.loads((ROOT/'results/synthetic.json').read_text())
    images, targets, hashes = dataset(CONFIG['evaluation_seed'], CONFIG['evaluation_samples'])
    assert digest(images, targets) == saved['data']['evaluation_sha256']
    assert hashes == saved['data']['evaluation_sample_sha256']
    out.mkdir()
    export_model(json.loads((ROOT/'examples/circles.json').read_text()), ROOT/'results/synthetic-weights.pt',
                 [1, 1, 33, 41], out/'bundle')
    np.save(out/'input.npy', images[:1].numpy(), allow_pickle=False)
    np.save(out/'target.npy', targets[:1].numpy(), allow_pickle=False)
    session, manifest = load_bundle(out/'bundle')
    logits = predict(session, manifest, images[:1].numpy())
    reference = dict(description='First unchanged held-out synthetic sample; foreground class 1',
                     dataset_sample_sha256=hashes[0], checkpoint_sha256=manifest['checkpoint_sha256'],
                     input_file_sha256=hashlib.sha256((out/'input.npy').read_bytes()).hexdigest(),
                     target_file_sha256=hashlib.sha256((out/'target.npy').read_bytes()).hexdigest(),
                     logits_bytes_sha256=hashlib.sha256(logits.tobytes()).hexdigest())
    (out/'sample.json').write_text(json.dumps(reference, indent=2, sort_keys=True)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='generate into a new directory instead of checking')
    args = parser.parse_args()
    if args.output:
        generate(args.output)
    else:
        with tempfile.TemporaryDirectory(prefix='unet-onnx-example-') as temporary:
            out = Path(temporary)/'example'
            generate(out)
            expected = ROOT/'examples/onnx'
            files = sorted(p.relative_to(out) for p in out.rglob('*') if p.is_file())
            assert files == sorted(p.relative_to(expected) for p in expected.rglob('*') if p.is_file())
            for relative in files:
                assert (out/relative).read_bytes() == (expected/relative).read_bytes(), relative
        print('Trained ONNX example, standalone source, manifests and held-out input reproduced byte-for-byte.')


if __name__ == '__main__':
    main()

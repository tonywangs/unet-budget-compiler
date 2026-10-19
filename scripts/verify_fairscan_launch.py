"""Exercise the documented compile/export/standalone CLI with fresh outputs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def main():
    saved = ROOT / 'results/fairscan'
    with tempfile.TemporaryDirectory(prefix='fairscan-launch-') as temp:
        temp = Path(temp)
        compiled, bundle, logits = temp / 'compiled', temp / 'bundle', temp / 'logits.npy'
        run('-m', 'unet_budget', 'examples/fairscan-v2.json', '--out', compiled)
        run('-m', 'unet_budget', 'export', 'examples/fairscan-v2.json', '--checkpoint',
            saved / 'weights.pt', '--shape', '1', '3', '65', '81', '--out', bundle)
        for name in ['model.py', 'architecture.json']:
            assert (compiled / name).read_bytes() == (bundle / name).read_bytes()
            assert (bundle / name).read_bytes() == (saved / 'bundle' / name).read_bytes()
        run('src/unet_budget/onnx_tiled.py', bundle, saved / 'input.npy', '--output', logits,
            '--overlap', '32', '40', '--blend', 'gaussian', '--threads', '1')
        actual = np.load(logits, allow_pickle=False)
        expected = np.load(saved / 'logits.npy', allow_pickle=False)
        assert actual.shape == (1, 2, 129, 161) and actual.dtype == np.float32
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        assert np.all(error <= 1e-5 + 1e-5 * np.abs(expected))
        assert error.max() <= 1e-4 and error.mean() <= 1e-5
        assert (actual.argmax(1) != expected.argmax(1)).mean() <= 0.001
        manifest = json.loads((bundle / 'manifest.json').read_text())
        assert manifest['input']['shape'] == [1, 3, 65, 81]
    print('Fresh budget compilation, checkpoint export and larger-real-image CLI passed.')


if __name__ == '__main__':
    main()

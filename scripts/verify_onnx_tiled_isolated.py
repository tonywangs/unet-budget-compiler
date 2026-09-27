"""Exercise the documented example offline with only deployment distributions."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile
import venv

from verify_onnx_isolated import run

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='unet-onnx-tiled-isolated-') as temporary:
        temp = Path(temporary)
        venv.EnvBuilder(with_pip=False).create(temp/'runtime-env')
        dependencies = temp/'dependencies'
        dependencies.mkdir()
        for distribution in ('numpy', 'onnxruntime', 'coloredlogs', 'flatbuffers',
                             'humanfriendly', 'packaging', 'sympy', 'mpmath'):
            dist = importlib.metadata.distribution(distribution)
            for name in sorted({str(f).split('/')[0] for f in dist.files}):
                if name in ('.', '..'):
                    continue
                source = Path(dist.locate_file(name))
                target = dependencies/name
                if source.exists() and not target.exists():
                    target.symlink_to(source, target_is_directory=source.is_dir())
        shutil.copytree(ROOT/'examples/onnx/bundle', temp/'bundle')
        shutil.copyfile(ROOT/'src/unet_budget/onnx_tiled.py', temp/'tiled_inference.py')
        shutil.copyfile(ROOT/'examples/onnx/input.npy', temp/'input.npy')
        for name in ('tiled-input.npy', 'tiled-sample.json'):
            shutil.copyfile(ROOT/'examples/onnx-tiled'/name, temp/name)
        script = '''
import hashlib, importlib.util, json, sys
from pathlib import Path
sys.path.append(sys.argv[1])
sys.addaudithook(lambda event, args: (_ for _ in ()).throw(RuntimeError('network forbidden')) if event.startswith('socket.') else None)
for name in ('torch', 'onnx', 'unet_budget'):
    assert importlib.util.find_spec(name) is None, name
import numpy as np
metadata = json.loads(Path('tiled-sample.json').read_text())
source = np.load('input.npy', allow_pickle=False)
x = np.load('tiled-input.npy', allow_pickle=False)
np.testing.assert_array_equal(x, np.tile(source, (1, 1, 2, 2))[..., :65, :81])
for name, expected in metadata['artifacts'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
module_spec = importlib.util.spec_from_file_location('standalone_tiled', 'tiled_inference.py')
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
assert module.main(['bundle', 'tiled-input.npy', '--output', 'logits.npy', '--overlap', '16', '20', '--blend', 'gaussian']) == 0
y = np.load('logits.npy', allow_pickle=False)
assert list(y.shape) == metadata['output_shape']
assert y.dtype == np.float32 and np.isfinite(y).all()
assert hashlib.sha256(y.tobytes()).hexdigest() == metadata['output_sha256']
assert module.main(['bundle', 'tiled-input.npy', '--output', 'logits.npy']) == 2
session, manifest = module.load_bundle('bundle')
np.testing.assert_array_equal(y, module.tiled_logits(session, manifest, x, overlap=(16, 20), blend='gaussian'))
for name in ('torch', 'onnx', 'unet_budget'):
    assert name not in sys.modules
print(json.dumps(dict(offline=True, shape=list(y.shape), provider=session.get_providers(),
                     input_shape=list(x.shape), tile_shape=manifest['input']['shape'],
                     torch_importable=False, compiler_importable=False, onnx_importable=False)))
'''
        print(run([temp/'runtime-env/bin/python', '-I', '-S', '-c', script, dependencies], temp))
    print('Documented tiled example passed in isolated offline NumPy/ORT-only deployment.')


if __name__ == '__main__':
    main()

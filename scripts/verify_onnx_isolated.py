"""Offline wheel export, then NumPy/ORT-only inference in a separate environment."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]


def run(command, cwd):
    result = subprocess.run([str(x) for x in command], cwd=cwd, check=True,
                            text=True, capture_output=True, timeout=180,
                            env={**os.environ, 'PIP_NO_INDEX': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
                                 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    return result.stdout.strip()


def main():
    with tempfile.TemporaryDirectory(prefix='unet-onnx-isolated-') as temporary:
        temp = Path(temporary)
        wheels = temp/'wheels'
        run([sys.executable, '-m', 'pip', 'wheel', ROOT, '--no-build-isolation', '--no-deps',
             '--no-index', '-w', wheels], temp)
        compiler_env = temp/'export-env'
        venv.EnvBuilder(with_pip=False).create(compiler_env)
        python = compiler_env/'bin/python'
        wheel, = wheels.glob('*.whl')
        run([sys.executable, '-m', 'pip', '--python', python, 'install', '--no-index', '--no-deps', wheel], temp)
        audit = "sys.addaudithook(lambda event, args: (_ for _ in ()).throw(RuntimeError('network forbidden')) if event.startswith('socket.') else None)"
        missing = '''
import importlib.util, sys
assert importlib.util.find_spec('torch') is None
from unet_budget.cli import main
assert main(['export', sys.argv[1], '--checkpoint', sys.argv[2], '--shape', '1', '1', '33', '41', '--out', sys.argv[3]]) == 2
from pathlib import Path
assert not Path(sys.argv[3]).exists()
print('Isolated compiler reports missing optional dependencies without creating output.')
'''
        spec, checkpoint, bundle = ROOT/'examples/circles.json', ROOT/'results/synthetic-weights.pt', temp/'bundle'
        print(run([python, '-I', '-c', missing, spec, checkpoint, bundle], temp))
        export_deps = temp/'export-dependencies'
        export_deps.mkdir()
        for entry in Path(sysconfig.get_path('purelib')).iterdir():
            if entry.name.startswith(('unet_budget', '__editable__')) or entry.suffix == '.pth':
                continue
            (export_deps/entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        export_script = '''
import sys
sys.path.append(sys.argv[1])
AUDIT
import torch
from unet_budget.cli import main
import unet_budget
assert 'export-env' in unet_budget.__file__, unet_budget.__file__
torch.set_num_threads(1)
assert main(['export', sys.argv[2], '--checkpoint', sys.argv[3], '--shape', '1', '1', '33', '41', '--out', sys.argv[4]]) == 0
'''.replace('AUDIT', audit)
        print(run([python, '-I', '-c', export_script, export_deps, spec, checkpoint, bundle], temp))
        runtime_env = temp/'runtime-env'
        venv.EnvBuilder(with_pip=False).create(runtime_env)
        runtime_deps = temp/'runtime-dependencies'
        runtime_deps.mkdir()
        # Provision only the runtime distributions from local installed files.
        # This view is offline and has neither torch, onnx, nor the compiler.
        for distribution in ('numpy', 'onnxruntime', 'coloredlogs', 'flatbuffers',
                             'humanfriendly', 'packaging', 'sympy', 'mpmath'):
            dist = importlib.metadata.distribution(distribution)
            for name in sorted({str(f).split('/')[0] for f in dist.files}):
                if name in ('.', '..'):
                    continue
                source = Path(dist.locate_file(name))
                target = runtime_deps/name
                if source.exists() and not target.exists():
                    target.symlink_to(source, target_is_directory=source.is_dir())
        runtime_script = '''
import importlib.util, json, sys
sys.path.append(sys.argv[1])
AUDIT
for name in ('torch', 'unet_budget', 'onnx'):
    assert importlib.util.find_spec(name) is None, (name, sys.path)
import numpy as np
spec = importlib.util.spec_from_file_location('standalone', sys.argv[2] + '/inference.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
session, manifest = helper.load_bundle(sys.argv[2])
x = np.random.default_rng(123).normal(size=(1, 1, 33, 41)).astype('float32')
y = helper.predict(session, manifest, x)
assert y.shape == (1, 2, 33, 41) and np.isfinite(y).all()
np.save('input.npy', x)
assert helper.main([sys.argv[2], 'input.npy', '--output', 'output.npy']) == 0
np.testing.assert_array_equal(np.load('output.npy'), y)
for name in ('torch', 'unet_budget', 'onnx'):
    assert name not in sys.modules
print(json.dumps({'offline': True, 'compiler_importable': False, 'torch_importable': False,
                  'onnx_importable': False, 'provider': session.get_providers(), 'shape': list(y.shape)}))
'''.replace('AUDIT', audit)
        print(run([runtime_env/'bin/python', '-I', '-S', '-c', runtime_script, runtime_deps, bundle], temp))
    print('Offline installed-wheel export and separate NumPy/ORT-only runtime passed.')


if __name__ == '__main__':
    main()

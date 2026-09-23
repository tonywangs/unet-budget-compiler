"""Build/install offline, compile without torch, run output without the compiler."""
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]


def run(args, cwd):
    result = subprocess.run([str(x) for x in args], cwd=cwd, check=True,
                            text=True, capture_output=True,
                            env={**os.environ, 'PIP_NO_INDEX': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1'})
    return result.stdout.strip()


def main():
    with tempfile.TemporaryDirectory(prefix='unet-isolated-') as temp:
        temp = Path(temp)
        wheels = temp / 'wheels'
        run([sys.executable, '-m', 'pip', 'wheel', ROOT, '--no-build-isolation',
             '--no-deps', '--no-index', '-w', wheels], temp)
        isolated = temp / 'compiler-env'
        venv.EnvBuilder(with_pip=False).create(isolated)
        python = isolated / 'bin' / 'python'
        wheel, = wheels.glob('*.whl')
        run([sys.executable, '-m', 'pip', '--python', python, 'install',
             '--no-index', '--no-deps', wheel], temp)
        assert run([isolated / 'bin' / 'unet-budget', '--version'], temp) == '0.1.0'
        # Audit hook makes socket connection attempts fail even on an online host.
        compile_check = '''
import importlib.util, runpy, sys
assert importlib.util.find_spec('torch') is None
sys.addaudithook(lambda event, args: (_ for _ in ()).throw(RuntimeError('network forbidden')) if event.startswith('socket.') else None)
sys.argv = ['unet-budget', sys.argv[1], '--out', sys.argv[2], '--with-inference']
runpy.run_module('unet_budget', run_name='__main__')
'''
        print(run([python, '-I', '-c', compile_check, ROOT/'examples/circles.json', temp/'artifact'], temp))
        # Make a dependency-only view that also works when the parent has a
        # non-editable compiler installation. Never expose its package or .pth
        # hooks, and never copy the large PyTorch installation.
        dependencies = temp / 'runtime-dependencies'
        dependencies.mkdir()
        for entry in Path(sysconfig.get_path('purelib')).iterdir():
            if entry.name.startswith(('unet_budget', '__editable__')) or entry.suffix == '.pth':
                continue
            (dependencies / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        # -I -S ignores cwd, PYTHONPATH, user site and all .pth hooks.
        # No src directory or compiler package is accessible.
        runtime = '''
import importlib.util, json, socket, sys
sys.path.append(sys.argv[1])
assert importlib.util.find_spec('unet_budget') is None, sys.path
sys.addaudithook(lambda event, args: (_ for _ in ()).throw(RuntimeError('network forbidden')) if event.startswith('socket.') else None)
import torch
spec = importlib.util.spec_from_file_location('standalone_model', sys.argv[2])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(1)
torch.manual_seed(17)
model = module.UNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
before = model.head.weight.detach().clone()
x = torch.randn(2, 1, 33, 41, requires_grad=True)
y = model(x)
assert list(y.shape) == [2, 2, 33, 41]
loss = torch.nn.functional.cross_entropy(y, torch.zeros(2, 33, 41, dtype=torch.long))
loss.backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
optimizer.step()
assert not torch.equal(before, model.head.weight)
helper_spec = importlib.util.spec_from_file_location('standalone_inference', sys.argv[3])
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)
assert helper.MIN_TILE_SIZE == 4 and helper.INPUT_CHANNELS == 1 and helper.OUTPUT_CHANNELS == 2
model.eval()
image = torch.randn(1, 1, 65, 81)
with torch.inference_mode():
    full = model(image)
    for blend in ('constant', 'gaussian'):
        tiled = helper.tiled_logits(model, image, tile_size=(32, 40), overlap=(16, 20),
                                    tile_batch_size=2, blend=blend)
        labels = tiled.argmax(dim=1)
        assert tuple(full.shape) == tuple(tiled.shape) == (1, 2, 65, 81)
        assert tuple(labels.shape) == (1, 65, 81)
        assert torch.isfinite(tiled).all()
    single = helper.tiled_logits(model, image, tile_size=(65, 81), overlap=(0, 0))
    torch.testing.assert_close(single, full, rtol=0, atol=0)
    tiny = helper.tiled_logits(model, image[:, :, :1, :3], tile_size=(8, 12), overlap=(0, 0))
    assert tuple(tiny.shape) == (1, 2, 1, 3)
assert importlib.util.find_spec('unet_budget') is None
print('Standalone full-image, both tiled blends, single-tile equality and small-image padding passed offline.')
print(json.dumps({'standalone_forward_backward_optimizer': 'passed', 'compiler_importable': False, 'torch': torch.__version__, 'parameters': sum(p.numel() for p in model.parameters())}, sort_keys=True))
'''
        print(run([sys.executable, '-I', '-S', '-c', runtime, dependencies,
                   temp/'artifact/model.py', temp/'artifact/inference.py'], temp))
    print('Offline wheel installation and isolated execution passed.')


if __name__ == '__main__':
    main()

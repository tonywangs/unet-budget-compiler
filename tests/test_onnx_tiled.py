"""Independent geometric oracle, runtime contracts and transactional CLI checks."""
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from unet_budget import onnx_tiled as runtime

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'examples/onnx/bundle'
PROTOCOL = json.loads((ROOT/'experiments/onnx_tiled_protocol.json').read_text())


def contract(tile=(5, 7), channels=2, classes=2):
    return dict(schema_version=1, format='unet-budget-onnx', precision='float32', opset=17,
                provider='CPUExecutionProvider',
                input=dict(name='image', shape=[1, channels, *tile], dtype='float32', layout='NCHW'),
                output=dict(name='logits', shape=[1, classes, *tile], dtype='float32', layout='NCHW'))


class Session:
    def __init__(self, function):
        self.function = function
        self.calls = []

    def run(self, names, feeds):
        assert names == ['logits'] and set(feeds) == {'image'}
        self.calls.append(feeds['image'].copy())
        return [self.function(feeds['image'])]


def oracle(image, tile, overlap, blend):
    """Scalar pixel gather in float64; no runtime grid/weights/stitch functions.

    Local-coordinate predictor mixes source values and local position so that an
    identity-only oracle cannot conceal wrong placement, padding or normalization.
    """
    h, w = image.shape[-2:]
    th, tw = tile
    starts = []
    for length, side, ov in zip((h, w), tile, overlap):
        values = [0]
        while values[-1] + side < length:
            values.append(values[-1] + side - ov)
        starts.append(values)
    peak = max(math.exp(-0.5*(((a-(th-1)/2)/(th/8))**2 + ((b-(tw-1)/2)/(tw/8))**2))
               for a in range(th) for b in range(tw))
    result = np.zeros_like(image, dtype=np.float64)
    counts = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            values, weights = [], []
            for sy in starts[0]:
                for sx in starts[1]:
                    a, b = y-sy, x-sx
                    if 0 <= a < th and 0 <= b < tw:
                        weight = 1 if blend == 'constant' else max(1e-6, math.exp(-0.5*(
                            ((a-(th-1)/2)/(th/8))**2 + ((b-(tw-1)/2)/(tw/8))**2))/peak)
                        # Include next local pixel to expose zero padding at tile boundaries.
                        neighbor = image[0, 0, y, x+1] if b+1 < tw and x+1 < w else 0
                        values.append([image[0, 0, y, x]+a/16+b/32+neighbor/4,
                                       image[0, 1, y, x]-a/32+b/16])
                        weights.append(weight)
            counts[y, x] = len(values)
            result[0, :, y, x] = np.average(values, axis=0, weights=weights)
    assert counts.min() > 0
    return result, starts


def position_predictor(tile):
    th, tw = tile.shape[-2:]
    a, b = np.indices((th, tw), dtype=np.float32)
    out = tile.copy()
    neighbor = np.zeros((th, tw), dtype=np.float32)
    neighbor[:, :-1] = tile[0, 0, :, 1:]
    out[0, 0] += a/16+b/32+neighbor/4
    out[0, 1] += -a/32+b/16
    return out


class GeometryTests(unittest.TestCase):
    def test_256_seeded_geometries_both_blends(self):
        config = PROTOCOL['geometry']
        rng = np.random.default_rng(config['seed'])
        fixed = [(1, 1, 5, 7), (1, 29, 5, 7), (29, 1, 5, 7), (5, 7, 5, 7),
                 (10, 14, 5, 7), (11, 15, 5, 7), (3, 4, 5, 7), (17, 23, 1, 1)]
        for index in range(config['cases']):
            h, w, th, tw = fixed[index] if index < len(fixed) else (
                int(rng.integers(1, 32)), int(rng.integers(1, 32)),
                int(rng.integers(1, 13)), int(rng.integers(1, 13)))
            overlap = [int(rng.integers(0, th)), int(rng.integers(0, tw))]
            # Integer / 16 coordinates have exact binary representation.
            image = (rng.integers(-32, 33, size=(1, 2, h, w))/16).astype(np.float32)
            original = image.copy()
            for blend in ('constant', 'gaussian'):
                with self.subTest(index=index, blend=blend):
                    expected, starts = oracle(image, (th, tw), overlap, blend)
                    session = Session(position_predictor)
                    actual = runtime.tiled_logits(session, contract((th, tw)), image,
                                                  overlap=overlap, blend=blend)
                    np.testing.assert_allclose(actual, expected, atol=config['oracle_atol'],
                                               rtol=config['oracle_rtol'])
                    self.assertEqual(actual.shape, image.shape)
                    self.assertEqual(len(session.calls), len(starts[0])*len(starts[1]))
                    for supplied, (y, x) in zip(session.calls, ((y, x) for y in starts[0] for x in starts[1])):
                        expected_tile = np.zeros((1, 2, th, tw), dtype=np.float32)
                        vh, vw = min(th, h-y), min(tw, w-x)
                        expected_tile[..., :vh, :vw] = image[..., y:y+vh, x:x+vw]
                        np.testing.assert_array_equal(supplied, expected_tile)
                    np.testing.assert_array_equal(image, original)
        print('Verified 256 seeded geometries, two blends, scalar gather oracle and every padded tile.')

    def test_identity_constant_and_single_tile(self):
        x = np.arange(2*13*17, dtype=np.float32).reshape(1, 2, 13, 17)/128
        for blend in ('constant', 'gaussian'):
            for data in (x, x[..., ::-1]):
                y = runtime.tiled_logits(Session(lambda t: t), contract(), data,
                                         overlap=(4, 6), blend=blend)
                np.testing.assert_allclose(y, data, rtol=2e-6, atol=2e-6)
                y = runtime.tiled_logits(Session(lambda t: np.ones_like(t)*3), contract(), data,
                                         overlap=(4, 6), blend=blend)
                np.testing.assert_allclose(y, 3, rtol=2e-6, atol=2e-6)
        y = runtime.tiled_logits(Session(lambda t: t), contract((13, 17)), x)
        np.testing.assert_array_equal(y, x)

    def test_limits_and_invalid_inputs_before_calls(self):
        x = np.ones((1, 2, 13, 17), np.float32)
        session = Session(lambda t: t)
        for kwargs in (dict(max_tiles=1), dict(max_image_pixels=220), dict(max_input_elements=441),
                       dict(max_output_elements=441), dict(max_tiles=True), dict(max_tiles=0),
                       dict(overlap=(5, 0)), dict(overlap=(-1, 0)), dict(overlap=(True, 1)),
                       dict(overlap=(0,)), dict(blend='reflect')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                runtime.tiled_logits(session, contract(), x, **kwargs)
        for bad in (x.astype('float64'), x[0], x[:, :1], np.repeat(x, 2, axis=0), x[..., :0],
                    [[1]], x*np.nan, x*np.inf):
            with self.assertRaises(ValueError):
                runtime.tiled_logits(session, contract(), bad)
        self.assertFalse(session.calls)
        p = runtime.tile_plan(x.shape, contract(), max_image_pixels=221,
                              max_input_elements=442, max_output_elements=442, max_tiles=9)
        self.assertEqual(p['tile_count'], 9)
        with self.assertRaisesRegex(ValueError, 'max_image_pixels'):
            runtime.tile_plan((1, 2, 10**50, 10**50), contract())

    def test_predictor_failures_and_overflow(self):
        x = np.ones((1, 2, 13, 17), np.float32)
        for fn in (lambda t: t.astype('float64'), lambda t: t[0], lambda t: t[:, :1],
                   lambda t: t*np.nan, lambda t: t*np.inf, lambda t: [1]):
            with self.assertRaises(ValueError):
                runtime.tiled_logits(Session(fn), contract(), x)
        with self.assertRaisesRegex(ValueError, 'overflowed'):
            runtime.tiled_logits(Session(lambda t: np.full_like(t, np.finfo(np.float32).max)),
                                 contract(), x, overlap=(4, 6))
        for error in (MemoryError(), KeyboardInterrupt(), SystemExit(143)):
            with patch.object(runtime, 'blend_weights', side_effect=error), self.assertRaises(type(error)):
                runtime.tiled_logits(Session(lambda t: t), contract(), x)


class BundleTests(unittest.TestCase):
    def test_process_signals_during_serialization(self):
        # Exercise real CLI signal delivery after partial bytes have been written.
        script = '''
import runpy, sys
from pathlib import Path
import numpy as np
def blocked_save(stream, *args, **kwargs):
    stream.write(b'partial')
    stream.flush()
    Path(sys.argv[2] + '.ready').touch()
    import time
    while True:
        time.sleep(0.01)
np.save = blocked_save
runpy.run_path(sys.argv.pop(1), run_name='__main__')
'''
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            source, output = temp/'input.npy', temp/'output.npy'
            np.save(source, np.ones((1, 1, 35, 43), np.float32))
            ready = Path(str(source) + '.ready')
            for sig in (signal.SIGINT, signal.SIGTERM):
                proc = subprocess.Popen([sys.executable, '-c', script,
                    str(ROOT/'src/unet_budget/onnx_tiled.py'), str(BUNDLE), str(source),
                    '--output', str(output)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    deadline = time.monotonic() + 30
                    while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(ready.exists(), 'worker did not reach serialization')
                    self.assertTrue(output.exists())
                    proc.send_signal(sig)
                    proc.communicate(timeout=30)
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertFalse(output.exists())
                    ready.unlink()
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    proc.communicate()

    def test_real_bundle_and_cli(self):
        session, manifest = runtime.load_bundle(BUNDLE)
        x = np.random.default_rng(9).normal(size=(1, 1, 65, 81)).astype('float32')
        y = runtime.tiled_logits(session, manifest, x, overlap=(16, 20), blend='gaussian')
        self.assertEqual(y.shape, (1, 2, 65, 81))
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            np.save(temp/'input.npy', x)
            args = [str(BUNDLE), str(temp/'input.npy'), '--output', str(temp/'out.npy'),
                    '--overlap', '16', '20', '--blend', 'gaussian']
            self.assertEqual(runtime.main(args), 0)
            np.testing.assert_array_equal(np.load(temp/'out.npy'), y)
            saved = (temp/'out.npy').read_bytes()
            self.assertEqual(runtime.main(args), 2)
            self.assertEqual((temp/'out.npy').read_bytes(), saved)
            (temp/'out.npy').unlink()
            (temp/'out.npy').symlink_to(temp/'missing')
            self.assertEqual(runtime.main(args), 2)
            self.assertTrue((temp/'out.npy').is_symlink())

    def test_corrupt_bundles_before_session(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)/'bundle'
            shutil.copytree(BUNDLE, dest)
            original = json.loads((dest/'manifest.json').read_text())
            changes = [lambda m: m.update(schema_version=True), lambda m: m.update(format='other'),
                       lambda m: m['input'].update(shape=[2, 1, 33, 41]),
                       lambda m: m['input'].update(shape=[1, 1, 'dynamic', 41]),
                       lambda m: m['input'].update(shape=[1, 1, 10**8, 10**8]),
                       lambda m: m['input'].update(dtype='float64'),
                       lambda m: m['output'].update(shape=[1, 2, 1, 1]),
                       lambda m: m['artifacts'].update({'../escape': {}}),
                       lambda m: m['artifacts']['model.onnx'].update(bytes=True),
                       lambda m: m['artifacts']['model.onnx'].update(sha256='0'*64)]
            for change in changes:
                manifest = copy.deepcopy(original)
                change(manifest)
                (dest/'manifest.json').write_text(json.dumps(manifest))
                with patch('onnxruntime.InferenceSession') as factory, self.assertRaises(ValueError):
                    runtime.load_bundle(dest)
                factory.assert_not_called()
            (dest/'manifest.json').write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                runtime.load_bundle(dest)
            (dest/'manifest.json').write_text(json.dumps(original))
            for name in runtime.ARTIFACTS:
                path = dest/name
                data = path.read_bytes()
                path.write_bytes(data+b'!')
                with self.assertRaises(ValueError):
                    runtime.load_bundle(dest)
                path.write_bytes(data)
                path.unlink()
                path.symlink_to(BUNDLE/name)
                with self.assertRaises(ValueError):
                    runtime.load_bundle(dest)
                path.unlink()
                path.write_bytes(data)
            for threads in (0, True, 33, 1.5):
                with self.assertRaises(ValueError):
                    runtime.load_bundle(dest, threads)
            # Even consistently hashed source files are never executed.
            data = b'raise RuntimeError("bundle source executed")\n'
            (dest/'model.py').write_bytes(data)
            original['artifacts']['model.py'] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            (dest/'manifest.json').write_text(json.dumps(original))
            runtime.load_bundle(dest)
            original['input']['shape'][-1] += 1
            original['output']['shape'][-1] += 1
            (dest/'manifest.json').write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'runtime interface'):
                runtime.load_bundle(dest)

    def test_cli_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            x = np.ones((1, 1, 35, 43), np.float32)
            np.save(temp/'input.npy', x)
            output = temp/'output.npy'
            args = [str(BUNDLE), str(temp/'input.npy'), '--output', str(output)]
            for error in (OSError('disk full'), MemoryError('allocation failed'), KeyboardInterrupt(), SystemExit(143)):
                def fail(stream, *args, **kwargs):
                    stream.write(b'partial')
                    raise error
                with patch('numpy.save', side_effect=fail):
                    if isinstance(error, Exception):
                        self.assertEqual(runtime.main(args), 2)
                    else:
                        with self.assertRaises(type(error)):
                            runtime.main(args)
                self.assertFalse(output.exists())
            with patch.object(runtime, 'tiled_logits', side_effect=KeyboardInterrupt()), self.assertRaises(KeyboardInterrupt):
                runtime.main(args)
            self.assertFalse(output.exists())
            # Exclusive-open race: preserve the winner's file.
            original = runtime.tiled_logits
            def race(*a, **kw):
                output.write_bytes(b'winner')
                return original(*a, **kw)
            with patch.object(runtime, 'tiled_logits', side_effect=race):
                self.assertEqual(runtime.main(args), 2)
            self.assertEqual(output.read_bytes(), b'winner')
            output.unlink()
            for bad in (x.astype('float64'), np.array([object()], dtype=object), x*np.nan):
                np.save(temp/'input.npy', bad)
                self.assertEqual(runtime.main(args), 2)
                self.assertFalse(output.exists())
            (temp/'input.npy').write_bytes(b'broken header')
            self.assertEqual(runtime.main(args), 2)
            np.savez(temp/'input.npz', x=x)
            self.assertEqual(runtime.main([str(BUNDLE), str(temp/'input.npz'), '--output', str(output)]), 2)
            np.save(temp/'input.npy', x)
            self.assertEqual(runtime.main(args+['--max-tiles', '1']), 2)
            self.assertFalse(output.exists())
            with self.assertRaises(SystemExit) as exc:
                runtime.terminate(15, None)
            self.assertEqual(exc.exception.code, 143)


if __name__ == '__main__':
    unittest.main()

"""Independent geometric oracle: scalar grid walk and per-pixel accumulation."""
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from unet_budget.inference import blend_weights, tile_plan, tiled_logits

ROOT = Path(__file__).resolve().parents[1]


def starts(length, side, overlap):
    result = [0]
    while result[-1] + side < length:
        result.append(result[-1] + side - overlap)
    return result


def oracle_weights(th, tw, mode):
    a = np.ones((th, tw), dtype=np.float64)
    if mode == 'gaussian':
        for y in range(th):
            for x in range(tw):
                a[y, x] = math.exp(-32 * (((y-(th-1)/2)/th)**2 + ((x-(tw-1)/2)/tw)**2))
        a = np.maximum(a / a.max(), 1e-6)
    return a


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_256_seeded_geometries_independent_oracles(self):
        rng = random.Random(8142)
        for case in range(256):
            n, c = rng.randint(1, 2), rng.randint(1, 3)
            h, w = rng.randint(1, 39), rng.randint(1, 43)
            th, tw = rng.randint(1, 27), rng.randint(1, 29)
            oh, ow = rng.randrange(th), rng.randrange(tw)
            batch = rng.randint(1, 9)
            dtype = torch.float64 if case % 2 else torch.float32
            # Distinct absolute coordinates, channels and samples expose misplacement.
            x = torch.arange(n*c*h*w, dtype=dtype).reshape(n, c, h, w) / (n*c*h*w)
            ys, xs = starts(h, th, oh), starts(w, tw, ow)
            plan = tile_plan(x.shape, tile_size=(th, tw), overlap=(oh, ow), tile_batch_size=batch)
            self.assertEqual(plan['y_starts'], ys)
            self.assertEqual(plan['x_starts'], xs)
            self.assertEqual(plan['tile_count'], n * len(ys) * len(xs))
            for mode in ('constant', 'gaussian'):
                with self.subTest(case=case, shape=x.shape, tile=(th, tw), overlap=(oh, ow), mode=mode):
                    weights = oracle_weights(th, tw, mode)
                    np.testing.assert_allclose(blend_weights((th, tw), mode, dtype=dtype).numpy(),
                                               weights, rtol=3e-6, atol=1e-9)
                    normalization = np.zeros((h, w))
                    coverage = np.zeros((h, w), dtype=np.int64)
                    # Local-position predictor: expected output differs in overlap regions.
                    local = np.array([[y * tw + z for z in range(tw)] for y in range(th)], dtype=float)
                    expected = np.zeros((h, w))
                    for y in ys:
                        for z in xs:
                            for iy in range(th):
                                for iz in range(tw):
                                    if y+iy < h and z+iz < w:
                                        normalization[y+iy, z+iz] += weights[iy, iz]
                                        coverage[y+iy, z+iz] += 1
                                        expected[y+iy, z+iz] += weights[iy, iz] * local[iy, iz]
                    self.assertTrue((coverage >= 1).all())
                    self.assertTrue((normalization > 0).all())
                    if oh == ow == 0:
                        self.assertTrue((coverage == 1).all())
                    expected /= normalization
                    kwargs = dict(tile_size=(th, tw), overlap=(oh, ow), tile_batch_size=batch, blend=mode)
                    known = [
                        (lambda t: torch.ones(t.shape[0], 4, th, tw, dtype=t.dtype) * 2.5,
                         torch.full((n, 4, h, w), 2.5, dtype=dtype)),
                        (lambda t: t, x),
                        (lambda t: torch.cat((t * 1.7 - 0.2, t.sum(1, keepdim=True)), 1),
                         torch.cat((x * 1.7 - 0.2, x.sum(1, keepdim=True)), 1)),
                        (lambda t: torch.tensor(local, dtype=t.dtype)[None, None].expand(t.shape[0], 2, th, tw),
                         torch.tensor(expected, dtype=dtype)[None, None].expand(n, 2, h, w)),
                    ]
                    for predictor, truth in known:
                        actual = tiled_logits(predictor, x, **kwargs)
                        self.assertEqual(actual.shape, truth.shape)
                        torch.testing.assert_close(actual, truth, rtol=5e-6, atol=3e-5 if dtype == torch.float32 else 1e-10)
        print('Verified 256 seeded geometries, both blends, four independent predictors.')

    def test_padding_and_batch_order(self):
        x = torch.arange(2*3*5*7, dtype=torch.float64).reshape(2, 3, 5, 7) + 1
        expected = []
        for n in range(2):
            for y in (0, 3):
                for z in (0, 4):
                    patch = torch.zeros(3, 4, 5, dtype=x.dtype)
                    for c in range(3):
                        for iy in range(4):
                            for iz in range(5):
                                if y+iy < 5 and z+iz < 7:
                                    patch[c, iy, iz] = x[n, c, y+iy, z+iz]
                    expected.append(patch)
        seen = []
        def predictor(t):
            self.assertLessEqual(len(t), 3)
            seen.extend(t.clone().unbind())
            return t
        out = tiled_logits(predictor, x, tile_size=(4, 5), overlap=(1, 1), tile_batch_size=3)
        torch.testing.assert_close(torch.stack(seen), torch.stack(expected), rtol=0, atol=0)
        torch.testing.assert_close(out, x, rtol=0, atol=0)

    def test_logits_are_blended_before_classification(self):
        # At absolute x=1, the first window votes class 0 weakly, the second
        # votes class 1 strongly. Equal label voting loses this distinction.
        image = torch.arange(3, dtype=torch.float32).reshape(1, 1, 1, 3)
        def predictor(t):
            first_window = t[:, 0, 0, 0] == 0
            logits = torch.zeros(len(t), 2, 1, 2)
            logits[first_window, 0] = 1
            logits[~first_window, 1] = 10
            return logits
        logits = tiled_logits(predictor, image, tile_size=(1, 2), overlap=(0, 1), tile_batch_size=2)
        torch.testing.assert_close(logits[0, :, 0, 1], torch.tensor([0.5, 5.0]))
        self.assertEqual(logits.argmax(1)[0, 0, 1].item(), 1)
        self.assertEqual(torch.tensor([0.5, 0.5]).argmax().item(), 0)

    def test_invalid_settings_before_predictor(self):
        x = torch.ones(2, 1, 11, 13)
        def forbidden(t):
            self.fail('invalid settings reached predictor')
        cases = [dict(tile_size=v) for v in (2, (1,), (0, 4), (True, 4), (2.5, 4), (2, '4'))]
        cases += [dict(overlap=v) for v in (1, (-1, 0), (4, 0), (0, 4), (False, 0))]
        cases += [dict(blend=v) for v in ('bad', None, 1)]
        for field in ('tile_batch_size', 'max_image_pixels', 'max_tiles', 'max_tile_batch_size', 'max_tile_pixels'):
            cases += [{field: v} for v in (0, -1, True, 1.2)]
        cases += [dict(max_image_pixels=285), dict(max_tiles=23),
                  dict(tile_batch_size=3, max_tile_batch_size=2), dict(max_tile_pixels=15)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                tiled_logits(forbidden, x, **(dict(tile_size=(4, 4), overlap=(0, 0)) | kwargs))
        # Exactly inclusive limits: 2 * 3 * 4 = 24 tiles and 286 pixels.
        tiled_logits(lambda t: t, x, tile_size=(4, 4), overlap=(0, 0), max_tiles=24,
                     max_image_pixels=286, max_tile_pixels=16, max_tile_batch_size=2, tile_batch_size=2)
        for bad in (None, [], torch.ones(1, 2, 3), torch.ones(0, 1, 2, 3),
                    torch.ones(1, 0, 2, 3), torch.ones(1, 1, 0, 3), x.long(), x.half(),
                    torch.ones(1, 1, 2, 3, device='meta'), x.to_sparse()):
            with self.subTest(input=type(bad)), self.assertRaises(ValueError):
                tiled_logits(forbidden, bad, tile_size=(4, 4), overlap=(0, 0))
        with self.assertRaisesRegex(ValueError, 'max_image_pixels'):
            tile_plan((1, 1, 10**12, 10**12))
        with self.assertRaisesRegex(ValueError, 'max_tiles'):
            tile_plan((1, 1, 1000, 1000), tile_size=(32, 32), overlap=(31, 31))

    def test_predictor_contract_and_numerics(self):
        x = torch.ones(1, 1, 5, 7)
        kwargs = dict(tile_size=(4, 4), overlap=(1, 1))
        bad = [None, lambda t: (t,), lambda t: t[0], lambda t: t[:, :, :-1],
               lambda t: t.expand(2, 1, 4, 4), lambda t: t[:, :0], lambda t: t.double(),
               lambda t: t.to_sparse(), lambda t: t.to('meta'),
               lambda t: t * float('nan'), lambda t: t * float('inf')]
        for predictor in bad:
            with self.assertRaises(ValueError):
                tiled_logits(predictor, x, **kwargs)
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                tiled_logits(lambda t: t, x * value, **kwargs)
        model = torch.nn.Conv2d(1, 1, 1)
        with self.assertRaisesRegex(ValueError, 'eval'):
            tiled_logits(model, x, **kwargs)
        model.eval().double()
        with self.assertRaisesRegex(ValueError, 'dtype'):
            tiled_logits(model, x, **kwargs)
        calls = []
        def changing(t):
            calls.append(1)
            return t.expand(-1, len(calls), -1, -1)
        with self.assertRaisesRegex(ValueError, 'changed'):
            tiled_logits(changing, x, **kwargs)
        with self.assertRaisesRegex(ValueError, 'overflowed'):
            tiled_logits(lambda t: t * torch.finfo(t.dtype).max, x, **kwargs)
        # Inference does not modify inputs, training flags or parameter gradients.
        model = torch.nn.Conv2d(1, 2, 1).eval()
        saved = x.clone()
        result = tiled_logits(model, x.requires_grad_(), **kwargs)
        self.assertFalse(result.requires_grad)
        self.assertFalse(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        torch.testing.assert_close(x, saved)
        # Noncontiguous storage is accepted.
        transposed = x.transpose(-1, -2)
        torch.testing.assert_close(tiled_logits(lambda t: t, transposed, **kwargs), transposed)

    def test_cli_helper_determinism_and_specialization(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = []
            for name in ('a', 'b'):
                out = Path(temp) / name
                subprocess.run([sys.executable, '-m', 'unet_budget', str(ROOT/'examples/circles.json'),
                                '--out', str(out), '--with-inference'], check=True, capture_output=True)
                source = (out/'inference.py').read_bytes()
                report = json.loads((out/'architecture.json').read_text())
                self.assertEqual(report['inference']['sha256'], hashlib.sha256(source).hexdigest())
                self.assertEqual((out/'model.py').read_bytes(), (ROOT/'examples/generated/model.py').read_bytes())
                outputs.append(source)
            self.assertEqual(*outputs)
            namespace = {}
            exec(outputs[0], namespace)
            with self.assertRaises(ValueError):
                namespace['tile_plan']((1, 1, 5, 7), tile_size=(3, 4), overlap=(0, 0))
            with self.assertRaises(ValueError):
                namespace['tile_plan']((1, 2, 5, 7), tile_size=(4, 4), overlap=(0, 0))
            with self.assertRaisesRegex(ValueError, 'output channels'):
                namespace['tiled_logits'](lambda t: t, torch.ones(1, 1, 5, 7), tile_size=(4, 4), overlap=(0, 0))


if __name__ == '__main__':
    unittest.main()

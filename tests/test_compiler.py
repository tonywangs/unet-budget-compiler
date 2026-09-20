import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch import nn

from unet_budget import SpecError, compile_spec
from reference import ReferenceUNet


torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def spec(**changes):
    return dict(input_channels=1, output_classes=2, depth=2, max_parameters=8000,
                width_candidates=[2, 4, 6, 8], input_height=33, input_width=41,
                **{}) | changes


def generated(raw):
    code, report = compile_spec(raw)
    namespace = {}
    exec(compile(code, '<generated>', 'exec'), namespace)
    return namespace['UNet'](), report


class CompilerTests(unittest.TestCase):
    def test_committed_example(self):
        root = Path(__file__).resolve().parents[1]
        raw = json.loads((root/'examples/circles.json').read_text())
        code, report = compile_spec(raw)
        self.assertEqual(code, (root/'examples/generated/model.py').read_text())
        self.assertEqual(report, json.loads((root/'examples/generated/architecture.json').read_text()))

    def test_invalid_specs(self):
        invalid = [None, [], {}, spec(extra=1), spec(depth=True), spec(depth=0),
                   spec(depth=9), spec(input_channels=0), spec(output_classes=1.5),
                   spec(max_parameters=-1), spec(max_parameters=float('inf')),
                   spec(width_candidates=[]), spec(width_candidates=[False]),
                   spec(width_candidates=[0]), spec(width_candidates=[1025]),
                   spec(width_candidates=[1]*257), spec(width_candidates='4'),
                   spec(input_height=3), spec(input_width=3), spec(input_width='33')]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(SpecError):
                compile_spec(raw)
        with self.assertRaisesRegex(SpecError, 'smallest candidate.*needs'):
            compile_spec(spec(max_parameters=1))

    def test_determinism_and_hashes(self):
        raw = spec(width_candidates=[8, 4, 2, 4, 6])
        saved = copy.deepcopy(raw)
        code, report = compile_spec(raw)
        self.assertEqual(raw, saved)
        self.assertEqual((code, report), compile_spec(dict(reversed(list(raw.items())))))
        self.assertEqual((code, report), compile_spec(spec()))
        digest = hashlib.sha256(json.dumps(report['specification'], sort_keys=True,
                                          separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(report['specification_sha256'], digest)
        self.assertEqual(report['generated_code_sha256'], hashlib.sha256(code.encode()).hexdigest())
        self.assertNotIn('unet_budget', code)

    def test_exhaustive_small_budgets(self):
        # Ground truth is an independently built model, not the compiler formula.
        for depth in (1, 2, 3):
            for inputs, classes in ((1, 1), (3, 4)):
                counts = {w: sum(p.numel() for p in ReferenceUNet(inputs, classes, depth, w).parameters())
                          for w in (1, 2, 3, 5)}
                budgets = {n + delta for n in counts.values() for delta in (-1, 0, 1)}
                for budget in sorted(budgets):
                    raw = spec(depth=depth, input_channels=inputs, output_classes=classes,
                               width_candidates=[5, 1, 3, 2], max_parameters=budget)
                    feasible = [w for w, n in counts.items() if n <= budget]
                    with self.subTest(depth=depth, inputs=inputs, classes=classes, budget=budget):
                        if not feasible:
                            with self.assertRaises(SpecError):
                                compile_spec(raw)
                            continue
                        model, report = generated(raw)
                        self.assertEqual(report['selection']['base_width'], max(feasible))
                        self.assertEqual(report['selection']['parameters'], sum(p.numel() for p in model.parameters()))
                        self.assertEqual({c['base_width']: c['parameters'] for c in report['candidates']}, counts)

    def test_shapes_counts_and_runtime_validation(self):
        for depth in (1, 2, 3):
            for h, w in ((32, 32), (24, 40), (33, 41), (2**depth, 2**depth)):
                model, report = generated(spec(depth=depth, width_candidates=[2],
                                              max_parameters=10**8, input_height=h, input_width=w))
                observed = {}
                handles = []
                for name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        def hook(mod, inputs, output, name=name):
                            observed[name] = (list(inputs[0].shape), list(output.shape),
                                              sum(p.numel() for p in mod.parameters()))
                        handles.append(module.register_forward_hook(hook))
                result = model(torch.randn(2, 1, h, w))
                self.assertEqual(tuple(result.shape), (2, 2, h, w))
                for layer in report['layers']:
                    if layer['kind'] == 'conv2d':
                        self.assertEqual(observed[layer['name']],
                                         ([2] + layer['inputs'][0][1:], [2] + layer['output'][1:], layer['parameters']))
                self.assertEqual(sum(l['parameters'] for l in report['layers']), report['selection']['parameters'])
                for handle in handles:
                    handle.remove()
                # Generated models are spatially flexible, unlike report example shapes.
                self.assertEqual(tuple(model(torch.randn(1, 1, h+1, w+3)).shape), (1, 2, h+1, w+3))
                for bad in (torch.randn(1, 1, 2**depth-1, w), torch.randn(1, 1, h, 2**depth-1),
                            torch.randn(1, 2, h, w), torch.randn(1, h, w), torch.randn(0, 1, h, w)):
                    with self.assertRaises(ValueError):
                        model(bad)

    def test_maximum_depth_and_single_output(self):
        model, report = generated(spec(depth=8, input_channels=3, output_classes=1,
                                      width_candidates=[1], max_parameters=10**9,
                                      input_height=256, input_width=257))
        self.assertEqual(report['selection']['parameters'], sum(p.numel() for p in model.parameters()))
        self.assertEqual(tuple(model(torch.randn(1, 3, 256, 257)).shape), (1, 1, 256, 257))

    def test_forward_input_and_parameter_gradients(self):
        # 24 seeded cases, float32 and float64, square/rectangular/odd/minimum.
        for dtype in (torch.float32, torch.float64):
            for seed in (7, 19, 41):
                for depth, h, w in ((1, 16, 16), (2, 12, 20), (3, 17, 25), (3, 8, 8)):
                    with self.subTest(dtype=dtype, seed=seed, depth=depth, h=h, w=w):
                        torch.manual_seed(seed)
                        model, _ = generated(spec(input_channels=2, output_classes=3, depth=depth,
                                                  width_candidates=[2], max_parameters=10**8,
                                                  input_height=h, input_width=w))
                        model = model.to(dtype)
                        ref = ReferenceUNet(2, 3, depth, 2).to(dtype)
                        actual_convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                        ref_convs = [m for m in ref.modules() if isinstance(m, nn.Conv2d)]
                        self.assertEqual(len(actual_convs), len(ref_convs))
                        for a, b in zip(actual_convs, ref_convs):
                            b.load_state_dict(a.state_dict())
                        x = torch.randn(2, 2, h, w, dtype=dtype, requires_grad=True)
                        y = x.detach().clone().requires_grad_(True)
                        a, b = model(x), ref(y)
                        atol, rtol = ((1e-6, 1e-5) if dtype == torch.float32 else (1e-10, 1e-8))
                        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
                        probe = torch.randn_like(a)
                        (a * probe).sum().backward()
                        (b * probe).sum().backward()
                        torch.testing.assert_close(x.grad, y.grad, atol=atol, rtol=rtol)
                        for ca, cb in zip(actual_convs, ref_convs):
                            for pa, pb in zip(ca.parameters(), cb.parameters()):
                                self.assertIsNotNone(pa.grad)
                                torch.testing.assert_close(pa.grad, pb.grad, atol=atol, rtol=rtol)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.json'
            source.write_text(json.dumps(spec()))
            command = [sys.executable, '-m', 'unet_budget', str(source), '--out']
            for folder in ('one', 'two'):
                run = subprocess.run(command + [str(root / folder)], capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
            for name in ('model.py', 'architecture.json'):
                self.assertEqual((root/'one'/name).read_bytes(), (root/'two'/name).read_bytes())
            run = subprocess.run(command + [str(root/'one')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn('new or empty directory', run.stderr)
            for bad in ('{', '{"depth":1,"depth":2}', '{"depth":NaN}'):
                source.write_text(bad)
                run = subprocess.run(command + [str(root/'bad')], capture_output=True, text=True)
                self.assertEqual(run.returncode, 2)
                self.assertNotIn('Traceback', run.stderr)
                self.assertFalse((root/'bad').exists())


if __name__ == '__main__':
    unittest.main()

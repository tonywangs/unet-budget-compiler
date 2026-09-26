"""Export validation, failure cleanup and standalone inference contracts."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np
import torch

from unet_budget import SpecError, compile_spec
from unet_budget.export import export_model, validate_contract, main, MAX_FILE_BYTES
from unet_budget.onnx_inference import load_bundle, predict, main as infer_main

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT/'examples/circles.json').read_text())
CHECKPOINT = ROOT/'results/synthetic-weights.pt'
SHAPE = [1, 1, 33, 41]


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root/'bundle'

    def assert_clean(self):
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.glob('.bundle-*')), [])

    def test_trained_export_manifest_and_standalone_contract(self):
        manifest = export_model(SPEC, CHECKPOINT, SHAPE, self.out)
        self.assertEqual(manifest, json.loads((self.out/'manifest.json').read_text()))
        self.assertEqual(manifest['checkpoint_sha256'], hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest())
        for name, info in manifest['artifacts'].items():
            data = (self.out/name).read_bytes()
            self.assertEqual(info, dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        session, contract = load_bundle(self.out)
        x = np.random.default_rng(4).normal(size=SHAPE).astype('float32')
        namespace = {}
        exec(compile_spec(SPEC)[0], namespace)
        model = namespace['UNet']().eval()
        model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
        with torch.inference_mode():
            reference = model(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(predict(session, contract, x), reference, atol=1e-5, rtol=1e-5)
        self.assertEqual(session.get_providers(), ['CPUExecutionProvider'])
        for bad in (x.astype('float64'), x[0], x[:, :, :-1], x.repeat(2, axis=0), x * np.nan):
            with self.assertRaises(ValueError):
                predict(session, contract, bad)
        # Noncontiguous but correctly shaped input is explicitly supported.
        self.assertTrue(np.isfinite(predict(session, contract, x[:, :, :, ::-1])).all())
        for threads in (True, 0, 33):
            with self.assertRaises(ValueError):
                load_bundle(self.out, threads)
        model_path = self.out/'model.onnx'
        model_path.write_bytes(model_path.read_bytes()+b'corruption')
        with self.assertRaisesRegex(ValueError, 'hash or size'):
            load_bundle(self.out)

    def test_malformed_checkpoints(self):
        state = torch.load(CHECKPOINT, weights_only=True)
        key = next(iter(state))
        cases = [None, {'state_dict': state}, {**state, 'extra': torch.tensor(1.)},
                 {k: v for k, v in state.items() if k != key},
                 {**state, key: state[key].flatten()}, {**state, key: state[key].double()},
                 {**state, key: 'not a tensor'}, {**state, key: torch.nn.Parameter(state[key])},
                 {**state, key: torch.full_like(state[key], float('nan'))},
                 {**state, key: torch.full_like(state[key], float('inf'))}]
        for index, value in enumerate(cases):
            with self.subTest(index=index):
                path = self.root/'bad.pt'
                torch.save(value, path)
                with self.assertRaises(SpecError):
                    export_model(SPEC, path, SHAPE, self.out)
                self.assert_clean()
        for data in (b'', b'not a checkpoint', b'x'*(MAX_FILE_BYTES+1)):
            path.write_bytes(data)
            with self.assertRaises(SpecError):
                export_model(SPEC, path, SHAPE, self.out)
            self.assert_clean()
        torch.save(state, path, _use_new_zipfile_serialization=False)
        with self.assertRaisesRegex(SpecError, 'ZIP'):
            export_model(SPEC, path, SHAPE, self.out)
        with self.assertRaises(SpecError):
            export_model(SPEC, self.root/'missing.pt', SHAPE, self.out)
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('oversized', b'0' * (MAX_FILE_BYTES+1))
        with self.assertRaisesRegex(SpecError, 'ZIP resource'):
            export_model(SPEC, path, SHAPE, self.out)
        with zipfile.ZipFile(path, 'w') as archive:
            for index in range(257):
                archive.writestr(str(index), b'')
        with self.assertRaisesRegex(SpecError, 'ZIP resource'):
            export_model(SPEC, path, SHAPE, self.out)

    def test_shape_and_resource_limits_before_dependencies(self):
        for shape in ([0, 1, 33, 41], [True, 1, 33, 41], [1, 2, 33, 41], [1, 1, 3, 41],
                      [17, 1, 33, 41], [1, 1, 1025, 41], [1, 1, 33], ['1', 1, 33, 41]):
            with self.subTest(shape=shape), mock.patch('unet_budget.export.dependencies') as imports:
                with self.assertRaises(SpecError):
                    export_model(SPEC, CHECKPOINT, shape, self.out)
                imports.assert_not_called()
        for spec, shape in (({**SPEC, 'depth': 5}, [1, 1, 33, 41]),
                            ({**SPEC, 'output_classes': 17, 'max_parameters': 100000}, SHAPE),
                            ({**SPEC, 'width_candidates': [33], 'max_parameters': 10**9}, SHAPE),
                            ({**SPEC, 'depth': 4, 'width_candidates': [32], 'max_parameters': 10**9}, SHAPE),
                            (SPEC, [16, 1, 1024, 1024])):
            with self.assertRaises(SpecError):
                validate_contract(spec, shape)

    def test_missing_optional_dependencies(self):
        for package in ('torch', 'onnx', 'onnxruntime'):
            with mock.patch.dict('sys.modules', {package: None}):
                with self.assertRaisesRegex(SpecError, 'optional'):
                    export_model(SPEC, CHECKPOINT, SHAPE, self.out)
                self.assert_clean()

    def test_collisions_and_serialization_cleanup(self):
        for kind in ('directory', 'file', 'dangling_symlink'):
            if kind == 'directory':
                self.out.mkdir()
            elif kind == 'file':
                self.out.write_text('preserve')
            else:
                self.out.symlink_to(self.root/'absent')
            with mock.patch('unet_budget.export.dependencies') as imports:
                with self.assertRaisesRegex(SpecError, 'must not exist'):
                    export_model(SPEC, CHECKPOINT, SHAPE, self.out)
                imports.assert_not_called()
            if kind == 'directory':
                self.out.rmdir()
            else:
                self.out.unlink()
        for patch in ('torch.onnx.export', 'onnx.checker.check_model'):
            with mock.patch(patch, side_effect=RuntimeError('injected failure')):
                with self.assertRaises(RuntimeError):
                    export_model(SPEC, CHECKPOINT, SHAPE, self.out)
            self.assert_clean()
        original_write = Path.write_text
        def fail_manifest(path, data, *args, **kwargs):
            if path.name == 'manifest.json':
                path.write_bytes(b'partial serialization')
                raise OSError('injected manifest serialization failure')
            return original_write(path, data, *args, **kwargs)
        with mock.patch.object(Path, 'write_text', fail_manifest):
            with self.assertRaisesRegex(OSError, 'serialization'):
                export_model(SPEC, CHECKPOINT, SHAPE, self.out)
        self.assert_clean()
        # Failure after exclusive destination creation must remove partial files.
        original = Path.rename
        moves = []
        def fail_second(source, target):
            moves.append(source)
            if len(moves) == 2:
                raise OSError('injected rename failure')
            return original(source, target)
        with mock.patch.object(Path, 'rename', fail_second):
            with self.assertRaises(OSError):
                export_model(SPEC, CHECKPOINT, SHAPE, self.out)
        self.assert_clean()

    def test_cli_and_output_failure_cleanup(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main([str(ROOT/'examples/circles.json'), '--checkpoint', str(CHECKPOINT),
                                   '--shape', '1', '1', '33', '41', '--out', str(self.out)]), 0)
            x = self.root/'input.npy'
            y = self.root/'logits.npy'
            np.save(x, np.zeros(SHAPE, dtype=np.float32))
            args = [str(self.out), str(x), '--output', str(y)]
            self.assertEqual(infer_main(args), 0)
            self.assertEqual(np.load(y).shape, (1, 2, 33, 41))
            old = y.read_bytes()
            self.assertEqual(infer_main(args), 2)
            self.assertEqual(y.read_bytes(), old)
            y.unlink()
            with mock.patch('numpy.save', side_effect=OSError('injected write failure')):
                self.assertEqual(infer_main(args), 2)
            self.assertFalse(y.exists())
            manifest_path = self.out/'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['input']['shape'][2] -= 1
            manifest['output']['shape'][2] -= 1
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'interface'):
                load_bundle(self.out)
            manifest['schema_version'] = 99
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                load_bundle(self.out)


if __name__ == '__main__':
    unittest.main()

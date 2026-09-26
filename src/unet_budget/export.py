"""Bounded, offline, fixed-shape ONNX export; optional imports stay lazy."""
import argparse
from collections import OrderedDict
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import warnings
import zipfile

from . import SpecError, compile_spec
from .cli import unique_object
from .compiler import architecture

OPSET = 17
MAX_FILE_BYTES = 10 * 1024**2
LIMITS = dict(max_parameters=2_000_000, max_activation_elements=16_000_000,
              max_multiply_accumulates=2_000_000_000, max_file_bytes=MAX_FILE_BYTES,
              max_depth=4, max_batch=16, max_channels=16, max_axis=1024,
              max_base_width=32)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def dependencies():
    try:
        import torch
        import onnx
        import onnxruntime
    except ImportError as exc:
        raise SpecError('ONNX export requires optional torch, onnx and onnxruntime packages; '
                        'see requirements-validation.txt') from exc
    return torch, onnx, onnxruntime


def validate_contract(raw, shape):
    code, report = compile_spec(raw)
    spec = report['specification']
    if not isinstance(shape, (list, tuple)) or len(shape) != 4 or any(type(x) is not int for x in shape):
        raise SpecError('input shape must be four integer NCHW dimensions')
    n, c, h, w = shape
    if not (1 <= n <= LIMITS['max_batch'] and c == spec['input_channels']
            and 2**spec['depth'] <= min(h, w) <= max(h, w) <= LIMITS['max_axis']):
        raise SpecError('unsupported fixed shape: batch 1..16, matching channels, axes 2**depth..1024 required')
    if (spec['depth'] > LIMITS['max_depth'] or max(c, spec['output_classes']) > LIMITS['max_channels']
            or report['selection']['base_width'] > LIMITS['max_base_width']):
        raise SpecError('export supports depth 1..4, channels/classes 1..16, base width 1..32')
    if report['selection']['parameters'] > LIMITS['max_parameters']:
        raise SpecError('export parameter resource limit exceeded')
    layers = architecture({**spec, 'input_height': h, 'input_width': w}, report['selection']['base_width'])
    # Sum all primitive outputs, deliberately including ReLUs and concatenations.
    elements = math.prod(shape) + sum(n * math.prod(layer['output'][1:]) for layer in layers)
    macs = sum(n * math.prod(layer['output'][1:]) * layer['in_channels'] * layer['kernel_size']**2
               for layer in layers if layer['kind'] == 'conv2d')
    if elements > LIMITS['max_activation_elements'] or macs > LIMITS['max_multiply_accumulates']:
        raise SpecError('export activation or compute resource limit exceeded')
    return code, report, dict(primitive_output_elements=elements, multiply_accumulates=macs)


def read_checkpoint(path, model, torch):
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise SpecError('checkpoint must be a nonempty file no larger than 10 MiB')
    with path.open('rb') as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise SpecError('checkpoint exceeds file resource limit')
    try:
        # Legacy pickle checkpoints are intentionally unsupported. Bound expanded
        # ZIP contents as well as the input file before calling weights_only load.
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > 256 or sum(e.file_size for e in entries) > MAX_FILE_BYTES:
                raise SpecError('checkpoint ZIP resource limit exceeded')
        state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    except SpecError:
        raise
    except Exception as exc:
        raise SpecError(f'cannot load tensor-only ZIP checkpoint: {type(exc).__name__}') from exc
    expected = model.state_dict()
    if type(state) not in (dict, OrderedDict) or any(type(k) is not str for k in state):
        raise SpecError('checkpoint must be a plain tensor state_dict')
    if state.keys() != expected.keys():
        raise SpecError('checkpoint keys do not match the compiled architecture')
    storage_bytes = 0
    for key, tensor in state.items():
        if (type(tensor) is not torch.Tensor or tensor.layout != torch.strided
                or tensor.device.type != 'cpu' or tensor.dtype != torch.float32
                or tuple(tensor.shape) != tuple(expected[key].shape)):
            raise SpecError(f'checkpoint tensor contract mismatch: {key}')
        storage_bytes += tensor.untyped_storage().nbytes()
        if storage_bytes > MAX_FILE_BYTES:
            raise SpecError('checkpoint tensor storage resource limit exceeded')
        if not torch.isfinite(tensor).all().item():
            raise SpecError(f'checkpoint contains nonfinite weights: {key}')
    model.load_state_dict(state, strict=True, assign=True)
    return sha256(data)


def export_model(raw, checkpoint, shape, out):
    """Emit a checked bundle into a *new* directory, cleaning up on failure.

    Only compiler-generated source is executed. No user Python is accepted.
    Resource estimates are workload limits, not an OS memory/security boundary.
    """
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise SpecError(f'export output path must not exist: {out}')
    code, report, estimates = validate_contract(raw, shape)
    torch, onnx, ort = dependencies()
    namespace = {}
    exec(compile(code, '<compiled-unet>', 'exec'), namespace)
    # Instantiate shapes only: no random weights can become an export fallback.
    with torch.device('meta'):
        model = namespace['UNet']()
    checkpoint_hash = read_checkpoint(checkpoint, model, torch)
    model.eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with tempfile.TemporaryDirectory(prefix=f'.{out.name}-', dir=out.parent) as temporary:
            staging = Path(temporary)
            model_path = staging / 'model.onnx'
            with torch.inference_mode(), warnings.catch_warnings():
                # Guards on Python shape values are deliberately specialized.
                warnings.simplefilter('ignore', torch.jit.TracerWarning)
                torch.onnx.export(model, torch.zeros(shape, dtype=torch.float32), model_path,
                                  input_names=['image'], output_names=['logits'],
                                  opset_version=OPSET, export_params=True, dynamo=False,
                                  do_constant_folding=True, external_data=False)
            if model_path.stat().st_size > MAX_FILE_BYTES:
                raise SpecError('serialized ONNX model exceeds 10 MiB')
            graph = onnx.load(model_path, load_external_data=False)
            if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
                raise SpecError('external tensor data is unsupported')
            onnx.checker.check_model(graph, full_check=True)
            expected_output = [shape[0], raw['output_classes'], shape[2], shape[3]]
            for values, name, dims in ((graph.graph.input, 'image', list(shape)),
                                       (graph.graph.output, 'logits', expected_output)):
                if len(values) != 1 or values[0].name != name:
                    raise SpecError('unexpected ONNX graph interface')
                tensor = values[0].type.tensor_type
                if tensor.elem_type != onnx.TensorProto.FLOAT or [d.dim_value for d in tensor.shape.dim] != dims:
                    raise SpecError('ONNX graph is not fixed-shape float32')
            helper = Path(__file__).with_name('onnx_inference.py').read_bytes()
            (staging/'inference.py').write_bytes(helper)
            (staging/'model.py').write_text(code, encoding='utf-8')
            (staging/'architecture.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
            files = {p.name: dict(sha256=sha256(p.read_bytes()), bytes=p.stat().st_size)
                     for p in staging.iterdir()}
            manifest = dict(schema_version=1, format='unet-budget-onnx', architecture=report,
                            input=dict(name='image', shape=list(shape), dtype='float32', layout='NCHW'),
                            output=dict(name='logits', shape=expected_output, dtype='float32', layout='NCHW',
                                        semantics='raw logits; no sigmoid or softmax'),
                            precision='float32', opset=OPSET, onnx_ir_version=graph.ir_version,
                            provider='CPUExecutionProvider', exporter='torch.onnx.export(dynamo=False)',
                            versions={name: importlib.metadata.version(name) for name in
                                      ('torch', 'numpy', 'onnx', 'onnxruntime', 'unet-budget-compiler')},
                            python=platform.python_version(), checkpoint_sha256=checkpoint_hash,
                            source_hashes=dict(specification=report['specification_sha256'],
                                               generated_code=report['generated_code_sha256'],
                                               exporter=sha256(Path(__file__).read_bytes()),
                                               inference=sha256(helper)),
                            resource_limits=LIMITS, resource_estimates=estimates, artifacts=files)
            (staging/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
            # Exercise the emitted standalone loader before making a bundle visible.
            helper_namespace = {'__name__': 'export_smoke'}
            exec(compile(helper, '<standalone-inference>', 'exec'), helper_namespace)
            session, contract = helper_namespace['load_bundle'](staging)
            import numpy as np
            helper_namespace['predict'](session, contract, np.zeros(shape, dtype=np.float32))
            out.mkdir()  # Exclusive reservation: even a pre-existing empty dir is a collision.
            created = True
            for source in staging.iterdir():
                source.rename(out/source.name)
        return manifest
    except BaseException:
        if created:
            shutil.rmtree(out)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--shape', nargs=4, type=int, metavar=('N', 'C', 'H', 'W'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.spec.stat().st_size > 64 * 1024:
            raise SpecError('specification exceeds 64 KiB')
        raw = json.loads(args.spec.read_text(), object_pairs_hook=unique_object)
        manifest = export_model(raw, args.checkpoint, args.shape, args.out)
    except Exception as exc:
        print(f'unet-budget export: {exc}', file=sys.stderr)
        return 2
    print(f"Exported checked float32 ONNX {manifest['input']['shape']} -> {manifest['output']['shape']} to {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

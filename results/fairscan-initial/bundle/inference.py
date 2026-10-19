"""Standalone fixed-shape CPU inference. Requires only NumPy and ONNX Runtime.

Input is a preprocessed float32 NCHW .npy array, never a photograph loader.
Output contains raw logits. No normalization, resizing, sigmoid or softmax.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

MAX_FILE_BYTES = 10 * 1024**2
MAX_ELEMENTS = 16_000_000


def load_bundle(directory, threads=1):
    import onnxruntime as ort
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError('threads must be an integer in 1..32')
    directory = Path(directory)
    path = directory/'manifest.json'
    if path.stat().st_size > 256 * 1024:
        raise ValueError('manifest too large')
    manifest = json.loads(path.read_text())
    if (manifest.get('schema_version') != 1 or manifest.get('format') != 'unet-budget-onnx'
            or manifest.get('precision') != 'float32' or manifest.get('opset') != 17
            or manifest.get('provider') != 'CPUExecutionProvider'):
        raise ValueError('unsupported manifest contract')
    for key, name in (('input', 'image'), ('output', 'logits')):
        contract = manifest[key]
        shape = contract['shape']
        if (contract.get('name') != name or contract.get('dtype') != 'float32'
                or contract.get('layout') != 'NCHW' or not isinstance(shape, list) or len(shape) != 4
                or any(type(d) is not int or d < 1 for d in shape)):
            raise ValueError('invalid fixed NCHW contract')
        import math
        if math.prod(shape) > MAX_ELEMENTS:
            raise ValueError('tensor resource limit exceeded')
    ins, outs = manifest['input']['shape'], manifest['output']['shape']
    if ins[0] != outs[0] or ins[2:] != outs[2:]:
        raise ValueError('incompatible input/output shapes')
    path = directory/'model.onnx'
    if not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError('ONNX file resource limit exceeded')
    model = path.read_bytes()
    recorded = manifest['artifacts']['model.onnx']
    if len(model) != recorded['bytes'] or hashlib.sha256(model).hexdigest() != recorded['sha256']:
        raise ValueError('ONNX artifact hash or size mismatch')
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(model, sess_options=options, providers=['CPUExecutionProvider'])
    if session.get_providers() != ['CPUExecutionProvider']:
        raise ValueError('unexpected execution provider')
    for nodes, key in ((session.get_inputs(), 'input'), (session.get_outputs(), 'output')):
        c = manifest[key]
        if len(nodes) != 1 or nodes[0].name != c['name'] or nodes[0].shape != c['shape'] or nodes[0].type != 'tensor(float)':
            raise ValueError('runtime interface does not match manifest')
    return session, manifest


def predict(session, manifest, image):
    import numpy as np
    if (not isinstance(image, np.ndarray) or image.dtype != np.float32
            or list(image.shape) != manifest['input']['shape']):
        raise ValueError('input must match fixed float32 NCHW contract exactly')
    if not np.isfinite(image).all():
        raise ValueError('input contains nonfinite values')
    logits, = session.run(['logits'], {'image': np.ascontiguousarray(image)})
    if (logits.dtype != np.float32 or list(logits.shape) != manifest['output']['shape']
            or not np.isfinite(logits).all()):
        raise ValueError('runtime returned invalid or nonfinite logits')
    return logits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('input', type=Path, help='preprocessed float32 NCHW .npy')
    parser.add_argument('--output', type=Path, required=True, help='new raw-logit .npy file')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    created = False
    try:
        import numpy as np
        if args.output.exists() or args.output.is_symlink():
            raise ValueError('output path must not exist')
        session, manifest = load_bundle(args.bundle, args.threads)
        # mmap checks the header and shape without loading an oversized tensor.
        image = np.load(args.input, mmap_mode='r', allow_pickle=False)
        logits = predict(session, manifest, image)
        with args.output.open('xb') as stream:
            created = True
            np.save(stream, logits, allow_pickle=False)
    except Exception as exc:
        if created:
            args.output.unlink(missing_ok=True)
        print(f'onnx inference: {exc}', file=sys.stderr)
        return 2
    print(f'Saved float32 raw logits {list(logits.shape)} to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

# Fixed-shape ONNX export and standalone CPU inference

Export a **matching, trained tensor state dictionary** into a checked ONNX model.
The compiler reconstructs its own generated architecture and replaces every
parameter with the supplied checkpoint. Missing/extra keys, wrong shapes, wrong
dtypes, non-tensor entries and nonfinite weights are errors. There is no
random-weight fallback, downloaded model, or training step.

## Setup and export

Use the pinned CPU validation environment in the README. Export additionally
needs `onnx==1.17.0` and `onnxruntime==1.20.1`; their dependencies are pinned in
`requirements-validation.txt`. Provision packages once from package indexes or
a local wheelhouse. Export, validation and inference make no network requests.
The dependency-free compilation command still works without these packages.

```sh
.venv/bin/python -m unet_budget export examples/circles.json \
  --checkpoint results/synthetic-weights.pt \
  --shape 1 1 33 41 --out /tmp/circles-onnx
```

`--shape N C H W` is required. It fixes **every** input dimension, including
batch size, and the output is `[N, output_classes, H, W]`. H/W may differ from the
specification's example dimensions. The original architecture report retains
those example dimensions; the manifest's input/output contracts specify the
actual export dimensions. A different runtime shape requires a new export.

The output path must **not exist**, even as an empty directory or symlink. The
exporter stages and checks all files first, then exclusively creates the output
directory. Ordinary failures remove staging and any newly created partial
output. Publication of a bundle is not atomic to concurrent readers; start
readers after successful exit. A killed process or machine failure can leave
staging/partial output for manual inspection. Existing files are never replaced.

The bundle contains:

| File | Contents |
| --- | --- |
| `model.onnx` | Float32 weights embedded in a fixed-shape opset-17 graph |
| `manifest.json` | Version-1 contract, architecture, precision, opset/IR version, dependency/Python versions, source/checkpoint/artifact SHA-256 hashes, file sizes and resource estimates |
| `architecture.json` | Unchanged compiler architecture report |
| `model.py` | Readable standalone PyTorch source for the exported architecture |
| `inference.py` | Standalone NumPy/ONNX Runtime loader and command-line example |

`onnx.checker.check_model(..., full_check=True)` checks the graph and inferred
shapes. The exporter also checks its exact input/output types and dimensions,
then runs zero-input inference through the emitted loader with
`CPUExecutionProvider`. A zero-input smoke test does not establish numerical
parity for arbitrary images; the frozen validation matrix supplies that evidence.
The manifest records SHA-256 hashes of the other artifacts, not of itself.
It is an integrity record, not a signed provenance/authenticity guarantee.

## Standalone inference, without PyTorch or this package

A ready-to-run trained example and the first unchanged held-out sample are in
[`examples/onnx`](../examples/onnx). Copy that directory to another environment:

```sh
python3 -m venv /tmp/onnx-runtime
/tmp/onnx-runtime/bin/python -m pip install -r requirements-onnx-runtime.txt
/tmp/onnx-runtime/bin/python examples/onnx/bundle/inference.py \
  examples/onnx/bundle examples/onnx/input.npy --output /tmp/logits.npy --threads 1
```

For offline provisioning, download the pinned wheels on a connected machine and
install using `pip install --no-index --find-links /path/to/wheels -r
requirements-onnx-runtime.txt`. After provisioning, no internet or account access
is needed. Runtime requirements do not include the compiler, PyTorch, or ONNX.

For your newly exported bundle, use `/tmp/circles-onnx/inference.py` and pass
`/tmp/circles-onnx` as the bundle directory. For programmatic use:

```python
import numpy as np
from inference import load_bundle, predict   # copy inference.py alongside your script

session, contract = load_bundle('/tmp/circles-onnx', threads=1)
x = np.load('preprocessed-input.npy', allow_pickle=False)
logits = predict(session, contract, x)
labels = logits.argmax(axis=1)              # mutually exclusive classes
# For one output channel: labels = logits[:, 0] > 0  (sigmoid threshold 0.5).
```

Inputs must be finite **float32 NCHW**, with the exact manifest shape. The helper
accepts noncontiguous arrays and makes them contiguous. It does not read JPEGs,
normalize intensity, reorder channels, add batch dimensions, resize, pad, or tile.
Match preprocessing used during training. This synthetic model was trained on
0/1 filled-circle masks plus Gaussian noise with standard deviation 0.20; its
inputs are not clipped or normalized. It is not a model for real photographs.

Output is a finite float32 NCHW `.npy` array of **raw logits**, without sigmoid or
softmax. Multiclass argmax is over axis 1. A one-channel model needs a threshold,
not argmax (which would always return zero). The CLI rejects output collisions
and removes a partially serialized output on ordinary failure. Input is loaded
with `allow_pickle=False` and memory mapping before checking its contract.

## Compatibility and bounds

The architecture family remains `padded-nearest-unet-v1`: padded biased
convolutions, ReLU, floor max pooling, nearest resize to each exact skip shape,
skip concatenation and raw output logits. Export supports a deliberately smaller
subset than the source compiler:

| Constraint | Export limit |
| --- | --- |
| Depth / base width | 1–4 / 1–32 |
| Input channels / output classes | 1–16 / 1–16 |
| Batch / each spatial axis | 1–16 / `2**depth`–1,024 |
| Selected parameters | At most 2,000,000 |
| Input plus sum of primitive output elements | At most 16,000,000 (includes ReLUs and concatenations) |
| Convolution multiply-accumulates | At most 2,000,000,000 per fixed input |
| Checkpoint / ONNX file | At most 10 MiB each |
| Checkpoint ZIP | At most 256 entries, at most 10 MiB total expanded contents |
| Precision / opset / provider | Float32 / 17 / CPUExecutionProvider |

Limits combine; some depth/width/shape combinations within the individual ranges
still exceed the aggregate bounds. Workload accounting is not an OS memory cap;
PyTorch, graph optimization, native workspaces and deserialization use additional
memory. Only use checkpoints and bundles from trusted sources. `weights_only=True`
and ZIP bounds reduce supported input scope; they are not a hostile-file sandbox.

A supported checkpoint is a `dict` or `OrderedDict` of CPU-loadable float32
strided tensors saved by modern `torch.save(model.state_dict(), path)` using its
ZIP format. Wrapped training checkpoints, optimizer state, serialized modules,
legacy pickle-only serialization, sparse/quantized/complex/float16/float64
tensors and external tensor files are unsupported. Parameter keys and shapes
must match exactly; shape validation cannot prove the training history of a
checkpoint with matching tensors. Provenance is tracked with hashes.

The pinned PyTorch 2.6 exporter explicitly uses `dynamo=False` (the legacy
TorchScript tracing path), with constant folding and no dynamic axes. Generated
Python input guards are specialized during tracing; the runtime instead enforces
the fixed contract. This choice keeps the pinned dependency set small and has
been verified for this family. Newer exporter defaults are not relied upon. No
claim is made for other PyTorch/ONNX/ORT versions, dynamic shapes, mixed precision,
GPU providers, quantization, custom layers, medical use or real-world accuracy.

## Verification and evidence

```sh
.venv/bin/python scripts/verify_all.py --log results/tests.log
```

This command retains the compilation, gradients, tiling and historical-result
regressions, and adds malformed-checkpoint/shape/dependency/collision/cleanup
checks, an offline installed-wheel export, a separate NumPy/ORT-only runtime,
byte-for-byte example regeneration, the [frozen parity protocol](onnx-protocol.md)
and a fresh 36-process paired benchmark. Runtime isolation asserts that PyTorch,
ONNX and the compiler cannot be imported. Python socket audit hooks reject Python networking
inside the isolated export and runtime processes. Installed local dependencies
are exposed via isolated symlink views, without package downloads or `.pth` hooks.

[Results and limitations](onnx-results.md) include measured logit error, task
metrics, timings, initialization costs and whole-process RSS. Exact output hashes
are pinned-environment evidence; timings and RSS are rerun and structurally
validated rather than required to match or improve. To collect a separate fresh
benchmark without replacing historical evidence:

```sh
.venv/bin/python experiments/benchmark_onnx.py --output /tmp/onnx-benchmark.json
```

## Existing implementations and primary references

This is a bounded packaging and verification workflow around existing tools;
it is not a new U-Net, graph compiler or inference engine. PyTorch's
[ONNX exporter tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
describes the established export/runtime workflow and the newer Dynamo exporter.
The [torch.load API](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
documents `weights_only` loading. ONNX provides the
[checker API](https://onnx.ai/onnx/api/checker.html). ONNX Runtime's
[Python inference examples](https://onnxruntime.ai/docs/api/python/tutorial.html)
and [thread management documentation](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
cover NumPy inputs, CPU sessions and thread settings. The original U-Net and
related PyTorch implementation are linked in the main README.

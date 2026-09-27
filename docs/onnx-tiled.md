# Standalone tiled ONNX inference

Run an existing batch-one fixed-shape export on a larger image using only NumPy,
ONNX Runtime and the standard library. The tile shape comes from the manifest;
there is no resizing, normalization, image decoding, or weight download. Inputs
are already preprocessed float32 NCHW `.npy` arrays. Outputs are float32 raw logits
at the original image dimensions. Classify **after** blending (argmax over channels,
or `logits[:, 0] > 0` for a one-channel binary model).

## Offline deployment example

Provision the pinned [runtime requirements](../requirements-onnx-runtime.txt)
using a local wheel directory, prepared beforehand for the target Python/platform:

```sh
python3 -m venv /tmp/unet-runtime
/tmp/unet-runtime/bin/python -m pip install --no-index --find-links /path/to/wheels -r requirements-onnx-runtime.txt
mkdir /tmp/unet-tiled-demo
cp src/unet_budget/onnx_tiled.py /tmp/unet-tiled-demo/tiled_inference.py
cp -r examples/onnx/bundle /tmp/unet-tiled-demo/bundle
cp examples/onnx-tiled/tiled-input.npy /tmp/unet-tiled-demo/input.npy
cd /tmp/unet-tiled-demo
/tmp/unet-runtime/bin/python tiled_inference.py bundle input.npy --output logits.npy --overlap 16 20 --blend gaussian --threads 1
```

Use new output/demo paths if running again. The existing trained `33×41` tile
bundle produces `[1, 2, 65, 81]` logits from `[1, 1, 65, 81]` input using nine
tiles. The sample repeats the original held-out image twice along both axes and
crops to `65×81`; it demonstrates deployment, not a new independent evaluation.
[Sample metadata](../examples/onnx-tiled/tiled-sample.json) records file/output hashes.
The bundle's Python files are hashed as inert data, **never imported or executed**.
The copied runtime file has no project-relative imports and can be placed anywhere.
PyTorch, ONNX (the exporter library), and the compiler are not deployment dependencies.

To export different trained weights, use the unchanged [export command](onnx.md)
with `--shape 1 C TILE_HEIGHT TILE_WIDTH`, then point the tiled command at that
bundle. Existing exported bundles need no modification. From an installed compiler
package the same command is available as `python -m unet_budget.onnx_tiled`; this
convenience is optional and not needed in deployment.

The isolated verifier copies the script, sample, and bundle to a temporary directory,
provisions **only** runtime distributions from local installed files, starts a fresh
venv with `-I -S`, and rejects Python socket audit events. It checks that PyTorch,
ONNX and the compiler cannot be imported, runs the documented command through
`main`, verifies exact output hashes, and tests collision handling. It is a
reproducible offline test, not an OS-level network sandbox or a cross-platform wheel test.

## Placement and blending

The grid matches [the existing PyTorch helper](tiled.md) with tile batch size one.
For axis length `L`, tile side `T`, overlap `O`, set `S=T-O`. Starts are
`0, S, 2*S, ...`, stopping at the first tile whose end covers `L`. Equivalently,
there are `1 + ceil(max(L-T, 0)/S)` starts. Tiles are visited row-major. The last
start is **not** shifted back. Bottom/right pixels beyond the image are zero-filled;
small images use a single top-left tile. Only each tile's valid image region is
accumulated, so the returned array is cropped to the original image extent.

Overlap is an integer pixel count per axis, default `(0, 0)`, strictly smaller
than the corresponding tile side. `constant` weights equal one. `gaussian`
weights are proportional to

```text
exp(-0.5 * (((y-(Th-1)/2)/(Th/8))**2 + ((x-(Tw-1)/2)/(Tw/8))**2))
```

They are normalized to a peak of one and clamped to `1e-6`. The weighted raw
logits are accumulated in float32, with a separate one-channel float32 weight sum,
then divided in place. NumPy/PyTorch exponential rounding can differ slightly;
parity uses the frozen numerical tolerances, not bitwise equivalence. No labels,
probabilities, halos, reflected padding, tile batching, asynchronous execution,
dynamic shapes, quantization or non-CPU providers are supported by this command.
Unrecognized CLI settings are rejected. Spatially downsampled outputs, non-float32
arrays, empty arrays, nonfinite inputs/outputs and batches other than one are rejected.

Overlap cannot reconstruct context missing outside a tile. Pooling restarts at
each tile origin, decoder resizing depends on tile dimensions, and padding small
images can alter intermediate features. Thus tiled predictions need not match
whole-image predictions, even away from seams. Runtime parity and these changes
are measured separately in [the results](onnx-tiled-results.md).

## Validation, resource bounds and failure behavior

The loader accepts schema 1, opset 17, float32 CPU bundles with exactly the original
four artifact records. It checks all four files' sizes and SHA-256 hashes, duplicate
JSON keys, bounded positive fixed dimensions, and the actual ONNX Runtime input/output
names, shapes and dtypes. File symlinks and unsupported inventories are rejected.
Manifest size is limited to 256 KiB and each artifact to 10 MiB. Input and output
tile tensors are each limited to 16,000,000 elements. Explicit CPUExecutionProvider,
sequential execution, one inter-op thread, and 1–32 intra-op threads (default one)
are required. Hashes detect corruption, not malicious replacement of both manifest
and files. Use trusted exporter bundles; this is not a sandbox for arbitrary ONNX.

| CLI option | Default | Checked quantity |
| --- | ---: | --- |
| `--max-image-pixels` | 16,777,216 | `H*W` |
| `--max-input-elements` | 16,000,000 | `C*H*W` |
| `--max-output-elements` | 16,000,000 | `K*H*W` |
| `--max-tiles` | 65,536 | Number of model invocations |

Limits must be positive integers and are checked before image-sized allocation or
model calls. They may be raised deliberately; they are workload guards, **not hard
memory caps**. The CLI memory-maps the input with pickle disabled and checks shape
before scanning its values in tile-sized strips. All inputs must be finite before
any model invocation. Each output tile and the normalized output are checked too.

Input storage still scales as `4*C*H*W` bytes, output accumulation as `4*K*H*W`,
and normalization as `4*H*W`. Mapping input does not eliminate disk storage, address
space or resident pages. Tile buffers, weights, multiplication temporaries, masks,
model weights/activations, runtime caches and interpreter memory add to this.
There is no streaming output, constant-total-memory claim, or hard out-of-memory
recovery guarantee. The returned accumulation is normalized in place; validation
masks use bounded strips. The caller must not mutate the input during inference.

The command requires a new output path, reserves it with exclusive creation, and
never overwrites an existing file or symlink. If serialization raises, it removes
its partial file. SIGINT/KeyboardInterrupt and CLI SIGTERM also clean up a partial
output; SIGTERM exits with status 143. Cancellation during a native runtime call
may wait until control returns to Python. SIGKILL, power loss, fatal native crashes,
or an uncooperative filesystem cannot guarantee cleanup. The output is visible
while being written: consume it only after successful command exit. Normal errors
return status 2. The library propagates cancellation and does not install signal
handlers; only execution as a script/module installs the SIGTERM handler.

## Reproduce everything

With the recorded [validation dependencies](../requirements-validation.txt) installed:

```sh
.venv/bin/python scripts/verify_all.py --log results/tests.log
```

This retains the original compiler, tiled PyTorch and fixed-shape ONNX regressions,
and adds 256 independent seeded geometric cases (both blends), malformed/oversized
input and bundle checks, output collisions/races, allocation/serialization failures,
real SIGINT/SIGTERM cleanup, the isolated deployment example, 24 frozen parity cases,
trained held-out replay without retraining, and 36 fresh tiled benchmark workers.
Historical training reproduction remains a separate unchanged regression; the new
tiled ONNX experiment only reads the saved checkpoint and regenerates held-out inputs.

[The frozen protocol](../experiments/onnx_tiled_protocol.json) was written before
parity evaluation, SHA-256
`3cea802d8c9e430ca2daa410f93bff6012e04bcc0cfc672934b2c543b73c6ce0`.
Per element, require `abs(ONNX-Torch) <= 1e-5 + 1e-5*abs(Torch)`, maximum absolute
error ≤ `1e-4`, mean absolute error ≤ `1e-5`, and label disagreement ≤ `0.001`.
All bounds must hold. No threshold is adjusted after seeing results. Failures are
saved before the experiment exits unsuccessfully. Exact numerical reproduction
requires the recorded environment; timing and RSS equality are never required.

## Related work

This integrates established sliding-window methods; no novelty is claimed.
[MONAI's implementation](https://github.com/Project-MONAI/MONAI/blob/dev/monai/inferers/utils.py)
offers constant/Gaussian blending and configurable padding/batching.
[nnU-Net's utilities](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/sliding_window_prediction.py)
compute Gaussian importance maps and image-covering tile positions. Our placement
and edge padding follow this repository's simpler pre-existing contract; numerical
compatibility with those libraries is not asserted. The standalone session uses
[ONNX Runtime's Python CPU/NumPy interface](https://onnxruntime.ai/docs/api/python/api_summary.html).
Primary implementations and runtime documentation were inspected on 2026-09-27.

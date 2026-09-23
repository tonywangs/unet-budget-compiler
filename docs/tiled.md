# Standalone tiled inference

Generate the optional helper with an otherwise unchanged specification:

```sh
.venv/bin/unet-budget examples/circles.json --out /tmp/tiled-unet --with-inference
```

The new directory contains `model.py`, `inference.py`, and `architecture.json`.
The report adds an `inference` entry with the helper's SHA-256. Default compilation
still emits exactly the original model and report. The helper is specialized to
the specification's input/output channel counts and minimum tile side `2**depth`.
Compilation reads its source without importing PyTorch. Copy the two Python files
into your application; runtime imports only PyTorch and the Python standard library.
No compiler package, network, weights download, or other inference framework is
needed. Install PyTorch beforehand as described in the README.

Run the following in the generated directory with a Python interpreter that has
PyTorch installed. Random weights demonstrate execution, not useful segmentation:

```python
import torch
from model import UNet
from inference import tiled_logits

torch.set_num_threads(1)
torch.manual_seed(17)
model = UNet().eval()               # load your trained state_dict here if available
image = torch.randn(1, 1, 65, 81)   # CPU NCHW, float32
with torch.inference_mode():
    full = model(image)
    tiled = tiled_logits(model, image, tile_size=(32, 40), overlap=(16, 20),
                         tile_batch_size=2, blend="gaussian")
labels = tiled.argmax(dim=1)       # classify AFTER blending raw logits
assert full.shape == tiled.shape == (1, 2, 65, 81)
assert labels.shape == (1, 65, 81)
```

For one output channel, use `tiled[:, 0] > 0` for a binary probability threshold
of 0.5, or apply sigmoid and a chosen threshold after blending. Do not pass an
argmax, threshold, sigmoid, or softmax predictor if you want logit blending.
`blend="constant"` gives every tile equal weight. `overlap=(0, 0)` gives a
nonoverlapping grid. Overlap is an **integer count of pixels on each axis**, not
a fraction. Rectangular tiles and different overlaps per axis are supported.

## Exact geometry and arithmetic

For axis length `L`, tile side `T`, and overlap `O`, stride is `S = T - O`.
There are `1 + ceil(max(L-T, 0)/S)` starts, at `0, S, 2*S, ...`.
The last tile is not shifted back toward the image. Its out-of-image bottom/right
pixels are filled with zero. Images smaller than a tile use a single top-left
aligned, zero-padded tile. There is no reflection, symmetric padding, halo crop,
or full-image padding allocation. Tile logits for padded pixels are discarded.
For example, length 11, side 4, overlap 1 yields starts `[0, 3, 6, 9]`; the last
tile uses two image pixels and two padding pixels. With zero overlap the starts
are `[0, 4, 8]`, and each real pixel is covered exactly once.

Tiles are traversed in image, row, column order. Up to `tile_batch_size` consecutive
tiles are copied into one zero-initialized batch, potentially spanning images.
The predictor must return a single same-resolution NCHW tensor, with the same
batch length, dtype and CPU device. Multiple output channels are accumulated
separately. A module must already be in evaluation mode, including all its
submodules; the helper does not change its state. Arbitrary callable predictors
must be batch independent and deterministic. Stateful or cross-sample operations
can depend on tile batching and are outside this contract.

For a Gaussian tile, the weight at local `(y,x)` is proportional to
`exp(-0.5 * (((y-(T_h-1)/2)/(T_h/8))**2 + ((x-(T_w-1)/2)/(T_w/8))**2))`.
Weights are normalized to a peak of 1, then clamped to at least `1e-6`.
Constant weights are exactly 1. This makes every covered pixel's normalization
strictly positive, including corners and one-pixel sides. The fixed Gaussian is
not configurable and is not claimed to reproduce another library's convention.

Each valid tile region contributes `logits * weight` to the matching image
coordinates and `weight` to a per-image, one-channel normalization map. The
returned tensor is their quotient, cropped to the original dimensions. Accumulation
uses the input dtype (float32 or float64), in traversal order, with inference mode
and no gradients. Finite input tile values, predictor outputs and final logits are
checked; an invalid value or arithmetic overflow raises `ValueError`. Input
nonfinites are discovered as their tiles are visited, potentially after earlier
predictor calls. The input is never modified. No labels are blended. Finite-precision
rounding can change with tile batch size or PyTorch kernels.

## Validation and resource limits

`tile_plan(image.shape, **settings)` validates and returns the grid without
running a predictor or allocating image-sized tensors. `tiled_logits` performs
this same validation before its first model call. Dimensions and limits must be
positive Python integers; booleans are rejected. Overlap must be nonnegative and
strictly less than the corresponding tile side. CPU strided float32 and float64
inputs, including noncontiguous tensors, are supported. Empty images, channels,
and batches, half precision, GPU tensors, spatially downsampled outputs, and
structured predictor outputs are rejected.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `tile_size` | `(256, 256)` | Height and width; each at least `2**depth` in generated helpers |
| `overlap` | `(32, 32)` | Integer height/width overlap; change this when choosing smaller tiles |
| `tile_batch_size` | `1` | Maximum tiles in one model invocation |
| `blend` | `"constant"` | `"constant"` or `"gaussian"` |
| `max_image_pixels` | `16,777,216` | Inclusive `N*H*W` limit, excluding channels |
| `max_tiles` | `65,536` | Inclusive total over all images |
| `max_tile_batch_size` | `64` | Inclusive limit on requested tile batch size |
| `max_tile_pixels` | `1,048,576` | Inclusive tile height × width limit |

Limits are explicit caller-controlled workload guards. Raising a limit does not
make a workload feasible. Full-image inference with `model(image)` does not use
these guards. `tile_plan` checks counts before building its axis-start lists.
The standalone low-level `blend_weights` utility does not apply these workload
guards; use `tile_plan` or `tiled_logits` for bounded settings.

**There is no hard process-memory cap.** For `b` bytes per element, the caller's
full input occupies `b*N*C*H*W` bytes. Returned logit accumulation occupies
`b*N*K*H*W`, and normalization occupies `b*N*H*W`, simultaneously. The helper does
not stream either full-image buffer to disk. It normalizes in place, avoiding a
second returned-logit tensor. Validation also creates temporary boolean tensors,
including an output-sized finite mask. Tile input and output storage depend on
`B*C*T_h*T_w` and `B*K*T_h*T_w`, and Gaussian weights on `T_h*T_w`.
The tile multiplication creates temporary tensors. Model activations depend on
tile dimensions, tile batch size and architecture, rather than full-image size;
model parameters, PyTorch workspaces, allocator caches, interpreter overhead and
predictor-owned state still consume memory. A custom predictor could retain
arbitrary data. Image-pixel limits do not bound channel count or total bytes.

## Why overlap does not imply full-image equivalence

Internal tile edges replace neighboring image context with zero padding.
Padding small images up to a tile can also allow intermediate features outside
the original boundary to feed back into the valid region. Pooling resets its
coordinate origin in each tile, and odd dimensions are floor-pooled. Decoder
nearest resizing depends on each local source/destination size. Thus even pixels
away from a tile edge can differ from full-image inference. Gaussian weighting
reduces contributions near tile edges; it cannot reconstruct omitted context,
restore pooling alignment, or guarantee matching labels. A single unpadded tile
with exactly the image dimensions is separately tested to match full inference.

This is an integration of established methods, not a novel inference algorithm.
[MONAI sliding-window inference](https://github.com/Project-MONAI/MONAI/blob/dev/monai/inferers/utils.py)
supports constant/Gaussian blending, tile batching, padding and more general
predictor outputs. [nnU-Net's sliding-window utilities](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/sliding_window_prediction.py)
compute Gaussian importance maps and distribute tile positions to cover images.
Our fixed-stride placement and bottom/right padding intentionally define a
simpler, different grid. Neither package is a dependency, and no numerical
compatibility with them is claimed. PyTorch documents the distinction between
[nearest and nearest-exact interpolation](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.interpolate.html).
These sources were inspected on 2026-09-23; the helper was written independently.

## Reproduction

With the recorded dependencies installed, one command runs original regressions,
independent stitching tests, offline wheel installation/standalone execution,
original synthetic reproduction, all tiled comparisons and fresh-process CPU
benchmark verification:

```sh
.venv/bin/python scripts/verify_all.py --log results/tests.log
```

For fresh artifacts in an existing directory (do not overwrite evidence you want
to keep):

```sh
.venv/bin/python experiments/compare_tiled.py --output /tmp/tiled-comparison.json --checkpoint /tmp/synthetic-weights.pt
.venv/bin/python experiments/benchmark_tiled.py --output /tmp/tiled-benchmark.json
```

`compare_tiled.py --check results/tiled-comparison.json` reruns training and
inference and compares all recorded numerical results, hashes and configurations
exactly. It checks the saved checkpoint file hash and every reproduced state tensor.
The default checkpoint is `results/synthetic-weights.pt`; it contains a PyTorch
state dict only, and the experiment loads it with `weights_only=True`. The model
architecture is generated from `examples/circles.json`. Exact reproduction is
limited to the recorded platform/dependencies, not guaranteed across machines.

`benchmark_tiled.py --check results/tiled-benchmark.json` validates saved raw
samples/summary consistency and remeasures in 36 fresh processes. It compares
output hashes, input hashes, generated model hashes, helper hash, settings and
tensor sizes, but never asserts that wall time or RSS must match or improve.
Every process has a 180-second timeout. The benchmark requires Linux because it
interprets `resource.ru_maxrss` as KiB. Console summaries during verification
report **new** measurements; the saved JSON retains the original raw samples.
The check also validates `results/tiled-benchmark-initial.json`, preserving the
first two-workload run before the padded-small workload was added.

See [tiled validation results](tiled-results.md) for accuracy, boundary effects,
measurements and limitations. These experiments use synthetic inputs only and
make no claims about real-world or medical segmentation.

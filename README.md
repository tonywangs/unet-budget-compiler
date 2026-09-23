# U-Net budget compiler

Turn a small JSON specification into readable, standalone PyTorch code and an
architecture report, without importing PyTorch during compilation. The compiler
selects the **largest listed base width that fits a parameter budget** within one
explicitly defined 2D U-Net family. It does not search for the most accurate model
or optimize latency, activation memory, or training cost.

## Install and use

Python 3.10 or newer is required. In a virtual environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/unet-budget examples/circles.json --out /tmp/my-unet
```

On Debian/Ubuntu, creating a virtual environment requires the `python3-venv`
package. The compiler has no runtime dependencies. Installation needs setuptools
and wheel; with those already available, use `pip install --no-index --no-deps
--no-build-isolation .` for an offline source installation. A prebuilt wheel can
be installed with `pip install --no-index path/to/unet_budget_compiler-0.1.0-py3-none-any.whl`.
Compilation makes no network requests and does not download weights or data.

The example selects **width 4, with 7,470 trainable parameters**, under an 8,000
parameter budget. The output directory must be new or empty. It contains:

- `model.py`: a standalone `UNet` class, importing only PyTorch.
- `architecture.json`: resolved specification, all candidate counts, selected
  width, primitive layer shapes/counts, compiler version, and SHA-256 hashes of
  the normalized specification and generated source.

Add `--with-inference` to also emit a standalone CPU `inference.py` helper with
constant or Gaussian logit blending, rectangular tiles, integer pixel overlap,
tile batching, and explicit workload limits. See the [tiled inference guide](docs/tiled.md)
for a runnable full-image/tiled example and [measured tradeoffs](docs/tiled-results.md).
Tiling bounds per-call image dimensions; it does not impose a process-memory cap
or guarantee full-image-equivalent predictions.

Install PyTorch separately to run the model; CPU validation here uses 2.6.0:

```sh
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

Copy `model.py` into your application, then:

```python
import torch
from model import UNet

model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x = torch.randn(2, 1, 33, 41)
target = torch.zeros(2, 33, 41, dtype=torch.long)
optimizer.zero_grad()
logits = model(x)                     # [2, 2, 33, 41]
loss = torch.nn.functional.cross_entropy(logits, target)
loss.backward()
optimizer.step()
```

Generated code is independent of the compiler package. Artifact generation is
deterministic; weight initialization is random unless you set a PyTorch seed.
The committed [example model](examples/generated/model.py) and
[report](examples/generated/architecture.json) are checked against regeneration.

## Specification

All fields are required; unknown fields, duplicate JSON keys, booleans in integer
fields, and out-of-range values are rejected with CLI exit status 2 and a diagnostic.

```json
{
  "input_channels": 1,
  "output_classes": 2,
  "depth": 2,
  "max_parameters": 8000,
  "width_candidates": [2, 4, 6, 8, 12, 16],
  "input_height": 33,
  "input_width": 41
}
```

| Field | Meaning and limits |
| --- | --- |
| `input_channels` | Positive integer, at most 1,024 |
| `output_classes` | Number of output logit channels, 1–1,024 |
| `depth` | Number of pooling operations, 1–8; encoder has `depth + 1` blocks |
| `max_parameters` | Inclusive trainable parameter budget, 1–10^12 |
| `width_candidates` | 1–256 integers, each 1–1,024; sorted and deduplicated |
| `input_height`, `input_width` | Example dimensions for the report, each `2**depth`–65,536 |

If no candidate fits, the error states the minimum candidate's required parameter
count and suggests increasing the budget or supplying smaller widths. Selection
exhaustively evaluates the supplied widths with integer arithmetic. It never
instantiates a tensor. The largest feasible width wins; nothing between supplied
candidates is searched. JSON whitespace, object-key order, and repeated/reordered
widths do not affect artifacts. Hashes cover canonical sorted-key, compact UTF-8
JSON after width normalization. Artifacts contain no timestamps or machine paths.

The input dimensions are a shape-report example, not a fixed runtime resolution.
Generated models accept any positive batch size and height/width at least
`2**depth`. Very large legal specifications can create impractically large models
or tensors: the parameter budget is **not a memory guarantee**. Dimensions above
the compilation limits are not promised or validated by this project.

## Architecture semantics

At encoder level `i`, the width is `base_width * 2**i`. Each block has two 3×3
stride-1 convolutions, each with a bias, one pixel of zero padding on every side,
and a non-inplace ReLU. There is no batch normalization, dropout, residual sum,
learned upsampling, or pretrained encoder. Pooling uses a 2×2 max window, stride
2, no padding, and floor rounding. Thus an odd axis loses its last row/column at
that pooling operation; the skip tensor retains its full original resolution.

The decoder resizes to the **exact corresponding skip height and width** using
PyTorch `interpolate(mode="nearest")`. This mode maps destination index `j` to
`floor(j * source_size / destination_size)`; it is not `nearest-exact`. Concatenate
skip channels first and resized channels second. Two padded 3×3 convolutions
reduce the concatenated channels to the skip width. A biased 1×1 convolution
maps the final width to `output_classes`. No cropping or post-resize padding is
used. Output is NCHW raw logits at the input resolution.

For mutually exclusive classes use cross entropy and an integer NHW target.
With one output channel, use binary cross entropy **with logits** and a matching
N1HW floating target. Apply sigmoid/softmax only as appropriate for inference;
the model does neither internally.

A biased convolution with input channels `a`, output channels `b`, and kernel
side `k` has `b * (a*k*k + 1)` parameters. A double block has
`9*a*b + 9*b*b + 2*b`. The report sums every convolution independently and checks
this total against the selection count. Pooling, resizing, ReLU and concatenation
have zero parameters. Shapes include an unspecified positive batch dimension `N`.

## Reproduce validation

The following installs the recorded CPU dependencies and runs the checks. Initial
dependency provisioning needs internet access or an existing local wheel cache;
the checks, compilation and experiment need no network or GPU.

```sh
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-validation.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python scripts/verify_all.py --log results/tests.log
```

`requirements-validation.txt` records the installed dependencies; Python 3.12.3 on
Linux x86-64 produced the committed results. Exact numerical reproduction is
checked for this environment, not promised across platforms or dependency
versions. To run a fresh experiment without comparing to recorded numbers:

```sh
.venv/bin/python experiments/train_synthetic.py --output /tmp/synthetic.json
```

Tests enumerate small budget boundaries using separately instantiated reference
models, compare counts and shapes with generated models, check invalid input and
artifact determinism, and exercise 24 seeded forward/input-gradient/parameter-
gradient cases in float32 and float64. Float32 tolerances are `atol=1e-6,
rtol=1e-5`; float64 tolerances are `atol=1e-10, rtol=1e-8`. The reference uses
module containers and integer-index resizing; it does not use the compiler's
counting or code-generation routines. Both implementations still rely on PyTorch
convolution and pooling primitives. Finite numerical tests are not a proof of
general equivalence. See [validation evidence](docs/validation.md).

`verify_isolated.py` builds a wheel without network/build isolation, installs it
into a fresh environment without PyTorch, and compiles there. A separate Python
process runs the result with no compiler on its import path. Socket operations
are rejected during compilation and execution. It checks output shape, finite
gradients, and a real optimizer update. The script requires the recorded build
and PyTorch dependencies to have been provisioned already. It also generates the
optional helper and executes full-image inference, both blending modes, exact
single-tile agreement and padding for an image smaller than a tile. The tiled
tests and experiment commands are described in the [inference guide](docs/tiled.md).

## Existing work and scope

[U-Net (Ronneberger, Fischer and Brox, 2015)](https://arxiv.org/abs/1505.04597)
established the contracting/expanding architecture with skip connections. This
project does not claim a new architecture or reproduce the original paper's
training results.

[The milesial PyTorch U-Net implementation](https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py)
provides double convolution blocks with normalization and bilinear or transposed
upsampling. This project implements a smaller family with biases, no
normalization, and nearest resizing directly to skip dimensions. Its purpose is
a bounded, inspectable specification-to-code workflow, not a replacement for a
full segmentation training library. Code here was written for this family; it
is not a copy of that implementation.

PyTorch defines the underlying
[Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html),
[MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html),
and [interpolate](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)
semantics. No claims are made about medical applicability, real-world accuracy,
GPU behavior, export to ONNX, mixed precision, TorchScript, or `torch.compile`.
The synthetic experiment is a small CPU integration demonstration only.

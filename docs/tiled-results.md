# Tiled inference: validation and measured tradeoffs

Results recorded on 2026-09-23 with Python 3.12.3, PyTorch 2.6.0+cpu and NumPy
2.2.6, Linux x86-64, one PyTorch CPU thread and deterministic algorithms. The
virtual CPU reports `DO-Regular`, with four logical CPUs available to the process.
No GPU, paid inference, private dataset, or downloaded weights were used.

The [comparison JSON](../results/tiled-comparison.json) preserves every generated
case and held-out method, inputs/configurations/seeds, dataset/sample hashes,
checkpoint file/tensor hashes, logit hashes and boundary metrics. The
[benchmark JSON](../results/tiled-benchmark.json) preserves raw timings and RSS,
workload/model/helper provenance, thread settings, hardware and dependency/build
information. The [initial two-workload run](../results/tiled-benchmark-initial.json)
is also retained and checked; it preceded addition of the padded-small case and
its raw samples were not discarded. [Plain verification output](../results/tests.log) is produced by the
[one-command verification procedure](tiled.md#reproduction).

## Independent stitching checks

Six inference test methods supplement the original eight compiler test methods.
The main check uses seed 8142 for **256 geometries**, each in both blending modes,
with 1–2 images, 1–3 input channels, odd/rectangular sizes from 1×1 up to 39×43,
tile sides 1–27 and 1–29, legal integer overlaps, and tile batches 1–9. Half use
float64 and half float32. It tests four predictor families for each blend:
constant four-channel output, identity, affine/channel-sum pointwise output, and
a tile-local coordinate predictor whose output changes across overlapping tiles.

The reference walks each axis until covered, computes Gaussian weights with
scalar Python `math.exp`, and accumulates each valid pixel with NumPy arrays.
It does not call the helper for expected positions, weights, counts, coverage,
normalization or outputs. It checks positive normalization and complete coverage;
zero-overlap grids cover each pixel once. Coordinate-encoded inputs and a separate
padding/batch-order test check exact zero padding, crop placement and multi-image
traversal. Output tolerances are `rtol=5e-6`, `atol=3e-5` for float32 and `1e-10`
absolute for float64 (the same relative tolerance); weight-map tolerances are
`rtol=3e-6`, `atol=1e-9`. Both implementations use finite-precision arithmetic;
this is numerical evidence, not a formal proof.

Other checks cover inclusive limits and exhaustion before predictor invocation,
extreme shape/count plans without allocating images, malformed settings/tensors,
training-mode modules, wrong predictor outputs, changing channel counts, invalid
dtype/device, nonfinite values and accumulation overflow. One explicit example
shows that blending logits yields a different class from equally voting tile
labels. CLI tests check deterministic helper hashes, specialization and unchanged
model bytes. Isolated offline wheel installation runs both blends, full-image
inference, exact one-tile agreement, and a 1×3 image padded to an 8×12 tile without
an importable compiler package. Original training/optimizer checks remain intact.

## Generated-model comparisons

There are **324 cases**: weight seeds 17/41/93, depths 1/2/3, width 3, two input
and three output channels, shapes 31×37 / 48×64 / 65×53, tiles 16×20 / 25×24,
overlap fractions 0 / 0.25 / 0.5 floored to integer pixels, and both blends.
Input seed is 5701; tile batch size is 3. A further **27 cases** use one exact,
unpadded full-image tile and match full inference bit-for-bit.

Each case records maximum/mean absolute logit errors and label disagreement,
both globally and inside/outside a tile-boundary region. That region is the union
of two pixels on either side of every internal tile start and clipped tile end.
The outer image perimeter is not itself an internal boundary. The same definition
is applied to the trained-model comparisons; boundary pixel counts are retained.

Across the 108 overlapping cases per blend, the largest maximum absolute logit
error was **0.036769 for constant blending** and **0.034551 for Gaussian blending**.
None of those untrained overlapping cases changed argmax labels. This does not
establish equivalence: their logits differ, and the trained experiment changes
both logits and labels. For example, seed 17, depth 1, image 31×37, tile 16×20,
overlap 8×10, Gaussian blending has maximum error 0.007520 and mean error 0.000540.
Geometry-dependent pooling, nearest resizing, missing context and edge padding
are described in the [inference semantics](tiled.md#why-overlap-does-not-imply-full-image-equivalence).

## Same-weight held-out segmentation

The experiment reproduces the original 7,470-parameter model and its training
exactly: 48 training and 16 disjoint evaluation images, 33×41 noisy synthetic
circles, 24 epochs, fixed train/evaluation/model/shuffle seeds 101/202/303/404.
It verifies every reproduced state tensor against the saved
[checkpoint](../results/synthetic-weights.pt), and verifies the original training
curve, dataset hashes, held-out predictions and weight hash. No method is
retrained, and tile settings are not selected through held-out optimization.

All tiled methods use 16×20 tiles and batch size 4. Nonoverlapping inference uses
zero overlap; constant and Gaussian overlapping methods use 8×10 overlap.
Classification and metrics are computed after blending logits. Scores aggregate
all held-out pixels; foreground IoU and Dice refer to class 1.

| Method | Cross entropy | Pixel accuracy | Foreground IoU | Foreground Dice | Label disagreement vs full |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full image | 0.013708 | 0.995103 | 0.952188 | 0.975508 | 0 |
| Nonoverlapping | 0.040777 | 0.991778 | 0.920536 | 0.958624 | 0.005728 |
| Overlap, constant | 0.025221 | 0.993071 | 0.932219 | 0.964920 | 0.003788 |
| Overlap, Gaussian | 0.012814 | 0.995519 | 0.955949 | 0.977479 | 0.000970 |

| Tiled method | Global mean logit error | Global max error | Boundary mean error | Interior mean error |
| --- | ---: | ---: | ---: | ---: |
| Nonoverlapping | 1.657202 | 29.013084 | 4.614151 | 0.088414 |
| Overlap, constant | 1.749981 | 20.577852 | 2.607106 | 0.129134 |
| Overlap, Gaussian | 0.450603 | 20.011469 | 0.683012 | 0.011110 |

Constant overlap improves IoU over nonoverlapping tiles here, but its **global
mean logit error is worse**. Both have worse held-out IoU than full-image inference.
Gaussian overlap slightly improves the held-out IoU in this one experiment, while
still disagreeing with full inference and retaining a maximum logit error above
20. These easy inputs encode the target in noisy intensity. One seed and 16
synthetic evaluation images support no claim of general accuracy improvement,
medical utility, real-world performance or statistical significance. All methods,
including unfavorable results, remain in the saved JSON. The original training-set
frequency baseline has foreground IoU 0 and is retained in the comparison artifact.

## CPU measurements

Each workload/method pair has three independent fresh interpreter processes.
Within each process, one warmup is followed by three timed calls. Seeds are 2901
for model weights and 2902 for input; intra-op and inter-op thread counts are 1.
Method order is rotated between process repetitions. Every process constructs the
same model/input for its workload; outputs are hashed and must agree across its
calls and repeated processes. There are 36 processes and 108 timed calls total.

The three workloads are: padded-small, 17×21 with width 4/depth 2 and 64×80 tiles;
small, 65×81 with width 4/depth 2 and 32×40 tiles; large, 513×641 with width 12/depth
3 and 128×160 tiles. All have one input image/channel, two output channels,
float32 CPU tensors and tile batch size 1. Overlap is half each tile side except
in the nonoverlapping method. For padded-small all tiled modes use one padded
tile, so their overlap settings cannot provide additional context.

Wall time uses `perf_counter_ns` around inference only, excluding the benchmark
script's hashing and output validation. The tiled API's own validation, copying
and stitching are inside the timed call; full inference times `model(image)`. Reported time is the median of nine calls. Peak RSS uses Linux
`ru_maxrss * 1024` for each entire process, including imports, setup, warmup,
measured calls and checks; its median is over three processes. Setup high-water
RSS is also saved, but subtraction is not presented as activation memory. The
host is shared, CPU load is not isolated, and allocator caches and library
workspaces are included. Small RSS differences can be ordinary process noise.

| Workload | Method | Median time (ms) | Median process peak RSS (MiB) | RSS / full | Time / full |
| --- | --- | ---: | ---: | ---: | ---: |
| padded_small | full | 1.371 | 208.95 | 1.000 | 1.00 |
| padded_small | nonoverlapping | 6.020 | 216.67 | 1.037 | 4.39 |
| padded_small | overlap_constant | 5.936 | 216.75 | 1.037 | 4.33 |
| padded_small | overlap_gaussian | 6.133 | 217.89 | 1.043 | 4.47 |
| small | full | 5.776 | 215.62 | 1.000 | 1.00 |
| small | nonoverlapping | 23.748 | 210.68 | 0.977 | 4.11 |
| small | overlap_constant | 40.148 | 210.66 | 0.977 | 6.95 |
| small | overlap_gaussian | 41.440 | 211.86 | 0.983 | 7.17 |
| large | full | 775.476 | 442.21 | 1.000 | 1.00 |
| large | nonoverlapping | 1478.016 | 237.02 | 0.536 | 1.91 |
| large | overlap_constant | 3603.017 | 236.63 | 0.535 | 4.65 |
| large | overlap_gaussian | 3793.462 | 238.21 | 0.539 | 4.89 |

**Failure to reduce memory:** all padded-small tiled modes increase median process
RSS, by 3.7–4.3%, and take 4.3–4.5 times as long. A tile much larger than its
image can make both resource measures worse. On the small workload the apparent
RSS reduction is only 1.7–2.3%, too small to assert a reliable saving on this shared
host. On the large workload the recorded medians are about 54% of full-image RSS,
but nonoverlapping tiles take 1.91 times as long and overlapping tiles take
4.65–4.89 times as long. There is no general speed or memory win. The earlier
saved two-workload run also shows slower tiled execution; all its raw samples
remain available. No measurement is dropped because it is unfavorable.

Tensor storage and measured process RSS are different quantities. In the large
workload, the full input has 1,315,332 bytes and output has 2,630,664 bytes. Tiled
inference additionally keeps a 1,315,332-byte full-image normalization map. Its
single-tile input is 81,920 bytes, output 163,840 bytes, and weight map 81,920
bytes. These are array sizes, not measured peaks or complete activation estimates.
The output shapes and each component size are stored per sample. Full-image input,
accumulation, normalization, validation temporaries and all model/allocator state
still coexist as described in the inference guide. The measured hundreds of MiB
include the interpreter, dependencies, model, activations and workspaces.

The CPU emits an NNPACK unsupported-hardware warning. PyTorch uses its available
CPU backend. Other CPUs, thread settings, architectures, image sizes, tile batches,
versions or devices can change both timings and memory. Reproduction verifies
numerical outputs and saved evidence, not matching performance numbers. This is
a bounded synthetic CPU comparison, not a deployment capacity recommendation.

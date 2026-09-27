# Tiled ONNX CPU results

The standalone path passed the frozen random-model parity matrix and the original
trained synthetic checkpoint replay. This establishes bounded runtime compatibility,
not equivalence between tiled and whole-image inference or real-world segmentation
accuracy. The prior fixed-shape ONNX tolerance failure is still preserved.

## Protocol and geometry

The pre-evaluation [protocol](../experiments/onnx_tiled_protocol.json) fixes 24
cases: depths 1–4, widths 2–4, one/three input channels, one/two output channels,
constant/Gaussian blending, odd rectangular tile shapes, exact-size, narrow and
larger-than-tile images, zero or roughly half-tile overlap, and recorded weight/input
seeds. All cases use identical PyTorch/ONNX weights and tile batch size one.
Tolerances are the original fixed-shape bounds, frozen without adjustment:
`atol=rtol=1e-5`, maximum absolute error ≤ `1e-4`, mean absolute error ≤ `1e-5`,
and label disagreement ≤ `0.001`. Every element must satisfy the relative/absolute
bound too. The protocol SHA-256 is
`3cea802d8c9e430ca2daa410f93bff6012e04bcc0cfc672934b2c543b73c6ce0`.

[Geometric tests](../tests/test_onnx_tiled.py) independently verified 256 seeded
cases with both blends. A scalar float64 pixel-gather oracle constructs starts
with a stopping loop and computes weights independently. Its predictor mixes
input values, local coordinates, and a neighboring pixel to expose wrong placement,
normalization and zero padding. Every tile passed to the predictor is separately
checked against a zero-filled reference; output shape and input immutability are
checked too. Cases include one-pixel axes/tiles, images smaller than tiles, odd
sizes, exact boundaries and overlaps near the tile size. Separate identity,
constant-output and single-tile tests check normalization and coverage. Oracle
bounds (`atol=2e-5`, `rtol=2e-6`) were also frozen in the protocol.

The 24 model cases have maximum absolute runtime error **1.1920929e-7**, no
label disagreement, and no tolerance failure. Per-case inputs, weights, generated
source and ONNX hashes, mean/max logit errors, worst scaled errors and label
comparisons are in [the raw comparison](../results/onnx-tiled-comparison.json).

## Trained checkpoint replay

The new experiment does **not retrain**. It loads `results/synthetic-weights.pt`,
regenerates the same 16 held-out `33×41` inputs with seed 202, and checks the original
input, target, weight, full-image logit hashes and task metrics. The tile is `16×20`.
All three methods predict each image separately in both runtimes. Each uses the
same tiles, padding, order, blend convention, and float32 accumulation.

| Method | Max runtime error | Mean runtime error | Max error / elementwise bound | Label disagreement |
| --- | ---: | ---: | ---: | ---: |
| No overlap, constant | 2.28882e-5 | 2.63930e-6 | 0.39003 | 0 |
| Overlap 8×10, constant | 2.28882e-5 | 2.14606e-6 | 0.38068 | 0 |
| Overlap 8×10, Gaussian | 2.28882e-5 | 3.07150e-6 | 0.46354 | 0 |

All runtime comparisons pass, including every element. PyTorch and ONNX task
metrics agree in this environment. No hyperparameters were selected on these
held-out inputs. The following are **task metrics**, not runtime-error statistics:

| Prediction | Foreground Dice | Foreground IoU | Pixel accuracy | Cross entropy |
| --- | ---: | ---: | ---: | ---: |
| Whole image, original PyTorch | 0.975508 | 0.952188 | 0.995103 | 0.0137081 |
| Tiled, no overlap, constant | 0.958624 | 0.920536 | 0.991778 | 0.0407766 |
| Tiled, overlap, constant | 0.964920 | 0.932219 | 0.993071 | 0.0252214 |
| Tiled, overlap, Gaussian | 0.977479 | 0.955949 | 0.995519 | 0.0128145 |

Gaussian blending slightly improves this particular synthetic task's Dice; this
is not evidence that it generally improves segmentation. Constant blending loses
accuracy here. The whole-image baseline and negative changes remain in the report.

Seam regions are the union of pixels within the half-open band `[edge-2, edge+2)`
of internal tile starts and clipped ends, matching the historical definition. Outer
image edges alone are excluded. For each method, the raw JSON reports seam and
interior results for **both** runtime parity and tiled-versus-whole-image differences.

| Method | ONNX–Torch seam mean error | ONNX tiled–whole seam mean error | Tiled–whole interior mean error | Tiled–whole label disagreement |
| --- | ---: | ---: | ---: | ---: |
| No overlap, constant | 1.98019e-6 | 4.61415 | 0.0884153 | 0.00572801 |
| Overlap, constant | 1.92620e-6 | 2.60711 | 0.129135 | 0.00378788 |
| Overlap, Gaussian | 3.10075e-6 | 0.683013 | 0.0111099 | 0.000970067 |

The tiled-versus-whole differences are orders of magnitude larger than runtime
errors; they arise from missing context, padding, pooling origin and local resizing.
Overlap does not guarantee whole-image-equivalent logits or labels.

## Performance measurement

[The benchmark](../experiments/benchmark_onnx_tiled.py) compares complete tiled
inference APIs on the same saved trained model, fixed `33×41` tile, `16×20` overlap,
Gaussian blend, seed 9001 inputs and three image sizes. Each method has three
warmups and seven timed calls per fresh worker, with six paired repetitions per
size (36 workers). Order alternates Torch/ONNX and ONNX/Torch. Timing includes input
checks, tile construction, model calls, accumulation and normalization, but excludes
file IO and initialization. Initialization, imports and model/input setup are
recorded separately. The runtimes' validation overheads differ.

Each worker is launched by a fresh standard-library-only supervisor, avoiding
inheriting the exporting parent's PyTorch peak RSS. The ONNX worker asserts that
PyTorch, ONNX and the compiler have not been imported. Linux `ru_maxrss * 1024`
includes imports, setup, allocator caches and all calls; it is not incremental
activation memory. Both runtimes use one intra-op and one inter-op thread;
OMP/MKL/OpenBLAS are also set to one. Raw per-call latencies, warmups, worker
initialization and process wall times, startup/setup/final peak RSS, artifact
hashes, CPU affinity/quota/build details and paired ratios are retained in
[the benchmark JSON](../results/onnx-tiled-benchmark.json). Timestamps and numerical
hashes do not substitute for latency measurements.

| Image H×W | Torch median ms | ONNX median ms | Median paired ONNX/Torch | Torch peak MiB | ONNX peak MiB | Torch init ms | ONNX init ms | Slower ONNX pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17×23 | 3.116 | 0.894 | 0.283 | 244.88 | 53.24 | 3158.9 | 152.2 | 0/6 |
| 65×81 | 23.591 | 6.187 | 0.262 | 245.02 | 53.37 | 3305.0 | 148.3 | 0/6 |
| 129×161 | 118.448 | 33.196 | 0.286 | 245.37 | 53.62 | 3249.3 | 161.3 | 0/6 |


Times are medians of all 42 timed calls per method/size; peak RSS and initialization
are medians over six processes. Ratios are medians of the six paired process-median
ratios, which need not equal the ratio of the pooled medians. These are shared-host
measurements on the CPU/build/affinity/quota recorded in JSON, not guaranteed speedups.
No paired median slowed down in this run, but outliers remain: for `17×23`, one
ONNX call took 5.573 ms, longer than the maximum Torch call of 3.692 ms. All raw
samples are retained. Tests explicitly accept slower paired timings as valid evidence.

RSS is dominated by runtime overhead at these small sizes. It does **not** establish
constant total memory: full input, accumulated output and normalization still grow
with image area. This comparison is Torch-tiled versus ONNX-tiled, not a claim
that tiled inference beats whole-image inference. The original
[PyTorch tiled benchmark](tiled-results.md) records that separate tradeoff.
All 18 paired numerical comparisons passed the same frozen bounds, with identical
labels. The largest absolute error was `3.81470e-5`; the worst elementwise error
was `0.973003` times its bound, so the largest workload has little margin at that
coordinate. No bounds were relaxed. Timing/RSS improvements are never used as
verification gates.

### Retained initial benchmark and known failures

The first tiled benchmark is preserved as
[`onnx-tiled-benchmark-initial.json`](../results/onnx-tiled-benchmark-initial.json),
with [benchmark source](../results/onnx-tiled-benchmark-initial-source.txt) and
[earlier runtime source](../results/onnx-tiled-benchmark-initial-runtime.txt).
A stricter manifest opset type check was added while it ran: small/medium workers
used the earlier runtime source, large workers used the new source. This did not
change valid-model predictions, but the recorded final source hash did not identify
all workers. It is unsuitable as final source-consistent evidence. None of its raw
samples or numerical results were discarded. The corrected benchmark captures
immutable runtime source bytes before measurement and checks each workload's
artifact hashes against them. The final JSON above was generated again from the
consistent source; tests reject the initial file under the stronger provenance rule.

The **previous** fixed-shape `129×161` whole-image case still exceeds its frozen
elementwise tolerance at one logit, with unchanged labels. Its original results,
failed bounds, optimization/float64 investigation, and superseded RSS measurements
are untouched: see [ONNX numerical results](onnx-results.md) and
[the investigation](../results/onnx-numerical-investigation.json). Passing tiled
cases do not erase that failure or justify relaxing its tolerance.

## Reproduction and limits

Run the single command from [the guide](onnx-tiled.md):

```sh
.venv/bin/python scripts/verify_all.py --log results/tests.log
```

The parity `--check` reruns all cases and requires exact saved numerical evidence,
protocol and source hashes. The benchmark `--check` validates the saved measurements,
reruns 36 fresh workers, and compares numerical/configuration evidence without
requiring timing or RSS equality. Console timing summaries during checks are new
measurements; the checked-in JSON is not overwritten. Fresh experiments can be
saved to new paths with `--output`.

Evidence is limited to Python 3.12.3, NumPy 2.2.6, PyTorch 2.6.0+cpu, ONNX 1.17.0,
ONNX Runtime 1.20.1 and the recorded Linux CPU environment. Runtime code requires
Python 3.10+; deployment on other versions/platforms is not tested here. The data
are synthetic, the trained model is tiny (7,470 parameters), and one training seed
cannot establish generalization or medical utility. Images larger than the model
input work within explicit workload limits, but arbitrarily large images and hard
memory guarantees are outside this implementation.

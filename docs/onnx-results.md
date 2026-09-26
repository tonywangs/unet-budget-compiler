# ONNX numerical and CPU deployment results

The fixed-shape export and standalone inference use case works in the pinned
CPU environment. All 24 random-weight matrix cases and the trained held-out
replay passed the predeclared tolerances. The additional 129×161 benchmark input
**failed the elementwise tolerance at one logit**, with no changed labels.
That negative result remains in the evidence and is investigated below.

## Reproducible inputs and numerical comparisons

The [frozen protocol](onnx-protocol.md) and
[machine-readable configuration](../experiments/onnx_protocol.json) were written
before the comparisons. [Raw parity results](../results/onnx-comparison.json)
cover depths 1–4, input channels 1/3, output channels 1/2, base widths 2/4, batches
1/2, rectangular even and odd dimensions, and weight seeds 17/29/43. All 24
random cases had identical labels; the largest absolute logit error was
`5.960464477539063e-08`. This is sampling of the supported family, not exhaustive
validation of every admitted architecture or input.

The trained case uses the unchanged 7,470-parameter circle model and all 16
original held-out examples at shape `[16, 1, 33, 41]`. It loads the existing
checkpoint without training and checks original per-sample, dataset, parameter,
and PyTorch logit hashes before comparison. Identifiers include:

- Checkpoint file SHA-256: `d469d95dfba0cde7b832aca42ed6df5a2a79540b96edb490286fade81bfc4633`
- Held-out image/target tensor digest: `6ee7156ea61b838fde5f18e68b66e9681d98d1859a06c407166007e9b80f9a44`
- Parameter tensor digest: `bd1a23fb71cee25ea5ec6ce92135ea49c44f465edea2bcf8a881151c15c68137`

Tensor digests retain the historical dtype/shape/bytes format documented by
`train_synthetic.py`; they differ from file hashes. The result also records
raw-array input/target hashes, both runtime logit hashes, and every sample hash.
The first held-out sample is provided in the standalone example.

| Trained held-out measure | PyTorch | ONNX Runtime |
| --- | ---: | ---: |
| Cross entropy | 0.013708069920539856 | 0.01370807085186243 |
| Pixel accuracy | 0.9951034784317017 | 0.9951034784317017 |
| Foreground IoU | 0.9521876409562472 | 0.9521876409562472 |
| Foreground Dice | 0.9755083179297597 | 0.9755083179297597 |

Trained held-out maximum absolute logit error was `1.9073486328125e-05`, mean
absolute error `2.9000768157311405e-06`, with **zero label disagreement**. Every
logit passed `atol=1e-5, rtol=1e-5`, and both aggregate limits passed. Metrics
are computed by the same PyTorch metric function on each runtime's raw logits
inside the experiment; standalone inference itself does not require PyTorch.
No evaluation outcome selected a model, changed weights, or changed tolerances.

## Additional benchmark input: a retained numerical failure

The trained model on a seed-9001 Gaussian input of shape `[1, 1, 129, 161]`
produced finite, correctly shaped outputs and identical labels. Maximum error
was `3.814697265625e-05` and mean error `4.311402280017611e-06`, both within their
aggregate limits. However, logit `[0, 1, 94, 51]` differed:

| Value | Number |
| --- | ---: |
| PyTorch float32 | -0.10544267296791077 |
| ONNX Runtime float32 | -0.10545620322227478 |
| Absolute error | 0.000013530254364013672 |
| Allowed elementwise error | 0.000011054426729679109 |
| Error / allowed bound | 1.2239670762561952 |

One of 41,538 logits failed. It reproduced exactly in all six paired processes.
The benchmark records `within_tolerance: false`; a successful reproduction
check is **not** a numerical-parity pass for this workload. The separate
[investigation](../experiments/investigate_onnx.py) and
[raw diagnostic evidence](../results/onnx-numerical-investigation.json) retain
this case without loosening the thresholds or changing the default runtimes.

| Diagnostic path | Elementwise violations vs default PyTorch float32 | Max error vs PyTorch float64 | Mean error vs PyTorch float64 |
| --- | ---: | ---: | ---: |
| PyTorch default | 0 | 3.22761e-5 | 3.79907e-6 |
| PyTorch, MKLDNN disabled | 2 | 3.02527e-5 | 3.76237e-6 |
| ONNX Runtime default | 1 | 2.39633e-5 | 3.18848e-6 |
| ONNX Runtime, graph optimizations disabled | 3 | 3.31505e-5 | 3.82685e-6 |

Disabling optimizations did not resolve the mismatch; even another PyTorch
float32 path failed against the default float32 reference. Default ONNX Runtime
was closer to the float64 diagnostic reference in these aggregate measures.
These observations support float32 accumulation differences as an explanation,
but do not isolate a particular native kernel or prove exact real-number
correctness. Float64 PyTorch is itself a numerical reference, not an exact oracle.
The frozen tolerance is therefore **not a universal guarantee for the supported
shape range**, even when labels agree. Check deployment-specific inputs.

## Warmup-separated, paired CPU measurements

[Raw final measurements](../results/onnx-benchmark.json) contain 36 fresh workers:
three shapes × six paired repetitions × two runtimes. Each worker runs three
warmups and seven timed calls. Pair order alternates, with three pairs per order.
Both methods use the identical existing trained checkpoint and NumPy input bytes.
Runtime calls use one intra-op thread, one inter-op thread; OMP/MKL/OpenBLAS
thread environment values are also 1. ONNX uses sequential execution and its
default graph optimizations. No speed or memory improvement is a pass condition.

Recorded environment: Linux x86-64, Python 3.12.3, PyTorch 2.6.0+cpu, NumPy 2.2.6,
ONNX 1.17.0 and ONNX Runtime 1.20.1. The virtual host reports four logical CPUs,
model string `DO-Regular`, and affinity `[0, 1, 2, 3]`. Build strings and thread
settings are stored in the JSON. Hardware identity and contention beyond that
virtual CPU description are not characterized. PyTorch reported unsupported
NNPACK hardware; worker stderr is retained.

| Shape (NCHW) | PyTorch median call, ms | ONNX median call, ms | Median paired ONNX/PyTorch latency | PyTorch median peak RSS, MiB | ONNX median peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1×1×4×5 | 0.882 | 0.0805 | 0.0919 | 239.75 | 52.75 |
| 1×1×33×41 | 1.956 | 0.545 | 0.2741 | 240.55 | 52.94 |
| 1×1×129×161 | 14.805 | 6.845 | 0.4583 | 250.33 | 58.38 |

Call medians pool the 42 post-warmup calls per method/workload; paired ratios
instead divide each pair's process medians and then take the median of the six
ratios. Thus the ratio column need not equal a ratio of the pooled medians.
Every call, process median, range and paired ratio remains in the raw result.

ONNX was faster in all 18 recorded pairs: **no slower ONNX pair occurred in this
run**. That is an observation, not an acceleration promise. Paired latency-ratio
ranges were 0.059–0.149 (tiny), 0.260–0.354 (trained), and 0.411–0.496 (larger).
The larger case's speed result must be read with its failed numerical tolerance.
The experiment uses one small architecture, synthetic/Gaussian inputs, a shared
virtual host and one thread. It establishes neither statistical significance nor
performance on GPUs, real data, larger models or other dependency versions.

| Workload | PyTorch median initialization, s | ONNX median initialization, s | PyTorch median process wall time, s | ONNX median process wall time, s |
| --- | ---: | ---: | ---: | ---: |
| tiny | 3.243 | 0.144 | 4.382 | 0.519 |
| trained | 3.148 | 0.160 | 4.368 | 0.588 |
| larger | 3.516 | 0.161 | 4.896 | 0.686 |

Initialization includes runtime imports, model/session loading and input setup;
import and setup components are recorded separately. Parent-observed process
wall time includes supervisor/interpreter startup and exit, checks and all calls.
The call timings exclude preprocessing, validation, hashing and serialization.
RSS includes imports, setup, caches, workspaces and checks; it is not incremental
activation memory. A stdlib-only supervisor avoids inheriting the exporter's
large RSS high-water mark. Worker startup RSS was 22.25–22.375 MiB, below each
runtime's setup RSS. Runtime workers never import the compiler; ONNX workers
also never import PyTorch or ONNX.

The original completed run and exact script snapshot are preserved in
[`onnx-benchmark-initial.json`](../results/onnx-benchmark-initial.json) and
[`onnx-benchmark-initial-source.txt`](../results/onnx-benchmark-initial-source.txt).
Its identical per-pair RSS values were inherited from the torch-loaded parent
and are **not valid comparative memory measurements**. It also overlapped a
brief parity replay at startup. The table above uses only the subsequent
supervised run. This correction changed the measurement harness, not weights,
inputs, tolerances, runtime options, warmups, calls, workloads, or pair ordering.

## Artifact sizes and verification

For batch 1 at 33×41, `model.onnx` is **35,261 bytes**, versus **36,500 bytes** for
the supplied checkpoint. The 2,347-byte generated model source, 4,929-byte
standalone helper, 9,906-byte architecture JSON and 13,360-byte manifest are
additional files. The manifest and benchmark record sizes and SHA-256 hashes;
these are serialized artifact sizes, not installed dependency or process sizes.

The [single verification command](onnx.md#verification-and-evidence) runs all
existing regressions, failure-path tests, installed-wheel export, a separate
offline NumPy/ORT-only runtime, byte-exact example regeneration, the strict
25-case parity matrix, the known-failure investigation, and all 36 fresh
benchmark workers. Saved timings/RSS are checked structurally and summarized
again; rerun values may vary and are not required to improve. All benchmark
numerical failures must exactly match the separately reproduced investigation;
new or changed failures cause verification to fail. The preserved initial
source hash and raw summaries are also checked.

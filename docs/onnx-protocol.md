# Fixed ONNX validation protocol

Frozen before final comparisons: [machine-readable protocol](../experiments/onnx_protocol.json).
It fixes 24 random-weight cases (eight depth/channel/shape configurations times
three seeds) and one replay of the existing trained model on all 16 held-out
synthetic examples. No training, parameter search, tolerance adjustment, or
selection based on evaluation scores is part of this experiment.

Every comparison requires equal shape, finite float32 logits, elementwise
`abs(ONNX - torch) <= 1e-5 + 1e-5 * abs(torch)`, maximum absolute error <= 1e-4,
mean absolute error <= 1e-5, and label disagreement <= 0.001. Binary single-logit
labels use `logit > 0`; multi-class labels use argmax. Any failure stops the
comparison with the case and metrics, for investigation; a result cannot pass
by omitting a failing case. These are numerical engineering tolerances, not
accuracy guarantees. Random weights do not establish task accuracy.

Each benchmark workload uses the *same existing trained checkpoint* for both
runtimes, with a seed-9001 NumPy float32 normal input. Six consecutive pairs of
fresh processes alternate execution order (three pairs in each order). Every
process makes three untimed warmups, then seven timed calls with one compute
thread and one inter-op thread. Initialization and import times are separate.
Both use direct model/session calls on prevalidated inputs; preprocessing,
NumPy validation, hashing and serialization are outside the measured calls.
Linux peak RSS includes the whole process: imports, setup, warmup, calls and
checks. Parent-observed wall time also includes interpreter startup/exit.

Preserve every timing, initialization cost, peak RSS, input/output hash,
artifact size, process order, environment and dependency version. Report paired
ratios, medians and ranges; impose no speedup or memory-improvement threshold.
Shared-host measurements with six pairs do not prove statistical significance
or generalize to other shapes, models, machines, threading choices or GPUs.

## Recorded deviation and negative result

The first benchmark attempt stopped on its first 129×161 comparison: one logit
exceeded the elementwise bound (all shape/finite/aggregate-error/label checks
passed). That interrupted attempt did not produce a timing-results file. The
thresholds, matrix, inputs, weights and runtime defaults remain unchanged.
The completed benchmark retains the failing comparison for all six pairs and
all raw timing/RSS samples. A passing *reproduction check* for the benchmark
means the recorded evidence, including this negative result, was reproduced;
it does not mean all benchmark logits passed tolerance. The original 25-case
parity matrix still strictly requires every numerical threshold to pass.

Every benchmark tolerance failure must exactly match the separately reproduced
[numerical investigation](../results/onnx-numerical-investigation.json). A new
failure causes verification to fail. The investigation compares graph
optimizations disabled, PyTorch MKLDNN disabled, and a float64 reference. It is
diagnostic only; it does not replace the default benchmark or tune it for speed
or agreement. See [the results](onnx-results.md) for interpretation and limits.

The first completed timing run is retained as
`results/onnx-benchmark-initial.json`, with its exact source snapshot in
`results/onnx-benchmark-initial-source.txt` (matching its recorded source hash).
Its RSS measurements are **invalid for comparing runtime memory**: direct
fork/exec from the torch-loaded exporter inherited the parent's high-water mark,
giving identical RSS for both methods. That run also overlapped a brief parity
replay at startup. Its timings are retained but not used for final conclusions.

Final measurements insert a lightweight stdlib-only supervisor between the
exporter and measured workers. Workers are forked from this small process, and
record startup, post-setup and final peak RSS separately. Verification requires
setup RSS to exceed startup RSS, preventing the observed inherited-RSS plateau.
Parent-observed wall time now includes the extra supervisor startup/exit; import,
model/input initialization and inference timings remain internal to the worker.

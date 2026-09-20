# Validation evidence

Validation on 2026-09-20 used Python 3.12.3, PyTorch 2.6.0+cpu, NumPy 2.2.6,
Linux x86-64, one PyTorch CPU thread, and deterministic algorithms. See
[recorded dependencies](../requirements-validation.txt),
[plain verification output](../results/tests.log), and
[full synthetic results](../results/synthetic.json). No GPU or paid inference was
used. The model and data were initialized from scratch without downloaded weights.

## Compiler and numerical checks

The unittest suite has eight test methods, with parameterized cases inside them:

- **Strict input handling:** missing and unknown keys, invalid types and ranges,
  repeated JSON keys, malformed JSON, infeasible budgets, dimensions too small
  for pooling, and output-directory collisions.
- **Determinism:** normalized input equivalence, unchanged caller input, SHA-256
  checks, byte-identical CLI outputs, and regeneration of the committed example.
- **Independent budget checks:** depths 1/2/3; channel/class pairs (1,1) and (3,4);
  widths 1/2/3/5. For each space, budgets immediately below, at, and above every
  independently instantiated model's count are checked. This is 72 budget cases,
  including six infeasible cases. Generated model totals and every candidate
  count agree with independently instantiated references.
- **Dimensions:** depths 1/2/3 with square 32×32, rectangular 24×40, odd 33×41,
  and minimum `2**depth` inputs. Hooks compare every generated convolution's
  actual input/output shape and parameter count to the report. Further shapes
  verify spatial flexibility; invalid channel counts, ranks, empty batches, and
  either spatial axis below minimum are rejected. A separate depth-8 check uses
  3 input channels, one output logit channel, width 1, and a 256×257 input.
- **Numerical equivalence:** seeds 7, 19, 41; four depth/shape pairs
  (1,16×16), (2,12×20), (3,17×25), (3,8×8); float32 and float64. All 24 cases
  compare forward outputs, input gradients, and each convolution weight/bias
  gradient using identical weights and a seeded random output probe. Tolerances
  are `atol=1e-6, rtol=1e-5` for float32 and `atol=1e-10, rtol=1e-8` for float64.

The reference has a separately written architecture constructor and forward
method. Its nearest resize uses integer indexing; generated code uses PyTorch
interpolation. It shares PyTorch convolution, ReLU, concatenation and max pooling
with the generated model. These checks give finite numerical evidence, not a
formal equivalence proof or independent verification of PyTorch itself.

## Installation and standalone execution

`scripts/verify_isolated.py` builds an actual wheel using already provisioned
build dependencies, installs it with `--no-index --no-deps` in a fresh venv,
exercises the console entry point, and generates the example without PyTorch in
that environment. It then launches an isolated Python interpreter with a
separate dependency view that excludes the compiler and editable-import hooks.
It asserts that `unet_budget` is not importable and that the generated model
completes a forward pass, finite input/parameter gradients, and an SGD update
which changes the head weights. Compilation and execution reject socket audit
events. This check targets POSIX environments and uses symlinks for provisioned
dependencies; it does not install a second copy of PyTorch or claim a clean OS
image deployment test.

## Bounded synthetic training

A fixed 7,470-parameter generated model (depth 2, width 4) trains for 24 epochs,
6 batches per epoch, using Adam at learning rate 0.01. The data are 33×41 images
of a filled circle with Gaussian pixel noise, with 48 training and 16 evaluation
examples. Train, evaluation, model, and shuffle seeds are 101, 202, 303, and 404.
All 64 image/target pair hashes are unique; train and evaluation hash sets are
disjoint. The evaluation tensors never enter optimizer steps. The fixed epoch
count does not use evaluation-based early stopping or model selection.

The constant baseline uses training-set class frequencies as probabilities at
every pixel, and thus predicts background for every pixel under argmax. Scores
are aggregated over all pixels in the held-out set; foreground Dice and IoU
measure class 1 only. The initial model also predicts all background, illustrating
why pixel accuracy alone is misleading on this imbalanced task.

| Held-out measure | Initial model | Trained model | Constant baseline |
| --- | ---: | ---: | ---: |
| Cross entropy | 0.612662 | 0.013708 | 0.326429 |
| Pixel accuracy | 0.899483 | 0.995103 | 0.899483 |
| Foreground IoU | 0.000000 | 0.952188 | 0.000000 |
| Foreground Dice | 0.000000 | 0.975508 | 0.000000 |

Mean training cross entropy is 0.534048 in epoch 1 and 0.013024 in epoch 24. The
full per-epoch curve, configuration, dataset/per-sample hashes, dependency
versions, final weight hash, and held-out logits hash are in the results JSON.
A second complete run exactly reproduced the saved data, losses, metrics,
weights, and logits in this environment. `--check` fails on any such mismatch;
it never requires an accuracy improvement. Fresh runs can write a separate
results path, preserving an unfavorable result as readily as a favorable one.

These data directly encode the target in noisy pixel intensity. They are an easy
integration exercise, with a small held-out set and one training seed, not evidence
of useful accuracy on natural or medical images. There are no confidence
intervals, real-data validation, latency benchmarks, or activation-memory claims.
PyTorch emitted an NNPACK unsupported-hardware warning on this CPU and used its
available CPU implementation; no GPU backend was tested. Exact numerical results
may change with dependencies, CPU features, or platforms.

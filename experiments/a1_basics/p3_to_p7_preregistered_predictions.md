# Problems 3-7 and example questions: pre-registered predictions

Recorded 2026-09-27, before any Problem 3-7 run was launched. Reference numbers from
earlier runs: default d8 = 2.9269; model seeds 1/2 = 2.9236/2.9299 (sd 0.0032);
2(a) d8 baseline (identical config, rerun) = 2.9264.

## Problem 3: measuring variation
| Quantity | Prediction |
|---|---|
| Joint seeds (model+data+hardware), sd of final val loss | 0.003-0.004 (range ~0.008-0.01), roughly normal |
| Model seed only, sd | ~0.003 |
| Data seed only, sd | ~0.003 |
| Hardware nondeterminism only (same seeds, H100), sd | ~0.001-0.002: smaller, but same order of magnitude (chaos amplifies it) |
| Two deterministic references, same H100 | bit-identical losses |
| A100 vs H100 (same seeds) | differs by ~0.001-0.003; deterministic A100 != deterministic H100 |
| Batch size 16 sd vs standard | 1-1.5x |
| LR 0.009 sd vs standard | 1-2x |
| d4 sd vs d8 | similar, within ~1.5x |
| **Example question (medium)**: which change more than doubles the sd? | **4. Neither change** |

Reasoning: the final loss averages over ~600M tokens and ends with LR decayed to 0,
which damps trajectory differences; each source alone should give most of the joint
spread because chaotic training makes any one source randomize the trajectory.

## Problem 4: amplification (deterministic)
- One-token change at step 0: final val loss differs from the reference by 0.001-0.003,
  i.e. the same order as seed noise (the perturbation is amplified to saturation).
- Later perturbations leave less time to amplify: final difference shrinks with T;
  at 95% it is < 0.0005.
- Bigger perturbations saturate sooner, but an early perturbation of any size ends at
  roughly the same (seed-noise) level.

## Problem 5
- beta1 0.5: slightly worse (+0.005-0.03) and a noisier curve; beta1 0.98: within +0.02.
- **Example question (easy)**: (a) **D, learning rate 0.009** (early spikes, fast initial
  drop); (b) **a, gap at most +0.05** (LR 6e-3 gave +0.015 and 1.2e-2 gave +0.034).

## Problem 1 example questions
- **Easy**: **A < C < B**. A = 2.927; C (LR 9e-3, bs 128) ~2.99, interpolating bs 128 at LR
  6e-3 (2.979) and 1.2e-2 (3.001); B (LR 1e-3, bs 128) ~3.00+, since low LR hurts more.
- **Medium** (LR 9e-3, wd 1.0): **C < B < A**. No warmup at high LR is the worst (A);
  linear decay beats WSD at high LR (1(c)), and 10% warmup is plenty.

## Problem 2 example (hard, not run)
**-0.15 < gap <= 0.03; do not trust the +0.055 extrapolation.** The pilots' gap comes from
training at shorter context than the 1024-token validation (d4-d6 scaled context < 1024),
which disappears at d8 (identical recipes) and flips at d10 (longer training context).
The d6 scaled-context curve is also overfitting (15 epochs), so the fitted form is wrong.

## Problem 6 example
**Alternative.** Residual-stream RMS in a healthy d8 run is O(1-10) (logged
model.layers.*/activation); 1e3-1e5 means the run is broken.

## Problem 7: own question (4x repetition, 614M tokens seen, 154M unique)
**d9 < d6 < d4** (bigger still better): d4 ~3.31, d6 ~3.09, d9 ~2.96. Up to ~4 epochs,
repetition is nearly as good as fresh data; the U-shape from 16x repetition should not
appear yet.

## Problem 3(c) addendum (recorded 2026-09-28, before launch)
Three more settings, joint seeds 1-3 each, sd compared with standard d8 (0.0030):
- Constant LR (no decay): **~2-3x** the standard sd - annealing is what pulls seeds together.
- LR 0.03 (near the unstable edge): **~1.5-2.5x** - more chaos early.
- Dropout 0.2 (extra per-step randomness): **~1-1.5x**.

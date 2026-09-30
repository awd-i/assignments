# Problem 2(a): pre-registered predictions

Recorded 2026-09-26, after stage 1 (d4-d7) and before stage 2 (d8, d9) was launched.
Final validation loss; ranges are rough 80% intervals.

| Variant | Fit used | d8 (34.6M) | d9 (48.0M) | d20 (486M) |
|---|---|---|---|---|
| Baseline (linear, LR 3e-3) | power law + floor | 2.93 (2.92-2.94)* | 2.88 (2.87-2.90) | 2.70 (2.62-2.76) |
| Constant LR | between fits | 3.17 (3.15-3.19) | 3.14 (3.11-3.17) | 2.95 (2.85-3.10) |
| Dropout 0.2 | power law, capped vs baseline | 3.05 (3.03-3.07) | 2.99 (2.97-3.01) | 2.79 (2.72-2.87) |
| LR 0.03 | power law + floor | 3.04 (3.00-3.10) | 3.03 (2.98-3.12) | >= 3.0, or unstable |

\* Not a blind prediction: the d8 baseline recipe was already trained earlier (2.927).

## Reasoning

- **Data is fixed at 614M tokens for every size.** At d20 that is ~1.3 tokens per
  parameter, far below the ~20 of compute-optimal training, so larger models are
  data-limited and loss should level off. A pure power law (baseline d20 = 2.41)
  is too optimistic; the floor fit (2.69) is the better guide for d20.
- **Baseline:** the d4-d7 points bend gently, and the floor fit reproduced the
  known d8 result (2.928 vs 2.927) without using it.
- **Constant LR:** no decay means no end-of-training annealing, so loss stays
  high. The d6->d7 gain (0.025) was much smaller than earlier steps (~0.09), so it
  is flattening, but one small step is near seed noise (0.006), so predictions sit
  between the two fits rather than trusting the steep floor fit (E = 3.07).
- **Dropout 0.2:** points are still a straight line, so the power law fits d8/d9.
  Training is single-epoch, so dropout only adds noise and never helps against
  overfitting. The gap to baseline is shrinking slowly (0.16 at d4 -> 0.15 at d7)
  but should not close, so d20 is capped at about baseline + 0.09 rather than the
  power law's 2.53 (which would beat the baseline).
- **LR 0.03 (10x default):** its gap to baseline grows with size (0.03 at d4 ->
  0.10 at d7), because wider models usually need a smaller LR. Expect it to
  plateau near 3.0 and possibly go unstable at larger sizes.

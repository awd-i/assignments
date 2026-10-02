# A2 Problem 1(c)/(d): pre-registered predictions

Recorded 2026-10-01 10:35 PDT, before any P1(c) or P1(d) target run was launched.

## P1(c): optimal AdamW LR at 4.9152B tokens
Source optima (parabola in log2 LR): 153.6M 2.38e-3, 307.2M 2.98e-3, 614.4M 3.19e-3, 1.2288B 3.88e-3, 1.8432B 3.43e-3, 2.4576B 2.91e-3.

| Fit | Power law | Predicted LR* at 4.9152B |
|---|---|---|
| All six budgets | LR* ~ D^+0.099 | **3.73e-3** |
| Larger three (1.2288B-2.4576B) | LR* ~ D^-0.409 | **2.23e-3** |

Target runs: LR {1.5e-3, 3e-3, 6e-3} + the two predictions (5 runs).

**Expected lower loss: the larger-three prediction (2.23e-3), by a small margin (<= 0.003, possibly within seed noise).**
Reasoning: the optimum peaked near 1.2B and then declined, so the recent regime suggests the 4.9B optimum is
below 3e-3. The all-six fit is low-variance but biased: it averages the rising small-budget regime with the
falling large-budget one. The larger-three fit is less biased but high-variance (three noisy optima from flat
curves, slope -0.41) and extrapolates 2x past its last point. The two predictions sit symmetrically (in log LR)
around 2.9e-3, and the loss-vs-LR curve is very flat at these budgets, so both should land within ~0.005 of
the best grid LR.

## P1(d): optimal Hyperball LR at 1.2288B tokens
Source optima (parabola in log2 LR through the best grid LR and its neighbours): 153.6M 1.37e-2, 307.2M 1.29e-2,
614.4M 1.21e-2 -> **LR* ~ D^-0.089, predicted LR*(1.2288B) = 1.14e-2** (5-point local fit gives D^-0.17 and 9.8e-3).
Compare AdamW on the same three budgets: D^+0.213 (rising). Hyperball's optimum falls smoothly and much more
regularly (the three optima are monotone and nearly log-linear).

Target runs: LR {5.7e-3, 8.0e-3, 1.14e-2 (prediction), 1.6e-2} (x sqrt2 spacing, extra point on the low side
because the trend is downward and the 5-point fit says 9.8e-3).
Expected: best sampled LR is 8.0e-3 or 1.14e-2; the prediction's loss is within 0.005 of the best sampled loss.

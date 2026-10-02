# A2 Problem 3.2(b): pre-registered batch-transfer predictions

Recorded 2026-10-02 02:58 PDT (hypothesis i) before any B = 128/256 run.

Hypothesis i (WD .1 fixed, scale peak LR). Fitted optima at 614.4M tokens (parabola in log2 LR):
B=8 1.06e-3, B=16 1.73e-3, B=32 3.26e-3, B=64 3.19e-3 (supplied P1a). Power law LR* ~ B^0.57 ->
**B=128: 5.59e-3, B=256: 8.28e-3.**
Expectation: the trend has already flattened between B=32 and 64 (the NQM's curvature/step-count limit), so I expect
the true optimum at 128/256 to be closer to 3-4e-3 than to the power law, i.e. the power-law prediction overshoots
and costs ~0.01-0.02. Target grid per batch: {.003 (no scaling), prediction, 2 x prediction}.

Hypothesis ii (LR .0015 fixed, scale WD): recorded below once its source sweep finishes.

## Hypothesis ii (recorded 2026-10-02 12:40 PDT, before any hypothesis-ii target run)

LR .0015 fixed, fitted optimal WD at 614.4M tokens: B=8 0.065, B=16 0.12, B=32 0.24, B=64 0.50 ->
WD* ~ B^0.98 (almost exactly linear: the AdamW averaging timescale B/(LR*WD) in tokens stays fixed) ->
**B=128: WD 0.95, B=256: WD 1.86.** Target grid: {pred/2, pred, 2 x pred}.
Expectation: hypothesis ii transfers better than hypothesis i. Its source trend is clean and shows no saturation
(B^0.92 on B<=32 vs B^0.98 with B=64), while hypothesis i's LR optimum already flattened between B=32 and 64.
Best losses should still rise slightly beyond B~32-64 (fewer updates at fixed tokens), regardless of hypothesis.

# A2 Problem 4.2(c): pre-registered width-1024 predictions

Recorded 2026-10-01 23:45 PDT, after P4.2(a) and before any width-1024 run.
Fitted optimal base LRs at 153.6M tokens (parabola in log2 LR, widths 128/256 from P4.2(a), 512 from supplied P1a):

| Prescription | w128 | w256 | w512 | power law | predicted LR* at w1024 |
|---|---|---|---|---|---|
| course baseline (SP) | 8.25e-3 | 4.21e-3 | 2.38e-3 | width^-0.90 | **1.26e-3** |
| muP | 2.98e-3 | 2.03e-3 | 2.38e-3 | width^-0.16 | **1.94e-3** |

Directly transferred source LR: .003 (P1's best sampled LR at 153.6M).
Expectation: for SP the fitted law beats direct transfer (the optimum falls ~1/width); for muP both are close
(the fit is nearly flat and its slope is mostly noise), with loss gaps < 0.01.
Local sweep at 1024: {.00075, .0015, .003, .006} plus each predicted LR.

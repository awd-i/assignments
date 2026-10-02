# A2 Problem 1(b): pre-registered optimal-LR predictions

Recorded 2026-09-30 16:14 PDT, after fitting P1(a) and BEFORE loading any P1(b) data
(experiments/a2/provided_sweeps.load('P1b') has not been called).

P1(a) fitted optima (parabola in log2 LR through 3 points; 80% bootstrap intervals with seed sd 0.003):
153.6M -> 2.38e-3 [2.30, 2.45]; 307.2M -> 2.98e-3 [2.79, 3.18]; 614.4M -> 3.19e-3 [3.03, 3.39].

| Budget D | Primary: power law LR* = 3.28e-3 (D/614.4M)^0.213 | Alternative: saturating LR* = 3.30e-3 - 0.92e-3 (D/153.6M)^-1.52 |
|---|---|---|
| 1.2288B | **3.80e-3** | 3.26e-3 |
| 1.8432B | **4.14e-3** | 3.28e-3 |
| 2.4576B | **4.40e-3** | 3.29e-3 |

Reasoning: the optimum rises with budget (opposite to convex-optimization intuition, which says the LR should
fall as ~1/sqrt(T)), so longer runs tolerate or need a larger peak LR. But the rise is decelerating
(+25% for the first doubling, +7% for the second), so the power law may overshoot; the saturating
form is the main alternative. The 307.2M curve is flat (curvature 0.015), so its optimum is the least certain.

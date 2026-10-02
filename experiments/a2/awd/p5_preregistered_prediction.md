# A2 Problem 5 (optional): pre-registered prediction

Recorded 2026-10-01 21:50 PDT, before launching the target runs.

**Question (single answer, Medium).** Can weight decay rescue a learning rate that is far too high?
Default d8 recipe, 153.6M tokens, linear decay, seed 42. The supplied joint sweep puts the optimum near
peak LR 1.9e-3, WD .85 (fitted loss 3.18; LR x WD = 1.57e-3). With the default WD .1, peak LR .006 gives 3.3056.
Keep peak LR at .006 but raise WD to .262 so that LR x WD matches the optimum's product (1.57e-3).
What final validation loss do you expect?
(a) above 3.30 (no rescue)  (b) 3.25-3.30 (partial)  (c) 3.20-3.25 (most of the gap)  (d) below 3.20 (full rescue)

**My prediction: (b), about 3.28.** Reasoning: the 153.6M quadratic fit gives 3.284 at (.006, .262). The
LR-WD valley is steeper than the constant-product line (log-slope -2.3), so matching the product only
partially compensates: LR also sets the per-step noise, which WD cannot remove.
Second probe (same question at LR .012, WD .131): quadratic predicts 3.448 (vs 3.462 at WD .1) - essentially no rescue.

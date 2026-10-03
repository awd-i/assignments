"""Build assignment2_report.pdf in the same style as the A1 sample write-up."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[3]
PLOTS = Path(__file__).resolve().parent / "plots"
OUT = ROOT / "assignment2_report.pdf"
LABEL = "CS 312 Assignment 2"

title = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=26, leading=32, spaceAfter=18)
h1 = ParagraphStyle("h1", fontName="Helvetica", fontSize=17, leading=22, spaceBefore=8, spaceAfter=8)
body = ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=13.5, spaceAfter=6)
bullet_style = ParagraphStyle("bullet", parent=body, spaceAfter=2)
cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=9.5, leading=12.5)
cell_bold = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
caption = ParagraphStyle("cap", parent=body, fontSize=9, leading=12, textColor=colors.HexColor("#555555"), alignment=1)


def bullets(*items):
    return ListFlowable([ListItem(Paragraph(t, bullet_style), leftIndent=14) for t in items],
                        bulletType="bullet", start="\u2022", bulletFontSize=9, leftIndent=14, spaceAfter=8)


def table(rows, widths):
    data = [[Paragraph(c, cell_bold if i == 0 else cell) for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


def figure(name, width=4.6 * inch, text=None):
    img = Image(str(PLOTS / name))
    img.drawWidth, img.drawHeight = width, width * img.imageHeight / img.imageWidth
    parts = [Spacer(1, 6), img]
    if text:
        parts.append(Paragraph(text, caption))
    parts.append(Spacer(1, 8))
    return KeepTogether(parts)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(0.9 * inch, 0.55 * inch, LABEL)
    canvas.drawRightString(letter[0] - 0.9 * inch, 0.55 * inch, str(doc.page))
    canvas.restoreState()



def pending(name):
    return [Paragraph(f"<i>{name}: results pending.</i>", body)]


def P1C_SECTION():
    return [
        sec("Problem 1c: Compare and test two scaling rules"),
        para("<b>Fits</b> (log-log least squares on the fitted optima): all six budgets give LR* ~ D^+0.099, predicting "
             "<b>3.73e-3</b> at 4.9152B; the larger three (1.2288-2.4576B) give LR* ~ D^-0.409, predicting <b>2.23e-3</b>."),
        para("<b>Pre-registered expectation</b> (p1cd_preregistered_predictions.md): the larger-three prediction gives "
             "slightly lower loss (by 0.003 or less), because the all-six fit averages a rising and a falling regime "
             "(bias) while the larger-three fit uses only the relevant regime (variance from three noisy optima); both "
             "within ~0.005 of the best grid LR because the curves are flat at this scale."),
        table([["LR at 4.9152B", "1.5e-3", "2.23e-3 (larger three)", "3e-3", "3.73e-3 (all six)", "6e-3"],
               ["Val loss", "2.7165", "2.7166", "2.7173", "2.7189", "2.7282"],
               ["Gap to best grid LR", "0", "<b>+0.0001</b>", "+0.0009", "<b>+0.0025</b>", "+0.0117"]],
              [1.3 * inch, 0.7 * inch, 1.3 * inch, 0.7 * inch, 1.2 * inch, 0.7 * inch]),
        figure("p1c_two_fits.png", width=4.6 * inch,
               text="Both fits extrapolated to 4.9152B with the measured target optimum (star). Inset: target loss minus "
                    "best grid loss (teal: larger-three prediction, orange: all-six prediction)."),
        bullets("<b>Fitted target optimum: 2.00e-3</b> (parabola through the grid; 80% interval 0.8-2.6e-3). The larger-three "
                "prediction was 11% high, the all-six prediction 87% high.",
                "<b>The smaller-model (larger-budget) fit gave a gain</b>: 0.0001 vs 0.0025 above the best sampled loss, "
                "as pre-registered. The gain is about one seed standard deviation, so the loss evidence alone is weak; "
                "the shape of the target curve (monotone rise from 1.5e-3 to 6e-3) makes the ordering clear.",
                "<b>Bias-variance:</b> more budgets lower the variance of the slope but bias it when the optimum's trend "
                "changes regime (here the trend reverses near 1.2B). Fitting only the nearby regime is noisier but "
                "aimed in the right direction, which mattered more.",
                "<b>Extrapolation range:</b> a 2x extrapolation from the larger three worked to 11% in LR. It worked partly "
                "because the landscape is flat: a 2x LR error costs only ~0.01 at this scale. The P1(a) rule "
                "extrapolated 8x (5.1e-3) would have cost ~0.007."),
    ]


def P1D_SECTION():
    return [
        sec("Problem 1d: Hyperball versus AdamW"),
        para("<b>Implementation</b> (hyperball.py, optimizer_builder): Adam direction U with bias correction; for every "
             "linear weight W, W~ = W - lr (||W||_F / ||U||_F) U, then W = (||W||_F / ||W~||_F) W~ (norm kept at its initial "
             "value, verified for all 57 matrices). Embeddings and norm gains use ordinary Adam at 0.1041 x LR; no WD. "
             "<b>Validation:</b> reproduced the supplied 153.6M, LR .01 run: 3.2081 vs 3.2113."),
        para("<b>Source fits</b> (24 supplied runs, parabola in log2 LR through the best grid LR and its neighbours):"),
        table([["Tokens", "Hyperball LR*", "Hyperball best loss", "AdamW LR*", "AdamW best loss"],
               ["153.6M", "1.37e-2", "3.204", "2.38e-3", "3.229"],
               ["307.2M", "1.29e-2", "3.032", "2.98e-3", "3.056"],
               ["614.4M", "1.21e-2", "2.918", "3.19e-3", "2.926"],
               ["Power law", "D^-0.09", "", "D^+0.21", ""]],
              [1.0 * inch, 1.3 * inch, 1.4 * inch, 1.2 * inch, 1.3 * inch]),
        para("<b>Pre-registered prediction</b> at 1.2288B (p1cd_preregistered_predictions.md): <b>1.14e-2</b> "
             "(a wider +-2-point fit gave 9.8e-3). Tested 5.7e-3, 8.0e-3, 1.14e-2, 1.6e-2."),
        table([["LR", "5.7e-3", "8.0e-3", "1.14e-2 (predicted)", "1.6e-2"],
               ["Val loss at 1.2288B", "2.8466", "2.8322", "<b>2.8279</b>", "2.8385"]],
              [1.5 * inch, 0.9 * inch, 0.9 * inch, 1.5 * inch, 0.9 * inch]),
        figure("p1d_hyperball_vs_adamw.png", width=5.9 * inch,
               text="Left: loss vs LR, Hyperball (solid) vs AdamW (dashed). Right: optimum normalised to its 153.6M value."),
        bullets("<b>The prediction was the best sampled LR</b> (loss gap 0). A parabola through the four target points "
                "puts the optimum at 1.03e-2, 9% below the prediction and between my two source fits. Hyperball's best "
                "loss beats AdamW's best at 1.2288B (2.8378) by 0.010.",
                "<b>The Hyperball optimum varies more regularly:</b> it falls monotonically (x0.94, x0.94, then x0.85 per "
                "doubling; 4-budget fit D^-0.13), while AdamW's optimum rose and then reversed over the same range. "
                "Its loss-LR curves also stay sharper (curvature 0.026 per log2 unit at 1.2B vs ~0.008 for AdamW), "
                "so its optimum is better determined.",
                "Hyperball's LR is a relative step size per matrix (step = LR x ||W||), so its numbers are ~4x AdamW's "
                "and not directly comparable. Fixed norms remove the LR x WD weight-norm equilibrium, which is one of the "
                "budget-dependent effects that move AdamW's optimum.",
                "<b>Uncertainty with three source budgets:</b> the slope rests largely on the 614.4M optimum, which "
                "moves from 1.21e-2 to 1.08e-2 depending on the fitting window, and any curvature in log-log "
                "(like the steeper last step) is invisible. Three points give one slope and no check on it."),
    ]


def P1_SYNTHESIS():
    return para("For AdamW with a linear schedule the optimal peak LR is nearly budget-independent: it rises ~35% from "
                "153.6M to ~1.2B tokens and then falls (2.9e-3 at 2.5B, 2.0e-3 at 4.9B), while the loss-LR curve "
                "flattens so much that a 2x LR error costs only ~0.01 at large budgets. Short-range power laws therefore "
                "extrapolate poorly when the trend changes regime; fitting only the nearby regime (higher variance, lower "
                "bias) predicted 4.9B well. Hyperball, which fixes weight norms, has an optimum that moves monotonically "
                "and predictably with budget (its 1.2288B prediction was the best sampled LR), consistent with weight-norm "
                "dynamics (LR x WD) being one source of AdamW's irregular optimum. The peak LR transfers across "
                "linear and cosine schedules to within ~15%.")


def P2C_SECTION():
    return [
        sec("Problem 2c: Test the joint rule"),
        para("<b>Fit.</b> Product law on {.1536, .3072, .6144, 1.2288}B: LR* x WD* ~ D^-0.59, predicting 2.78e-4 at 2.4576B, "
             "so at peak LR .003 the rule gives <b>WD = .0927</b>. All runs linear decay, 2.4576B tokens (held out)."),
        table([["Recipe at 2.4576B", "Peak LR", "WD", "Val loss", "Rule's gap"],
               ["Product rule", ".003", ".0927", "<b>2.7693</b>", "-"],
               ["(i) best 153.6M pair, no retuning", ".0015", "1.6", "2.8526", "-0.0834 (rule better)"],
               ["(ii) WD sweep at LR .003", ".003", ".05 / .1 / .2", "2.7723 / <b>2.7683</b> / 2.7729",
                "+0.0010 vs best (WD .1, supplied run)"]],
              [2.2 * inch, 0.6 * inch, 0.9 * inch, 1.6 * inch, 1.5 * inch]),
        bullets("<b>The product rule gives an effective recipe</b>: without any tuning at the target it lands within 0.001 "
                "of the best of a three-point WD sweep (well inside seed noise, sd ~0.003) and is 0.083 better than reusing "
                "the small-budget optimum.",
                "Reusing (.0015, 1.6) fails because its LR x WD (2.4e-3) is ~9x too large at 2.46B: the AdamW averaging "
                "window is ~400 of 37,500 updates, so the weights effectively forget most of training.",
                "<b>Caveat:</b> the rule's WD (.093) happens to be next to the default .1, so this target cannot distinguish "
                "it from 'use the default'; the rule's value is that it predicts the right WD at budgets where the default "
                "is badly wrong (WD* ~ .85 at 153.6M, where tuning WD gained 0.045)."),
    ]


def P2_SYNTHESIS():
    return para("In AdamW, LR and WD act largely through their product, which sets the averaging timescale 1/(LR x WD) "
                "of the weights. Contours of loss are tilted ellipses along roughly constant-product lines, the optimal WD "
                "(and the product) follow clean power laws in the token budget (WD* ~ D^-0.82, R^2 0.994; product "
                "~ D^-0.59, R^2 0.967) while the optimal LR alone does not, and the benefit of tuning WD shrinks from "
                "0.045 at 153.6M to 0.002 at 1.2B. Scaling the product with budget gives a working recipe at a held-out "
                "2x larger budget (within 0.001 of a WD sweep), whereas reusing a small-budget optimum costs 0.083. "
                "The same coupling reappears in Problem 3.2 (WD ~ B at fixed LR) and Problem 4.2 (muP's eta/m also "
                "rescales hidden-matrix decay).")


def P32_SECTION():
    return [
        sec("Problem 3.2: What should scale when batch size changes?"),
        para("<b>Setup.</b> Default d8 recipe at 614.4M tokens, microbatches of min(B, 64) sequences (B = 128 / 256 use "
             "2 / 4 accumulation steps and skip 64 / 192 sequences: 0.011% / 0.032% of tokens). B = 64 points reuse the "
             "supplied P1a (WD .1) and P2a (LR .0015) runs. Diagnostics off. 31 new runs (p32_analysis.py)."),
        figure("p32_ab.png", width=6.2 * inch,
               text="Loss vs LR at WD .1 and vs WD at LR .0015 per batch, fitted optima vs batch with source fits "
                    "(B <= 64), and best measured loss vs batch."),
        sub("(a) LR sweep at WD .1."),
        table([["Batch", "8", "16", "32", "64"],
               ["Fitted LR*", "1.06e-3", "1.73e-3", "3.26e-3", "3.19e-3"],
               ["Best measured loss", "2.9367", "2.9228", "2.9193", "2.9256"]],
              [1.6 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch]),
        bullets("The optimal LR grows like sqrt(B) from 8 to 32 (overall fit B^0.57) and then saturates between 32 and 64.",
                "At fixed tokens, performance is U-shaped in batch: best at B = 32-64 (best measured overall 2.9178), "
                "worse at small batch (noise) and at large batch (fewer updates)."),
        sub("(b) Two scaling hypotheses (predictions pre-registered in p32b_preregistered_predictions.md)."),
        bullets("<b>(i) WD .1 fixed, scale LR:</b> LR* ~ B^0.57 predicts 5.59e-3 at B = 128 and 8.28e-3 at B = 256.",
                "<b>(ii) LR .0015 fixed, scale WD:</b> fitted WD* = 0.065, 0.12, 0.24, 0.50 at B = 8-64, i.e. WD* ~ B^0.98, "
                "predicting 0.95 at B = 128 and 1.86 at B = 256."),
        table([["Target", "Hyp (i): LR .003 / pred / 2 x pred", "Hyp (ii): WD pred/2 / pred / 2 x pred", "(ii) - (i), best vs best"],
               ["B = 128", "2.9500 / 2.9608 / 3.0015", "2.9446 / <b>2.9332</b> / 2.9430", "-0.017"],
               ["B = 256", "3.0154 / 3.1213 / 3.1405", "2.9854 / <b>2.9680</b> / 2.9886", "-0.047"]],
              [0.8 * inch, 2.2 * inch, 2.3 * inch, 1.4 * inch]),
        bullets("<b>Hypothesis (ii) transfers better</b>: its predicted WD is the best sampled point at both target batches, "
                "and beats hypothesis (i)'s best by 0.017 and 0.047. Hypothesis (i)'s power law overshoots (the predicted "
                "LR costs +0.011 and +0.106 vs simply keeping .003), as pre-registered: the LR optimum had already "
                "flattened between B = 32 and 64.",
                "<b>Why:</b> with decoupled WD the weights average over 1/(LR x WD) updates = B/(LR x WD) tokens. Holding "
                "that window fixed in tokens gives WD ~ B at fixed LR, with no stability limit, whereas raising the LR "
                "eventually hits the curvature / step-count limit (the NQM's breakdown). This is the same coupling as "
                "Problem 2's product law.",
                "<b>Agreement with the NQM:</b> the sqrt(B) LR rule for Adam and its saturation past a critical batch agree "
                "with 3.1(b, c); the WD rule is outside the NQM, which has no weight decay."),
        sub("(c) Momentum ablation (best LR-WD pair per batch held fixed, beta2 .95)."),
        table([["beta1", "0", "0.5", "0.8", "0.9", "0.95", "0.98"],
               ["B = 8 (LR .0015, WD .05)", "2.9772", "2.9605", "2.9464", "2.9358", "<b>2.9314</b>", "2.9339"],
               ["B = 256 (LR .0015, WD 1.86)", "3.2165", "3.0696", "2.9878", "2.9680", "<b>2.9672</b>", "3.0200"]],
              [1.9 * inch, 0.65 * inch, 0.65 * inch, 0.65 * inch, 0.65 * inch, 0.7 * inch, 0.65 * inch]),
        figure("p32c_momentum.png", width=4.4 * inch, text="Loss vs beta1 and cost relative to beta1 = .9."),
        bullets("<b>Removing momentum costs 6x more at large batch</b> (+0.249 at B = 256 vs +0.041 at B = 8), as the NQM "
                "predicts: with little gradient noise, training is curvature-limited and momentum accelerates it.",
                "<b>Too much momentum hurts only at large batch</b> (beta1 .98: +0.052 at B = 256 with 2,344 updates, "
                "+0.002 relative to .95 at B = 8 with 75,000 updates): an averaging window of ~50 updates lags when there "
                "are few updates, as in the NQM's 32-update case.",
                "<b>Unlike the NQM, momentum still helps at small batch.</b> We held LR fixed (the NQM comparison retuned "
                "it), but Adam's update is normalised, so beta1 is not just an LR rescaling: averaging across steps "
                "reduces the noise of the update direction, which dominates at B = 8."),
        sub("(d) What does the NQM explain?"),
        table([["Question", "NQM prediction", "LM measurement", "Verdict"],
               ["LR vs batch (Adam)", "LR* ~ sqrt(B) while noise dominates; saturates past a critical batch",
                "sqrt(B) from 8 to 32; .003 best at 64, 128 and 256", "Agrees"],
               ["Loss vs batch, fixed tokens", "Flat, then worse past the critical batch",
                "Best at 32-64 (2.918); B = 256 +0.050", "Agrees"],
               ["WD vs batch", "Not modelled", "WD* ~ B^0.98; transfers to 128/256", "Effect not modelled"],
               ["Momentum, large batch", "Large benefit; too much hurts with few updates", "+0.249 without; beta1 .98 +0.052",
                "Agrees"],
               ["Momentum, small batch", "Little benefit after LR retuning", "+0.041 without (LR held fixed)",
                "Prediction fails (partly protocol)"]],
              [1.3 * inch, 1.9 * inch, 1.9 * inch, 1.2 * inch]),
        para("<b>Failed prediction vs unmodelled effect.</b> The small-batch momentum result contradicts a definite NQM "
             "prediction; the WD scaling is simply outside the model. <b>Change to the NQM:</b> add decoupled weight decay "
             "and a drifting optimum (the target moves as new examples arrive). The model then needs an averaging window "
             "of fixed length in examples, which yields WD ~ B at fixed LR and explains why scaling WD transfers when "
             "scaling LR does not."),
        sub("Synthesis (Problem 3)."),
        para("Batch size trades gradient noise against the number of updates. The NQM correctly predicts the square-root "
             "LR rule for Adam, its breakdown past a critical batch (~32-64 here at 614.4M tokens), and that momentum "
             "matters most at large batch. What it misses is weight decay: the most reliable batch rule in the language "
             "model is to keep the AdamW averaging window fixed in tokens, i.e. scale WD linearly with batch at fixed "
             "LR, which transferred to 4x larger batches with its prediction landing on the best sampled point."),
        PageBreak(),
    ]


def P41_SECTION():
    return [
        sec("Problem 4.1: Five-update width and depth stress tests"),
        para("<b>Implementation.</b> p31_student.py dispatches to p41_policies.py (A2_POLICY env var). Kaiming: embedding "
             "variance 1, hidden 1/fan-in, readout 1/n, readout multiplier 1, every LR eta. muP (n0 = 512): readout "
             "variance 1/n0, readout multiplier 1/m, hidden LR eta/m. Depth runs (width = n0 = 64, r = L/2): Depth-muP "
             "residual and block LR r^-1/2, block eps 1e-8 r^-1/2; CompleteP residual 1/r, block eps 1e-8/r. All 107 "
             "five-update runs ran sequentially in one GPU container (p41_modal.py); the optimum is a parabola in log2 LR "
             "through the best point and its neighbours."),
        sub("(a) Width transfer."),
        para("<b>Prediction:</b> Kaiming's optimum falls roughly like 1/width (Adam updates are Theta(1) per coordinate, so "
             "the feature change grows like n x LR); muP's stays fixed and muP should also reach lower loss."),
        table([["Width", "Kaiming LR*", "Kaiming min loss", "muP LR*", "muP min loss", "Loss at w640-best LR (K / muP)"],
               ["640", "1.13e-3", "6.94", "1.61e-3", "6.83", "6.95 / 6.87"],
               ["2560", "1.63e-4", "7.13", "1.49e-3", "6.60", "8.28 / 6.68"],
               ["5120", "5.3e-5", "7.32", "1.46e-3", "6.58", "10.10 / 6.67"]],
              [0.6 * inch, 0.95 * inch, 1.15 * inch, 0.8 * inch, 1.0 * inch, 2.1 * inch]),
        figure("p41_width_curves.png", width=5.4 * inch, text="Loss after 5 updates vs base LR (stars: fitted optima)."),
        figure("p41_width_transfer.png", width=4.2 * inch, text="Fitted optimal base LR and minimum loss vs width."),
        bullets("<b>muP transfers its base LR</b> (10% change over 8x width) while Kaiming's optimum falls 21x (steeper than "
                "1/m). Reusing the width-640 LR at width 5120 costs Kaiming 3.2 nats and muP nothing.",
                "<b>muP also achieves lower loss and improves with width</b> (6.83 to 6.58); Kaiming gets worse with width (6.94 "
                "to 7.32) because stability forces a tiny LR at which hidden features barely move."),
        sub("(b) Width probes (best sampled LR of each configuration)."),
        figure("p41_width_probes.png", width=6.1 * inch,
               text="Logit RMS, final-norm feature movement M_t, mean update alignment per matrix type, and omega_move."),
        bullets("<b>Feature movement</b> after 5 updates: muP 1.09 / 1.04 / 1.01 across widths (feature learning preserved); "
                "Kaiming 1.08 / 0.64 / 0.53 (vanishing with width).",
                "<b>Initial logits</b> under muP shrink as 1/sqrt(m) (0.90, 0.45, 0.32), the predicted Theta(n^(1/2-a-b)), and "
                "grow to ~1.4-1.5 at every width by update 5.",
                "<b>Update alignment alpha lies between none and full</b>: q/k/v/gate/up/down 0.64-0.77, attention output "
                "0.88-0.91, readout 0.76-0.86, rising slowly with width (q: 0.66 to 0.73). The alpha = 1 assumption is "
                "approximately but not exactly right at these widths; the LR transfer it implies still worked.",
                "<b>omega_move = 0.50-0.53 (no alignment)</b> at every width: in five updates the feature change is "
                "unaligned with the initial readout, contradicting the derivation's omega = 1. By the 4.0 table, alpha = 1 "
                "with omega = 1/2 would allow a larger readout initialisation (variance m/n0); the 1/n0 choice is "
                "conservative but stable."),
        sub("(c, d) Depth transfer and probes (width 64)."),
        para("<b>Prediction:</b> CompleteP transfers best; standard muP's residual stream grows like sqrt(L)."),
        table([["Depth", "muP LR* / loss", "Depth-muP LR* / loss", "CompleteP LR* / loss"],
               ["2", "1.78e-2 / 6.84", "(identical)", "(identical)"],
               ["100", "1.90e-2 / 7.06", "1.72e-2 / 6.91", "1.58e-2 / 6.82"],
               ["1000", "1.94e-2 / 7.13", "2.02e-2 / 6.88", "1.52e-2 / 6.82"]],
              [0.7 * inch, 1.6 * inch, 1.7 * inch, 1.7 * inch]),
        figure("p41_depth_curves.png", width=5.8 * inch, text="Loss vs base LR at depths 2, 100, 1000 per prescription."),
        figure("p41_depth_probes.png", width=6.1 * inch,
               text="Residual RMS before the final norm, last-block FFN branch RMS (before c_L), midpoint normalized-feature "
                    "movement, and omega_move at the best sampled LR."),
        bullets("<b>All three prescriptions transfer the LR within ~15% over 500x depth</b>, but <b>better or equal transfer "
                "did not imply equal loss</b>: from depth 2 to 1000 standard muP loses 0.29 nats, Depth-muP 0.04, CompleteP 0.01. "
                "In five updates the optimal LR is set by embedding/readout dynamics and first-step stability, not by how "
                "much the deep blocks contribute.",
                "<b>Similar normalized activations hide a growing residual stream.</b> Under standard muP the residual RMS "
                "starts at 10 (L = 100) and 34 (L = 1000), about sqrt(L), and reaches 585 and 4,786 after 5 updates; "
                "Depth-muP and CompleteP stay at 1-7. Pre-norm makes every block's normalized input look alike (movement "
                "~1 in all cases), but under standard muP each block's branch is a vanishing fraction of the residual "
                "stream, so depth adds little.",
                "<b>Bounded vs feature-learning:</b> standard muP's signals stay bounded only through normalization; "
                "CompleteP's 1/L multiplier keeps the residual stream Theta(1) and each block's change order one, the depth "
                "analogue of maximal update. omega_move is ~0.50 at every depth."),
        sub("Synthesis (Problem 4.1)."),
        para("Width muP did what the derivation promised: one base LR works from width 640 to 5120 and wider is better, "
             "while Kaiming needs a 21x smaller LR and still loses feature learning. The measured alignments explain why "
             "the rule is approximately right: hidden updates are strongly (alpha 0.65-0.9, rising with width) though not "
             "fully aligned with their inputs. The readout assumption (omega = 1) is not supported in five updates, so the "
             "1/n0 readout initialisation is conservative rather than tight. For depth, LR transfer is a weak test: all "
             "prescriptions transfer, but only the depth-corrected ones keep the residual stream bounded and the loss flat."),
    ]


def P42_SECTION():
    return [
        sec("Problem 4.2: From five updates to a longer training budget"),
        para("<b>Implementation</b> (mup.py: model builder + optimizer builder). muP changes exactly four settings vs the "
             "course baseline, with n0 = 512, m = n/n0 and the same truncated-Gaussian G: embedding init G/n0 (baseline "
             "G/n), readout init G/sqrt(n0) (baseline G/sqrt(n)), readout multiplier 1/m (forward behaviour fixed in the "
             "constructor), hidden-matrix LR eta/m (separate AdamW groups). Depth rules (reference depth 8, r = L/8) scale "
             "each block's two branches, block-matrix and block-norm LRs and Adam eps as in 4.1. At width 512, depth 8 all "
             "prescriptions equal the baseline, so the supplied P1a 153.6M runs serve as that point. 48 new runs, "
             "153.6M tokens, diagnostics on."),
        sub("(a, b) Transfer to widths 128 and 256, and tuned performance."),
        para("<b>Prediction:</b> P1's best sampled LR (.003) transfers under muP but not under the baseline, whose optimum "
             "should rise as width shrinks."),
        table([["Width", "Baseline LR*", "Baseline min loss", "Baseline at .003", "muP LR*", "muP min loss", "muP at .003"],
               ["128", "8.3e-3", "3.6215", "3.6931 (+0.07)", "3.0e-3", "3.6307", "3.6307 (+0.00)"],
               ["256", "4.2e-3", "3.3967", "3.4067 (+0.01)", "2.0e-3", "3.4015", "3.4177 (+0.02)"],
               ["512", "2.4e-3", "3.2240", "3.2291", "2.4e-3", "3.2240", "3.2291"],
               ["1024", "1.06e-3", "3.0905", "3.2431 (+0.15)", "2.33e-3", "3.0928", "3.1032 (+0.01)"]],
              [0.55 * inch, 0.85 * inch, 1.0 * inch, 1.05 * inch, 0.7 * inch, 0.9 * inch, 1.0 * inch]),
        figure("p42_width_curves.png", width=6.1 * inch,
               text="Loss vs base LR per width (stars: fitted optima) and fitted optimum vs width; dotted line: LR .003."),
        figure("p42_width_losses.png", width=3.6 * inch, text="Fitted minimum loss and loss at the transferred LR vs width."),
        bullets("<b>muP transfers better</b>: its optimum stays within 2-3e-3 over 8x width while the baseline's falls as "
                "width^-0.90.",
                "<b>But tuned losses are essentially equal</b> (within 0.01 at every width). Unlike the five-step test, where "
                "muP won by 0.1-0.7 nats, 2,344 Adam updates let the baseline adapt its badly scaled layers once its LR is "
                "tuned. Better transfer is not better tuned performance here."),
        figure("p42_alignment.png", width=6.1 * inch,
               text="At LR .003: mean update alignment alpha over training, omega_move, and final-norm feature movement."),
        bullets("<b>Beyond five updates, update alignment rises then decays</b>: alpha ~0.7 at update 1, ~0.9 by update 10, "
                "then falling steadily to ~0.6 by the end at every width and under both prescriptions. omega_move stays "
                "at 0.50 throughout.",
                "<b>Finite-width effects:</b> at update 1 the embedding holds 90-99% of the squared gradient norm and clipping "
                "scales updates by 0.03-0.26, differently per width (baseline: 0.26 at w128, 0.06 at w512; muP: 0.03 at "
                "w128). The baseline's embedding scale 1/n is comparable to sqrt(eps) = 0.003, so its first-norm output RMS "
                "falls from 0.93 (w128) to 0.52 (w512); under muP it is 0.52 at every width. Both effects vanish after "
                "~1% of training (clip coefficient 1.0)."),
        sub("(c) Predict a held-out width (1024)."),
        para("Pre-registered (p42c_preregistered_predictions.md): power laws over widths 128-512 predict "
             "<b>1.26e-3</b> (baseline, width^-0.90) and <b>1.94e-3</b> (muP, width^-0.16)."),
        table([["Width 1024", ".00075", ".0015", ".003 (transfer)", ".006", "Prediction", "Local optimum"],
               ["Baseline", "3.1406", "3.1418", "3.2431 (+0.140)", "3.3249", "1.26e-3: <b>3.1033</b> (best)", "1.06e-3"],
               ["muP", "3.2397", "3.1245", "3.1032 (+0.005)", "3.2540", "1.94e-3: <b>3.0983</b> (best)", "2.33e-3"]],
              [0.75 * inch, 0.6 * inch, 0.6 * inch, 1.05 * inch, 0.6 * inch, 1.6 * inch, 0.9 * inch]),
        bullets("<b>Fitting the width dependence beats direct transfer for both prescriptions</b>: both predictions were the "
                "best sampled LR. For the baseline it is essential (transfer costs 0.140); for muP it is a small refinement "
                "(0.005) because its optimum barely moves.",
                "Tuned losses are again within 0.003 of each other (3.0905 vs 3.0928)."),
        sub("(d) Transfer across depth (width 512; depths 4, 8, 16)."),
        table([["Depth", "muP: min / at .003", "Depth-muP: min / at .003", "CompleteP: min / at .003"],
               ["4", "3.3150 / 3.3175", "3.3238* / 3.3491", "<b>3.3127</b> / 3.3150"],
               ["8", "3.2240 / 3.2291", "(identical)", "(identical)"],
               ["16", "3.1723* / 3.1763", "<b>3.1635</b> / 3.1639", "3.1713* / 3.1868"]],
              [0.6 * inch, 1.8 * inch, 1.9 * inch, 1.9 * inch]),
        para("* optimum at the lowest sampled LR (.0015)."),
        figure("p42_depth.png", width=6.1 * inch, text="Loss vs LR at depths 4, 8, 16 and loss vs depth (fitted minimum and "
                                                         "loss at the transferred LR)."),
        bullets("Over 4-16 layers the depth corrections matter by only ~0.01. At depth 16, Depth-muP both transfers best "
                "(optimum ~2.8e-3, cost at .003 = 0.000) and reaches the lowest loss (0.009 below muP). At depth 4 it is "
                "worst: its r^-1/2 multipliers enlarge the branches and block LR when r < 1. CompleteP transfers at depth 4 "
                "but its optimum drifts low at 16 (+0.016 at .003).",
                "A 4x depth range is too small for the sqrt(L) residual growth seen in 4.1 to dominate; standard muP already "
                "transfers within 0.005. Depth corrections are a large-depth effect."),
        sub("(e) Identify the training regime."),
        table([["Config (LR)", "Logit RMS: peak updates 0-5 -> end", "Feature movement: update 5 -> end", "alpha: early -> late",
                "omega_move"],
               ["Baseline w128 (.003)", "0.99 -> 4.36", "1.29 -> 2.02", "0.71 -> 0.61", "0.50"],
               ["Baseline w1024 (.003)", "1.01 -> 4.16", "1.33 -> 1.16", "0.86 -> 0.67", "0.50"],
               ["Baseline w1024 (best .00126)", "0.99 -> 3.94", "1.33 -> 1.36", "0.84 -> 0.64", "0.50"],
               ["muP w128 (.003)", "1.97 -> 4.44", "1.37 -> 1.26", "0.77 -> 0.59", "0.50"],
               ["muP w1024 (.003)", "0.73 -> 4.07", "1.32 -> 1.39", "0.85 -> 0.64", "0.50"],
               ["CompleteP d16 (best .0015)", "0.99 -> 4.01", "1.32 -> 1.53", "0.82 -> 0.62", "0.50"]],
              [1.6 * inch, 1.5 * inch, 1.4 * inch, 1.0 * inch, 0.7 * inch]),
        bullets("<b>Alignment by stage:</b> early updates are strongly aligned (alpha 0.71-0.86, rising with width), "
                "supporting muP's alpha ~ 1; late in training alpha ~0.6-0.67 at every width and depth, so muP's eta/m is "
                "conservative then. omega_move = 0.50 at every width, depth and stage: the omega = 1 readout assumption is "
                "never supported.",
                "<b>Larger early transients can still give a better final loss:</b> muP at width 128 has twice the early "
                "logit RMS of the baseline (1.97 vs 0.99) yet ends 0.06 lower at the same LR (3.631 vs 3.693).",
                "<b>A shift in optimal LR alone does not show the model has left the stable muP regime.</b> muP's optimum "
                "moves between 2.0e-3 and 3.0e-3 across widths while alpha, omega, feature movement and final logits are "
                "unchanged; on these flat curves such shifts are within ~0.01 nats and fitting noise.",
                "<b>Fixed WD:</b> the course AdamW decays by LR x WD per step, so muP's hidden LR eta/m also scales hidden "
                "decay by 1/m. With WD fixed at .1 the hidden averaging window 1/(LR x WD) grows m-fold at width 1024 and "
                "shrinks 4x at width 128. Problem 2 showed LR x WD is what matters, so part of muP's residual LR drift is "
                "likely this coupling; a width-invariant recipe would scale hidden WD by m."),
        sub("Synthesis (Problem 4)."),
        para("muP makes the base LR approximately width-invariant both over five updates and over a full run, so a single "
             "tuned LR transfers from width 128 to 1024 (+0.005) while the baseline needs a fitted width^-0.9 rule. "
             "After tuning, however, both reach the same loss at these widths. The derivation's assumptions hold only "
             "partly: updates are strongly aligned with their inputs early (alpha ~0.9) but weakly late (~0.6), and the "
             "readout never aligns with the feature change (omega = 0.5). Depth corrections are essential at depth 100-1000, "
             "where standard muP's residual stream explodes, but are a ~0.01 effect at depth 4-16."),
    ]


def P5_SECTION():
    code = ParagraphStyle("code", fontName="Courier", fontSize=8, leading=10)
    diff = """+def p5():
+    key = "a2-p5-wdrescue"
+    pairs = [(0.006, 0.262), (0.012, 0.131), (0.012, 0.1)]
+    return [config(tokens=153_600_000, learning_rate=lr, weight_decay=wd,
+                   run_name_suffix=key, wandb_tags=(key,)) for lr, wd in pairs]
# launch: python -m experiments.a2.awd.queue_driver experiments.a2.awd.later_runs:p5"""
    return [
        PageBreak(),
        sec("Problem 5 (optional): Can weight decay rescue a learning rate that is too high?"),
        para("<b>Question (single answer, Medium).</b> Default d8 recipe, 153.6M tokens, linear decay, seed 42. The supplied "
             "joint sweep puts the optimum near peak LR 1.9e-3, WD .85 (LR x WD = 1.57e-3, loss ~3.18). With the default "
             "WD .1, peak LR .006 gives 3.3056. Keep peak LR .006 but raise WD to .262, so that LR x WD matches the "
             "optimum's product. What final validation loss do you expect? (a) above 3.30 (no rescue); (b) 3.25-3.30 "
             "(partial); (c) 3.20-3.25 (most of the gap); (d) below 3.20 (full rescue)."),
        para("<b>Pre-registered prediction</b> (p5_preregistered_prediction.md, before running): <b>(b), ~3.28</b>, because "
             "the 153.6M quadratic gives 3.284 and the LR-WD valley is steeper than the constant-product line "
             "(log-slope -2.3): matching the product only partly compensates, since LR also sets per-step noise."),
        Paragraph(diff.replace("\n", "<br/>").replace(" ", "&nbsp;"), code),
        Spacer(1, 6),
        table([["Peak LR", "WD", "LR x WD", "Val loss"],
               [".006", ".1 (supplied)", "6.0e-4", "3.3056"],
               [".006", ".262 (product-matched)", "1.57e-3", "<b>3.2870</b>"],
               [".012", ".1", "1.2e-3", "3.3548"],
               [".012", ".131 (product-matched)", "1.57e-3", "3.3471"]],
              [0.9 * inch, 1.7 * inch, 0.9 * inch, 1.0 * inch]),
        para("<b>Answer: (b), 3.287.</b> Matching the product recovers 0.019 of the 0.13 gap to the joint optimum at 3.2x "
             "the optimal LR, and only 0.008 at 6.5x. The answer is robust to seed noise (sd ~0.003; 3.287 is 0.013 from "
             "the nearest boundary). My prediction was right in band and within 0.003 in value; the quadratic "
             "extrapolation to LR .012 (3.448) was far too pessimistic (measured 3.347), but the small size of the "
             "rescue held. <b>Lesson:</b> LR and WD trade off only near the optimum; far above it, the LR's per-step "
             "noise dominates and no WD fixes it."),
    ]


def sec(title):
    return Paragraph(title, h1)


def sub(title):
    return Paragraph(f"<b>{title}</b>", body)


def para(text):
    return Paragraph(text, body)


story = [Paragraph(LABEL, title)]

# ---------------- Problem 1 ----------------
story += [
    sec("Problem 1a: Learning from small training budgets"),
    para("<b>Method.</b> Provided runs at 153.6M, 307.2M and 614.4M tokens with peak LR 1.5e-3, 3e-3 and 6e-3 "
         "(all other settings default). For each budget I fit a parabola of final val loss vs log2(LR) and took its "
         "minimum as the optimal LR. Uncertainty comes from resampling each loss with seed noise (sd 0.003, from A1)."),
    table([["Tokens", "Val loss (LR 1.5e-3 / 3e-3 / 6e-3)", "Fitted optimal LR", "80% interval"],
           ["153.6M", "3.2442 / 3.2291 / 3.3056", "2.38e-3", "2.30 - 2.45e-3"],
           ["307.2M", "3.0709 / 3.0562 / 3.0715", "2.98e-3", "2.79 - 3.18e-3"],
           ["614.4M", "2.9473 / 2.9256 / 2.9408", "3.19e-3", "3.03 - 3.39e-3"]],
          [0.9 * inch, 2.5 * inch, 1.4 * inch, 1.4 * inch]),
    figure("p1_loss_vs_lr.png", width=4.0 * inch, text="Loss vs LR per budget (parabola fits; star = fitted optimum). P1(b) budgets included."),
    Paragraph("<b>The optimal LR increases with the token budget.</b>", body),
    bullets("Proposed rule: <b>LR* = 3.28e-3 x (D / 614.4M)^0.21</b> (log-log least squares through the three optima).",
            "This is the opposite of convex-optimization intuition, where more steps means a smaller LR (~1/sqrt(T)). "
            "Longer runs tolerate, and want, a slightly larger peak LR.",
            "The increase is slowing down: +25% for the first doubling of D, only +7% for the second. "
            "So a saturating rule (LR* = 3.30e-3 - 0.92e-3 (D/153.6M)^-1.52, levelling off near 3.3e-3) fits the three points about as well."),
    PageBreak(),
    sec("Problem 1b: Predict before increasing the budget"),
    para("<b>Pre-registered</b> before loading any P1(b) data (p1b_preregistered_predictions.md, 2026-09-30 16:14):"),
    table([["Tokens", "Predicted (power law)", "Predicted (saturating)", "Fitted from P1(b)", "80% interval"],
           ["1.2288B", "3.80e-3", "3.26e-3", "<b>3.88e-3</b>", "3.36 - 5.76e-3"],
           ["1.8432B", "4.14e-3", "3.28e-3", "<b>3.43e-3</b>", "2.98 - 4.57e-3"],
           ["2.4576B", "4.40e-3", "3.29e-3", "<b>2.91e-3</b>", "2.52 - 3.28e-3"]],
          [0.9 * inch, 1.4 * inch, 1.45 * inch, 1.3 * inch, 1.25 * inch]),
    figure("p1_optimal_lr_vs_budget.png", width=3.8 * inch,
           text="Fitted optimal LR vs budget with both pre-registered rules. Left of the dotted line = fit (P1a), right = held out (P1b)."),
    Paragraph("<b>The small-budget trend does not continue.</b>", body),
    bullets("The power law nailed 1.2B (3.80e-3 predicted vs 3.88e-3), then the optimum <b>turned around and fell</b> "
            "(3.43e-3, then 2.91e-3). At 2.46B the power law overshot by about 50%.",
            "The saturating rule was closer at 1.84B but also missed the decline.",
            "<b>The loss-vs-LR curve flattens with budget</b> (curvature 0.046 at 153.6M vs ~0.008 at 1.2-2.5B), so the exact "
            "LR matters less at large budgets and the fitted optima get much less certain.",
            "Overall the optimum rises ~35% from 153.6M to ~1B tokens, then plateaus or declines, staying within about "
            "2.4-3.9e-3. Extrapolating a small-budget power law several times beyond the fitted range overshoots."),
]
story += P1C_SECTION()
story += P1D_SECTION()
story += [
    sec("Problem 1e (optional): Schedule variants"),
    para("Supplied cosine-decay runs (P1e, WD .1) analysed with the same parabola method and compared with linear decay (P1a)."),
    table([["Tokens", "Linear LR*", "Cosine LR*", "Linear best loss", "Cosine best loss", "Cosine penalty"],
           ["153.6M", "2.38e-3", "2.26e-3", "3.2240", "3.2411", "+0.017"],
           ["307.2M", "2.98e-3", "2.53e-3", "3.0562", "3.0643", "+0.008"],
           ["614.4M", "3.19e-3", "3.19e-3", "2.9255", "2.9331", "+0.008"]],
          [0.85 * inch, 0.95 * inch, 0.95 * inch, 1.15 * inch, 1.15 * inch, 1.1 * inch]),
    bullets("The optimal peak LR transfers between schedules to within ~15% and rises with budget under both "
            "(linear D^0.21, cosine D^0.25). Reusing the linear optimum for cosine would cost under 0.002.",
            "Cosine is consistently slightly worse (0.008-0.017), most at the smallest budget. Linear decay spends more "
            "of training at intermediate LRs and gives a longer annealing phase."),
    sub("Synthesis (Problem 1)."),
    P1_SYNTHESIS(),
]

# ---------------- Problem 2 ----------------
story += [
    sec("Problem 2a: Let LR and WD vary together"),
    para("<b>Method.</b> For each budget I fit L(x, y) = a + bx + cy + dx^2 + exy + fy^2 with x = ln LR, y = ln WD "
         "to the 9 supplied runs (least squares) and took the stationary point (a minimum at every budget: both Hessian "
         "eigenvalues positive)."),
    table([["Tokens", "Quad R^2", "LR*", "WD*", "LR* x WD*", "Best measured (LR, WD): loss", "P1 best (WD .1)", "Gain"],
           ["153.6M", "0.996", "1.86e-3", "0.85", "1.57e-3", "(.0015, 1.6): 3.1844", "3.2291", "0.045"],
           ["307.2M", "0.974", "1.60e-3", "0.52", "8.3e-4", "(.0015, .8): 3.0282", "3.0562", "0.028"],
           ["614.4M", "0.984", "2.47e-3", "0.26", "6.4e-4", "(.003, .2): 2.9178", "2.9256", "0.008"],
           ["1.2288B", "0.984", "2.70e-3", "0.16", "4.4e-4", "(.003, .2): 2.8361", "2.8378", "0.002"]],
          [0.7 * inch, 0.6 * inch, 0.65 * inch, 0.5 * inch, 0.75 * inch, 1.55 * inch, 0.85 * inch, 0.5 * inch]),
    figure("p2_contours.png", width=6.1 * inch,
           text="Fitted contours (labels: loss above the fitted minimum), measured points, fitted optimum (star) and the "
                "line LR x WD = LR* x WD* (dashed)."),
    figure("p2_optima_vs_tokens.png", width=5.9 * inch,
           text="Fitted optima vs tokens with power-law fits (grey: P1's LR optimum at fixed WD .1)."),
    bullets("<b>Optimal WD falls cleanly with budget</b> (WD* ~ D^-0.82). <b>Optimal LR does not</b> follow a single power "
            "law (R^2 0.68): it dips at 307M and then rises, and it sits below P1's fixed-WD optimum at every budget "
            "(the model prefers a smaller LR when it can also use more decay).",
            "The LR trend is similar in shape to Problem 1's (rising from 307M to 1.2B) but ~20-40% lower in level."),
    figure("p2_joint_vs_p1.png", width=4.6 * inch, text="Best measured loss with and without tuning WD, and the difference."),
    bullets("<b>The benefit of tuning WD shrinks with budget:</b> 0.045 at 153.6M, 0.028, 0.008, and 0.002 at 1.2B. "
            "The default WD .1 is far too small for short runs and close to right by ~1B tokens."),
    sec("Problem 2b: From coupling to a joint scaling law"),
    bullets("<b>Increasing one hyperparameter can compensate for decreasing the other.</b> Every fitted contour is a "
            "tilted ellipse (normalised cross-term 0.48-0.88), and its flat direction has log-log slope between -1.1 and "
            "-2.8: along it, raising LR while lowering WD barely changes the loss. The constant-product line "
            "(slope -1) runs close to the valley floor, exactly so at 1.2B.",
            "<b>Power-law R^2 across the four budgets:</b> WD* 0.994, LR* x WD* 0.967, LR* 0.678. The product is far "
            "more regular than LR, but WD alone is the tightest here; most of the product's trend comes from WD.",
            "Product law: <b>LR* x WD* = 2.78e-4 at 2.4576B</b> (~D^-0.59). In steps, the AdamW averaging timescale "
            "1/(LR x WD) is 637, 1200, 1570 and 2290 updates, i.e. 27%, 26%, 17%, 12% of training: longer runs prefer "
            "averaging over more steps but a smaller fraction of training (a D^-1 law would keep the fraction fixed)."),
]
story += P2C_SECTION()
story += [sub("Synthesis (Problem 2)."), P2_SYNTHESIS()]

# ---------------- Problem 3.1 ----------------
story += [
    sec("Problem 3.1: What does the noisy quadratic model predict about batch size?"),
    para("<b>Method.</b> Exact simulation of f(w) = w'Hw/2, H = diag(1, 10), gradient noise N(0, sigma^2 I/B), "
         "N = 8192 examples (N/B constant-LR updates), w0 ~ N(0, diag(1, .1)). 2048 independent samples per point, "
         "common random numbers across LRs, 41-61 log-spaced LRs per batch; the optimum is a parabola in log2 LR "
         "(log loss) through the best grid point and its neighbours. Code: p31_nqm.py."),
    sub("(a, b) SGD, RMSProp and Adam (sigma = 1)."),
    table([["Optimizer", "Exponent p (B = 1-64)", "Predicted LR at 256", "Tuned LR at 256", "Loss: predicted / tuned"],
           ["SGD", "0.99", "0.147", "0.125", "6.9e-4 / 5.3e-4 (x1.29)"],
           ["RMSProp", "0.90", "0.094", "0.095", "1.20e-2 / 1.21e-2 (x0.99)"],
           ["Adam", "0.89", "0.091", "0.179", "1.49e-2 / 1.35e-2 (x1.11)"]],
          [0.9 * inch, 1.45 * inch, 1.3 * inch, 1.15 * inch, 1.9 * inch]),
    figure("p31_ab.png", width=5.8 * inch, text="Optimal LR vs batch (dashed: fit on B <= 64; star: prediction at 256), "
                                                "and best loss after tuning LR."),
    bullets("<b>SGD follows linear scaling (p = 0.99) up to B ~ 128 and breaks at 256-512</b>: the predicted LR runs into "
            "the stability limit 2/h_max = 0.2, so the tuned LR saturates (0.16 at B = 512) and the loss jumps (x4 from "
            "256 to 512). At fixed N, doubling B halves the gradient-noise variance but also halves the number of "
            "updates; doubling LR compensates exactly only while the loss is noise-dominated and the LR is far "
            "below the curvature limit.",
            "<b>RMSProp and Adam scale with p ~ 0.9</b>, close to SGD's but not equal. A similar exponent did not imply "
            "similar transfer: RMSProp's prediction is essentially exact at 256, while Adam's tuned LR is 2x the "
            "prediction (its loss-LR curve is very flat there, so the cost is only x1.11).",
            "<b>Prediction for the language model:</b> Adam's optimal LR should grow sublinearly with batch, between "
            "sqrt(B) (noise-dominated second moment) and B (signal-dominated), and the rule should hold only below "
            "a critical batch where fewer updates start to dominate. Beyond it, the optimal LR saturates and the "
            "loss rises."),
    sub("(c) Noise level and curvature."),
    table([["sigma", "SGD p (2D / scalar)", "RMSProp p (2D / scalar)", "Adam p (2D / scalar)",
            "Adam loss x at predicted LR, B=256 (2D / scalar)"],
           ["1", "0.99 / 1.00", "0.90 / 0.96", "0.89 / 0.94", "1.11 / 1.18"],
           ["10", "0.99 / 0.99", "0.54 / 0.56", "0.52 / 0.55", "1.18 / 1.36"],
           ["100", "0.99 / 1.01", "0.50 / 0.51", "0.50 / 0.51", "1.01 / 1.01"],
           ["300", "1.01 / 1.04", "0.51 / 0.54", "0.51 / 0.53", "1.01 / 1.02"]],
          [0.55 * inch, 1.3 * inch, 1.45 * inch, 1.3 * inch, 2.1 * inch]),
    figure("p31_c.png", width=5.4 * inch, text="Fitted exponent vs noise scale for both curvature models."),
    bullets("<b>The ratio of signal to noise changes the rule for adaptive optimizers but not for SGD.</b> Adam's "
            "step is m/sqrt(v) with v ~ (Hw)^2 + sigma^2/B. When noise dominates v, the step is ~ LR sqrt(B)/sigma "
            "times the gradient, i.e. SGD with effective LR LR sqrt(B)/sigma; SGD wants an effective LR ~ B, so "
            "LR* ~ sqrt(B) (p = 0.5, seen for sigma >= 10). At sigma = 1 the signal term is a large part of v, the "
            "update becomes sign-like and nearly batch-independent, and p moves toward 1.",
            "Transfer is worst in the crossover (sigma = 10): the batch at which signal starts to dominate v falls "
            "between the source range and 256, so the source exponent no longer applies (RMSProp x2.07 on the "
            "scalar loss).",
            "<b>Curvature sensitivity:</b> the exponents are almost identical for the scalar loss, but the breakdown point "
            "moves: with h = 1 SGD's stability limit is 2 instead of 0.2, so linear scaling still transfers at B = 256 "
            "(x1.04 vs x1.29). The range of validity depends on curvature; the exponents depend on noise."),
    sub("(d) Momentum at small and large batches."),
    table([["Setting", "B = 16 best loss (LR*)", "B = 256 best loss (LR*)"],
           ["SGD mu = 0", "3.33e-4 (0.010)", "5.39e-4 (0.125)"],
           ["SGD mu = 0.9", "3.07e-4 (0.00096), -8%", "4.09e-3 (0.060), x7.6 worse"],
           ["SGD best mu", "mu = .9: 3.07e-4", "mu = .7: 2.67e-4, -51%"],
           ["Adam beta1 = 0", "9.89e-4 (0.0077)", "1.21e-2 (0.095)"],
           ["Adam beta1 = 0.9", "9.15e-4, -7%", "1.35e-2, +11%"],
           ["Adam best beta1", "beta1 = .9: 9.15e-4", "beta1 = .8: 1.91e-3, -84%"]],
          [1.5 * inch, 2.3 * inch, 2.4 * inch]),
    figure("p31_d.png", width=6.1 * inch, text="Left: best loss vs beta1 / mu relative to no momentum. Right: learning "
                                                "curves of the best momentum setting vs the no-momentum baseline."),
    bullets("<b>Momentum helps little at small batch and a lot at large batch</b> (Adam: -7% at B = 16 vs -84% at 256; "
            "SGD: -8% vs -51%). At B = 16 the problem is noise-dominated and momentum mostly rescales the LR "
            "(SGD's tuned LR falls in proportion to 1 - mu: 0.010 to 0.00096). At B = 256 noise is small and the "
            "32 updates are curvature-limited, so momentum genuinely accelerates.",
            "<b>Too much momentum fails at large batch.</b> With only 32 updates, heavy ball contracts no faster than "
            "sqrt(mu) per step (0.95^32 ~ 0.19 for mu = 0.9), and Adam with beta1 >= 0.95 averages over more steps "
            "than it has. The best momentum shrinks as the update count shrinks.",
            "<b>Carry-over to language models:</b> I expect the direction (beta1 matters more at large batch) but a much "
            "smaller effect: LM runs have thousands of updates even at B = 256, so a moderate beta1 never "
            "runs out of steps, and LM losses are not purely curvature-limited."),
    sub("Synthesis (Problem 3.1)."),
    para("In the NQM, fixed-data batch scaling is a race between less gradient noise per update and fewer updates. "
         "While noise dominates, LR can be scaled to keep progress per example constant (linear for SGD, sqrt(B) "
         "for Adam once noise dominates the second moment). Beyond a critical batch set by curvature (stability) "
         "and signal-to-noise, the extra batch buys nothing, the optimal LR saturates, and only methods that use "
         "curvature better (momentum) still help. Exponents fitted on small batches are only trustworthy while "
         "you stay in the same regime."),
]
story += P32_SECTION()

# ---------------- Problem 4 ----------------
story += [
    sec("Problem 4.0: Scaling rules under each alignment assumption"),
    para("Write R(U, x0) = Theta(n^alpha) for update alignment and S(V0, dx) = Theta(n^omega) for readout/feature-change "
         "alignment (full: 1, none: 1/2). Use a = 0 for hidden and input matrices and c = 0 for the readout."),
    bullets("<b>Hidden</b> (p ~ n): order-one init needs a + b = 1/2, so b = 1/2. The direct update is "
            "Theta(eta n^(alpha-a-c)), so order one needs <b>c = alpha</b>. The W0 dx term is O(n^(1/2-a-b)) = O(1) automatically.",
            "<b>Readout</b> (q fixed): the direct update needs alpha - a - c = 0, so <b>a = alpha</b>. An order-one response to "
            "the feature change n^(omega-a-b) needs <b>b = omega - alpha</b>. Initial logits are Theta(n^(1/2-omega)), "
            "which is O(1) for omega >= 1/2.",
            "<b>Input</b> (fixed fan-in d): width never enters, so a + b = 0 and a + c = 0, i.e. (0, 0, 0) in every case.",
            "<b>Interaction:</b> ||n^-a dW dx||_RMS <= eta n^(alpha-a-c) ||U|| ||dx|| = O(eta) because c = alpha (hidden) "
            "or a = alpha (readout); it stays bounded in all four cases."),
    table([["alpha (update)", "omega (readout)", "Hidden (a,b,c)", "Readout (a,b,c)", "Hidden LR", "Readout multiplier",
            "Readout init var"],
           ["1 (full)", "1 (full)", "(0, 1/2, 1)", "(1, 0, 0)", "eta/m", "1/m", "1/n0"],
           ["1 (full)", "1/2 (none)", "(0, 1/2, 1)", "(1, -1/2, 0)", "eta/m", "1/m", "m/n0"],
           ["1/2 (none)", "1 (full)", "(0, 1/2, 1/2)", "(1/2, 1/2, 0)", "eta/sqrt(m)", "1/sqrt(m)", "1/n = 1/(m n0)"],
           ["1/2 (none)", "1/2 (none)", "(0, 1/2, 1/2)", "(1/2, 0, 0)", "eta/sqrt(m)", "1/sqrt(m)", "1/n0"]],
          [0.85 * inch, 0.9 * inch, 0.9 * inch, 0.95 * inch, 0.8 * inch, 0.95 * inch, 0.95 * inch]),
    para("All layers: hidden multiplier 1 and init variance 1/k (fan-in); input multiplier 1, init variance 1/d, LR eta; "
         "readout LR eta. <b>Update alignment sets the hidden LR exponent</b> (eta/m vs eta/sqrt(m)) and the readout "
         "multiplier; <b>readout alignment sets the readout initialisation</b>. The full/full row recovers the muP table "
         "(hidden eta/m, readout multiplier 1/m, readout variance 1/n0)."),
]
story += P41_SECTION()
story += P42_SECTION()
story += P5_SECTION()

SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.9 * inch, rightMargin=0.9 * inch, topMargin=0.85 * inch,
                  bottomMargin=0.85 * inch, title=LABEL).build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)

# Introduction

This project investigates the lifecycle of gender bias in NLP models. By
training a model from scratch and then fine-tuning it under different
constraints, we identify how linguistic representations can both inherit
and mitigate societal disparities.

## Overall Training Workflow

Figure [1](#fig:workflow){reference-type="ref" reference="fig:workflow"}
summarises the four-phase pipeline: (1) pre-training a BERT model on
general text, (2) profiling the Bias-in-Bios dataset, (3) fine-tuning
under two contrasting regimes, and (4) comparative evaluation of the
resulting models.

<figure id="fig:workflow" data-latex-placement="!htb">

<figcaption>End-to-end workflow. Solid arrows: data/model flow; dashed:
dataset reuse.</figcaption>
</figure>

# Phase 1: Base BERT Pre-training

The model was trained on 1M rows of OpenWebText (approx. 3.2 GB) with 14
layers and 12 heads (${\sim}$`<!-- -->`{=html}128.8M parameters),
totalling approximately 12 hours.

## Pre-training Performance Metrics

<figure id="fig:loss_perp" data-latex-placement="!htb">
<img src="/metrics/loss_perplexity.png" />
<figcaption>Training and Validation Loss/Perplexity.</figcaption>
</figure>

<figure id="fig:acc_f1" data-latex-placement="!htb">
<img src="/metrics/accuracy_f1.png" />
<figcaption>MLM Accuracy and F1 Score evolution.</figcaption>
</figure>

## Architecture Diagnostics

<figure id="fig:pos_acc" data-latex-placement="!htb">
<img src="/metrics/accuracy_by_position.png" />
<figcaption>MLM Accuracy by token position in sequence.</figcaption>
</figure>

<figure id="fig:per_tok" data-latex-placement="!htb">
<img src="/metrics/per_token_accuracy.png" />
<figcaption>Accuracy for top-K frequent tokens.</figcaption>
</figure>

<figure id="fig:tok_dist" data-latex-placement="!htb">
<img src="/metrics/token_distribution.png" />
<figcaption>Distribution of predicted vs. gold tokens.</figcaption>
</figure>

<figure id="fig:loss_dist" data-latex-placement="!htb">
<img src="/metrics/loss_distribution.png" />
<figcaption>Cross-entropy loss distribution across samples.</figcaption>
</figure>

The loss distribution (Figure [7](#fig:loss_dist){reference-type="ref"
reference="fig:loss_dist"}) shows a symmetric spread around a mean batch
loss of 5.90.

# Phase 2: Dataset Bias Profiling

We analyzed the Bias-in-Bios dataset (396k biographies) to identify
inherent gender skews across 28 professions. The overall split shows a
slight male majority (53.9% vs. 46.1%), but per-profession distributions
reveal far more extreme imbalances.

<figure id="fig:gender_stacked" data-latex-placement="!htb">
<img src="/biasbios_metrics/02_gender_breakdown_stacked.png" />
<figcaption>Absolute gender counts per profession.</figcaption>
</figure>

<figure id="fig:gender_pct" data-latex-placement="!htb">
<img src="/biasbios_metrics/03_gender_pct_horizontal.png" />
<figcaption>Percentage split of genders per profession.</figcaption>
</figure>

<figure id="fig:prob_female" data-latex-placement="!htb">
<img src="/biasbios_metrics/04_prob_female_dotplot.png" />
<figcaption>Probability of a biography being female given the
profession.</figcaption>
</figure>

<figure id="fig:bias_div" data-latex-placement="!htb">
<img src="/biasbios_metrics/05_bias_diverging.png" />
<figcaption>Diverging Bias Score (%Female <span
class="math inline">−</span> 50%).</figcaption>
</figure>

<figure id="fig:abs_gap" data-latex-placement="!htb">
<img src="/biasbios_metrics/06_absolute_gap.png" />
<figcaption>Absolute representation gap (samples needed for
parity).</figcaption>
</figure>

<figure id="fig:heatmap" data-latex-placement="!htb">
<img src="/biasbios_metrics/07_gender_heatmap.png" />
<figcaption>Gender distribution heatmap (% per profession).</figcaption>
</figure>

The heatmap (Figure [13](#fig:heatmap){reference-type="ref"
reference="fig:heatmap"}) shows *Rapper* (90% male) and *Dietitian* (93%
female) at the extremes, with *Journalist* and *Physician* near parity.

# Phase 3 & 4: Biased vs. Debiased Models

The "Biased" model was trained with stereotypical oversampling; the
"Debiased" model used counterfactual augmentation and balancing.

## Fine-tuning Architecture and Process

#### Classification head.

The `[CLS]` token's 768-dim hidden state passes through dropout
($p=0.1$) and a linear projection to 28 logits. During inference,
softmax and argmax select the predicted profession.

<figure id="fig:finetune_arch" data-latex-placement="!htb">

<figcaption>Fine-tuning architecture. Only <code>[CLS]</code> is
forwarded to the head.</figcaption>
</figure>

#### Training settings.

Both models share: AdamW, lr $2\times10^{-5}$, linear warm-up (6%),
weight decay $0.01$, `fp16`, gradient clip 1.0, 3 epochs. The only
difference is the training set.

## Debiasing: Counterfactual Data Augmentation {#sec:debiasing}

### Step 1: Gender Balancing {#step-1-gender-balancing .unnumbered}

For each profession, male and female subsets are sub-sampled to their
common minimum, producing a perfectly balanced dataset.

### Step 2: CDA {#step-2-cda .unnumbered}

Each biography is duplicated with gendered cues swapped
(*he*$\leftrightarrow$*she*, *Mr.*$\leftrightarrow$*Ms.*, etc.). The
profession label is preserved; the gender label is flipped.

<figure id="fig:cda" data-latex-placement="!htb">

<figcaption>CDA: each biography is duplicated with swapped gendered
surface forms. Profession label unchanged; gender label
flipped.</figcaption>
</figure>

## Overall Results

The debiased model reduced the average gender gap from 26.1% to 5.1%.

<figure id="fig:overall_comp" data-latex-placement="!htb">
<img src="/comparison_metrics/01_overall_comparison.png" />
<figcaption>Summary of Accuracy, F1, and Gender Gaps.</figcaption>
</figure>

## Bias Metrics

We focused on TPR Gap, EqOdds, and Stereotype Score. The EqOdds Gap fell
from 0.2740 to 0.0531, a 76% improvement, ensuring equal classification
probability regardless of gender
(Figure [20](#fig:radar){reference-type="ref" reference="fig:radar"}).

<figure id="fig:gender_gap" data-latex-placement="!htb">
<img src="/comparison_metrics/04_gender_gap_comparison.png" />
<figcaption>Per-profession gender accuracy gap comparison.</figcaption>
</figure>

<figure id="fig:gap_red" data-latex-placement="!htb">
<img src="/comparison_metrics/05_gap_reduction.png" />
<figcaption>Reduction in bias per category: Biased <span
class="math inline">→</span> Debiased.</figcaption>
</figure>

<figure id="fig:tpr_fpr" data-latex-placement="!htb">
<img src="/comparison_metrics/07_tpr_fpr_gaps.png" />
<figcaption>True Positive Rate and False Positive Rate
gaps.</figcaption>
</figure>

<figure id="fig:radar" data-latex-placement="!htb">
<img src="/comparison_metrics/08_bias_radar.png" />
<figcaption>Radar chart: biased model (red) vs. debiased (green) across
six bias axes.</figcaption>
</figure>

<figure id="fig:dem_parity" data-latex-placement="!htb">
<img src="/comparison_metrics/12_demographic_parity.png" />
<figcaption>Demographic parity difference comparison.</figcaption>
</figure>

## Performance Breakdown

The biased model degraded sharply on counter-stereotypical samples
(e.g., female surgeons, male nurses), confirming it used gender as a
classification shortcut. The debiased model achieved near-uniform
accuracy across stereotypical and counter-stereotypical samples alike.
The overall accuracy cost was negligible (69.5% $\to$ 69.0%) against a
26.1% $\to$ 5.1% reduction in Average Gender Gap.

<figure id="fig:gender_acc" data-latex-placement="!htb">
<img src="/comparison_metrics/02_gender_accuracy.png" />
<figcaption>Accuracy by gender: biased gap 0.0136 vs. debiased gap
0.0010.</figcaption>
</figure>

<figure id="fig:per_prof" data-latex-placement="!htb">
<img src="/comparison_metrics/03_per_profession_accuracy.png" />
<figcaption>Per-profession accuracy comparison.</figcaption>
</figure>

<figure id="fig:stereo" data-latex-placement="!htb">
<img src="/comparison_metrics/09_stereotype_analysis.png" />
<figcaption>Accuracy on Stereotypical vs. Counter-Stereotypical
samples.</figcaption>
</figure>

<figure id="fig:confusion" data-latex-placement="!htb">
<img src="/comparison_metrics/06_confusion_matrices.png" />
<figcaption>Normalized confusion matrices for both models.</figcaption>
</figure>

<figure id="fig:training_curves" data-latex-placement="!htb">
<img src="/comparison_metrics/10_training_curves.png" />
<figcaption>Fine-tuning training curves: loss, validation accuracy, F1
macro.</figcaption>
</figure>

## Analysis of Fine-Tuning and Counterfactual Impact

The transition from a standard fine-tuned model to a debiased version
reveals critical insights into how BERT processes gendered data beyond
simple classification metrics.

**Mechanics of Initial Bias:** Initial fine-tuning on the Bias-in-Bios
dataset achieved a baseline accuracy of $\sim$`<!-- -->`{=html}69.5%.
However, this performance was \"brittle.\" As evidenced by the high
gender-profession correlations in the training data (Section 3), the
model learned to utilize gendered heuristics, such as pronouns and
name-based cues, as primary shortcuts for classification rather than
relying on professional context.

**Impact of Counterfactual Data Augmentation (CDA):** By implementing
CDA, we forced the model to encounter counter-stereotypical biographies.
This shift had a notable effect on the internal logic of the classifier.
Example include the Gender Accuracy Gap dropping from 26.1% in the
biased model to 5.1% in the debiased version (as summarized in Figure
18), the minor 0.5% dip in overall accuracy (falling to 69.0%). This
indicates the model successfully stopped \"gaming\" the accuracy score
by using biased shortcuts, resulting in a more robust and ethically
sound classifier that generalizes better to counter-stereotypical
individuals, as well as the Equality of Odds (EqOdds) Gap falling by 76%
(from 0.2740 to 0.0531). As visualized in the Radar Chart (Figure 23),
the debiased model exhibits significantly higher parity across all
measured fairness dimensions.

# Conclusion

Across all 30 metrics, counterfactual data augmentation significantly
improves model fairness without substantial loss in predictive power.
LLM-based pipelines inherit systemic biases from datasets like
Bias-in-Bios, where professions such as "Dietitian" and "Nurse" are
heavily female-skewed while "Surgeon" and "Software Engineer" remain
male-dominated. By balancing the training distribution via CDA, we
successfully decoupled gender indicators from professional labels,
achieving a 76% reduction in EqOdds Gap at negligible cost to overall
classification accuracy.

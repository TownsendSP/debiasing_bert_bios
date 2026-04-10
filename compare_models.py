#!/usr/bin/env python3
"""
compare_models.py
Compare biased vs debiased BERT profession classifiers.
Generates comprehensive metrics and publication-quality charts.

Usage:
    python compare_models.py
    python compare_models.py --biased_dir ./biased_bert --debiased_dir ./debiased_bert
    python compare_models.py -n 0  # use full test set
"""

import argparse
import os
import json
import time
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from transformers import BertForSequenceClassification, BertTokenizerFast
from datasets import load_from_disk, load_dataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROFESSION_LABELS = {
    0: "accountant", 1: "architect", 2: "attorney", 3: "chiropractor",
    4: "comedian", 5: "composer", 6: "dentist", 7: "dietitian",
    8: "dj", 9: "filmmaker", 10: "interior_designer", 11: "journalist",
    12: "model", 13: "nurse", 14: "painter", 15: "paralegal",
    16: "pastor", 17: "personal_trainer", 18: "photographer", 19: "physician",
    20: "poet", 21: "professor", 22: "psychologist", 23: "rapper",
    24: "software_engineer", 25: "surgeon", 26: "teacher", 27: "yoga_teacher",
}
GENDER_LABELS = {0: "male", 1: "female"}
NUM_LABELS = len(PROFESSION_LABELS)
DATASET_DIR = os.path.abspath("./biasbios_data")
OUT_DIR = "./comparison_metrics"

# Dark theme
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family": "monospace",
    "font.size": 11,
})

# Colors
BIASED_COLOR    = "#f47067"   # red
DEBIASED_COLOR  = "#57ab5a"   # green
MALE_COLOR      = "#6cb6ff"   # blue
FEMALE_COLOR    = "#f69d50"   # orange
ACCENT          = "#58a6ff"
NEUTRAL         = "#8b949e"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_model(model_dir):
    print(f"[*] Loading model: {model_dir}")
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"    {n/1e6:.1f}M params on {device}")
    return model, tokenizer, device


def load_test_data(dataset_dir=None):
    ddir = dataset_dir or DATASET_DIR
    if os.path.isdir(ddir):
        try:
            ds = load_from_disk(ddir)
            return ds["test"]
        except FileNotFoundError:
            ds = load_dataset(ddir)
            return ds["test"]
    return load_dataset("LabHC/bias_in_bios", split="test")


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i], truncation=True,
            max_length=self.max_len, padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def predict_batch(texts, model, tokenizer, device, batch_size=64, return_probs=False):
    """Predict profession IDs and optionally return full probability distributions."""
    ds = TextDataset(texts, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    all_preds, all_probs = [], []

    for batch in tqdm(loader, desc="Predicting", leave=False):
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.no_grad(), autocast("cuda"):
            logits = model(input_ids=ids, attention_mask=mask).logits
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        if return_probs:
            all_probs.extend(F.softmax(logits, dim=-1).cpu().numpy().tolist())

    if return_probs:
        return np.array(all_preds), np.array(all_probs)
    return np.array(all_preds)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_all_metrics(true_ids, pred_ids, genders, probs=None):
    """Compute comprehensive metrics broken down by gender and profession."""
    results = {}

    # Overall
    results["overall"] = {
        "accuracy": accuracy_score(true_ids, pred_ids),
        "f1_macro": f1_score(true_ids, pred_ids, average="macro", zero_division=0),
        "f1_weighted": f1_score(true_ids, pred_ids, average="weighted", zero_division=0),
        "precision_macro": precision_score(true_ids, pred_ids, average="macro", zero_division=0),
        "recall_macro": recall_score(true_ids, pred_ids, average="macro", zero_division=0),
    }

    # Per gender overall
    for g_id, g_name in GENDER_LABELS.items():
        g_mask = genders == g_id
        results[f"overall_{g_name}"] = {
            "accuracy": accuracy_score(true_ids[g_mask], pred_ids[g_mask]),
            "f1_macro": f1_score(true_ids[g_mask], pred_ids[g_mask], average="macro", zero_division=0),
            "f1_weighted": f1_score(true_ids[g_mask], pred_ids[g_mask], average="weighted", zero_division=0),
            "n_samples": int(g_mask.sum()),
        }

    # Per profession
    results["per_profession"] = {}
    for prof_id in range(NUM_LABELS):
        name = PROFESSION_LABELS[prof_id]
        mask = true_ids == prof_id
        if mask.sum() == 0:
            continue

        acc = (pred_ids[mask] == true_ids[mask]).mean()

        # Gender-specific
        m_mask = mask & (genders == 0)
        f_mask = mask & (genders == 1)
        m_acc = (pred_ids[m_mask] == true_ids[m_mask]).mean() if m_mask.sum() > 0 else 0
        f_acc = (pred_ids[f_mask] == true_ids[f_mask]).mean() if f_mask.sum() > 0 else 0

        # False positive rate per gender: how often does the model predict this
        # profession INCORRECTLY for each gender?
        not_mask = true_ids != prof_id
        m_not = not_mask & (genders == 0)
        f_not = not_mask & (genders == 1)
        m_fpr = (pred_ids[m_not] == prof_id).mean() if m_not.sum() > 0 else 0
        f_fpr = (pred_ids[f_not] == prof_id).mean() if f_not.sum() > 0 else 0

        # True positive rate (recall) per gender
        m_tpr = (pred_ids[m_mask] == prof_id).mean() if m_mask.sum() > 0 else 0
        f_tpr = (pred_ids[f_mask] == prof_id).mean() if f_mask.sum() > 0 else 0

        results["per_profession"][name] = {
            "accuracy": float(acc),
            "male_accuracy": float(m_acc),
            "female_accuracy": float(f_acc),
            "gender_acc_gap": float(abs(m_acc - f_acc)),
            "gender_acc_gap_signed": float(m_acc - f_acc),  # positive = favors men
            "male_tpr": float(m_tpr),
            "female_tpr": float(f_tpr),
            "tpr_gap": float(abs(m_tpr - f_tpr)),
            "male_fpr": float(m_fpr),
            "female_fpr": float(f_fpr),
            "fpr_gap": float(abs(m_fpr - f_fpr)),
            "n_total": int(mask.sum()),
            "n_male": int(m_mask.sum()),
            "n_female": int(f_mask.sum()),
        }

    # Aggregate bias metrics
    gaps = [v["gender_acc_gap"] for v in results["per_profession"].values()]
    tpr_gaps = [v["tpr_gap"] for v in results["per_profession"].values()]
    fpr_gaps = [v["fpr_gap"] for v in results["per_profession"].values()]

    results["bias_summary"] = {
        "mean_gender_acc_gap": float(np.mean(gaps)),
        "max_gender_acc_gap": float(np.max(gaps)),
        "mean_tpr_gap": float(np.mean(tpr_gaps)),
        "mean_fpr_gap": float(np.mean(fpr_gaps)),
        "equality_of_odds_gap": float(np.mean(tpr_gaps) + np.mean(fpr_gaps)),
        # Demographic parity: difference in positive prediction rates by gender
        "male_overall_acc": results["overall_male"]["accuracy"],
        "female_overall_acc": results["overall_female"]["accuracy"],
        "overall_gender_gap": abs(results["overall_male"]["accuracy"] - results["overall_female"]["accuracy"]),
    }

    # Stereotype score: for stereotyped professions, how much does the model
    # favor the stereotypical gender?
    female_stereo = {"nurse", "teacher", "dietitian", "yoga_teacher", "interior_designer",
                     "model", "paralegal", "psychologist"}
    male_stereo = {"surgeon", "software_engineer", "rapper", "dj", "pastor",
                   "chiropractor", "architect", "filmmaker"}

    stereo_scores = []
    for name, metrics in results["per_profession"].items():
        if name in female_stereo:
            # Stereotypical = female does better → gap should be negative
            stereo_scores.append(metrics["gender_acc_gap_signed"] * -1)
        elif name in male_stereo:
            # Stereotypical = male does better → gap should be positive
            stereo_scores.append(metrics["gender_acc_gap_signed"])

    results["bias_summary"]["stereotype_amplification_score"] = float(np.mean(stereo_scores)) if stereo_scores else 0

    return results


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------
def save_fig(fig, name, out_dir):
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_01_overall_comparison(m_biased, m_debiased, out):
    """Bar chart: overall metrics side by side."""
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    labels = ["Accuracy", "F1\n(macro)", "F1\n(weighted)", "Precision\n(macro)", "Recall\n(macro)"]
    b_vals = [m_biased["overall"][m] for m in metrics]
    d_vals = [m_debiased["overall"][m] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax.bar(x - w/2, b_vals, w, label="Biased", color=BIASED_COLOR, alpha=0.85)
    bars2 = ax.bar(x + w/2, d_vals, w, label="Debiased", color=DEBIASED_COLOR, alpha=0.85)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, color=BIASED_COLOR)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, color=DEBIASED_COLOR)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score"); ax.set_title("Overall Model Performance", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(max(b_vals), max(d_vals)) * 1.12)
    return save_fig(fig, "01_overall_comparison", out)


def plot_02_gender_accuracy(m_biased, m_debiased, out):
    """Grouped bars: accuracy by gender for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Male (Biased)", "Female (Biased)", "Male (Debiased)", "Female (Debiased)"]
    values = [
        m_biased["overall_male"]["accuracy"],
        m_biased["overall_female"]["accuracy"],
        m_debiased["overall_male"]["accuracy"],
        m_debiased["overall_female"]["accuracy"],
    ]
    colors = [MALE_COLOR, FEMALE_COLOR, MALE_COLOR, FEMALE_COLOR]
    hatches = ["", "", "//", "//"]
    edgecolors = [BIASED_COLOR, BIASED_COLOR, DEBIASED_COLOR, DEBIASED_COLOR]

    bars = ax.bar(categories, values, color=colors, edgecolor=edgecolors, linewidth=2, alpha=0.8)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    # Draw gap lines
    gap_b = abs(values[0] - values[1])
    gap_d = abs(values[2] - values[3])
    ax.annotate(f"Gap: {gap_b:.4f}", xy=(0.5, min(values[0], values[1])),
                fontsize=10, ha="center", color=BIASED_COLOR, fontweight="bold")
    ax.annotate(f"Gap: {gap_d:.4f}", xy=(2.5, min(values[2], values[3])),
                fontsize=10, ha="center", color=DEBIASED_COLOR, fontweight="bold")

    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Gender", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(values) * 1.1)
    return save_fig(fig, "02_gender_accuracy", out)


def plot_03_per_profession_accuracy(m_biased, m_debiased, out):
    """Horizontal bar chart: per-profession accuracy, biased vs debiased."""
    profs_b = m_biased["per_profession"]
    profs_d = m_debiased["per_profession"]
    names = sorted(profs_b.keys(), key=lambda n: profs_b[n]["accuracy"])

    fig, ax = plt.subplots(figsize=(12, 10))
    y = np.arange(len(names))

    ax.barh(y - 0.2, [profs_b[n]["accuracy"] for n in names], 0.35,
            color=BIASED_COLOR, alpha=0.8, label="Biased")
    ax.barh(y + 0.2, [profs_d[n]["accuracy"] for n in names], 0.35,
            color=DEBIASED_COLOR, alpha=0.8, label="Debiased")

    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Accuracy"); ax.set_title("Per-Profession Accuracy", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3, axis="x")
    return save_fig(fig, "03_per_profession_accuracy", out)


def plot_04_gender_gap_comparison(m_biased, m_debiased, out):
    """Butterfly chart: gender accuracy gap per profession, biased vs debiased."""
    profs_b = m_biased["per_profession"]
    profs_d = m_debiased["per_profession"]
    names = sorted(profs_b.keys(), key=lambda n: profs_b[n]["gender_acc_gap"], reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), sharey=True)
    y = np.arange(len(names))

    # Signed gaps: positive = male favored, negative = female favored
    gaps_b = [profs_b[n]["gender_acc_gap_signed"] for n in names]
    gaps_d = [profs_d[n]["gender_acc_gap_signed"] for n in names]

    colors_b = [MALE_COLOR if g > 0 else FEMALE_COLOR for g in gaps_b]
    colors_d = [MALE_COLOR if g > 0 else FEMALE_COLOR for g in gaps_d]

    ax1.barh(y, gaps_b, color=colors_b, alpha=0.8)
    ax1.axvline(0, color="#c9d1d9", lw=0.5)
    ax1.set_xlabel("<- Female favored     Male favored ->")
    ax1.set_title("BIASED Model", fontsize=13, fontweight="bold", color=BIASED_COLOR)
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=9)
    ax1.grid(True, alpha=0.3, axis="x")

    ax2.barh(y, gaps_d, color=colors_d, alpha=0.8)
    ax2.axvline(0, color="#c9d1d9", lw=0.5)
    ax2.set_xlabel("<- Female favored     Male favored ->")
    ax2.set_title("DEBIASED Model", fontsize=13, fontweight="bold", color=DEBIASED_COLOR)
    ax2.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Gender Accuracy Gap by Profession", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    return save_fig(fig, "04_gender_gap_comparison", out)


def plot_05_gap_reduction(m_biased, m_debiased, out):
    """Slope chart showing gap reduction per profession."""
    profs_b = m_biased["per_profession"]
    profs_d = m_debiased["per_profession"]
    names = sorted(profs_b.keys(), key=lambda n: profs_b[n]["gender_acc_gap"], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, name in enumerate(names):
        gap_b = profs_b[name]["gender_acc_gap"]
        gap_d = profs_d[name]["gender_acc_gap"]
        improved = gap_d < gap_b

        color = DEBIASED_COLOR if improved else BIASED_COLOR
        ax.plot([0, 1], [gap_b, gap_d], color=color, alpha=0.6, linewidth=2)
        ax.scatter([0], [gap_b], color=BIASED_COLOR, s=50, zorder=5)
        ax.scatter([1], [gap_d], color=DEBIASED_COLOR, s=50, zorder=5)

        # Label on left
        ax.text(-0.05, gap_b, name, ha="right", va="center", fontsize=8, color="#8b949e")

    # Mean lines
    mean_b = np.mean([profs_b[n]["gender_acc_gap"] for n in names])
    mean_d = np.mean([profs_d[n]["gender_acc_gap"] for n in names])
    ax.axhline(mean_b, color=BIASED_COLOR, ls="--", alpha=0.5, label=f"Mean biased: {mean_b:.4f}")
    ax.axhline(mean_d, color=DEBIASED_COLOR, ls="--", alpha=0.5, label=f"Mean debiased: {mean_d:.4f}")

    ax.set_xticks([0, 1]); ax.set_xticklabels(["Biased Model", "Debiased Model"], fontsize=12)
    ax.set_ylabel("Gender Accuracy Gap (|male - female|)")
    ax.set_title("Gender Gap Reduction: Biased → Debiased", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
    return save_fig(fig, "05_gap_reduction", out)


def plot_06_confusion_matrices(true_ids, biased_preds, debiased_preds, out):
    """Side-by-side confusion matrices (normalized)."""
    names = [PROFESSION_LABELS[i] for i in range(NUM_LABELS)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    for ax, preds, title, cmap_color in [
        (ax1, biased_preds, "Biased Model", "Reds"),
        (ax2, debiased_preds, "Debiased Model", "Greens"),
    ]:
        cm = confusion_matrix(true_ids, preds, labels=range(NUM_LABELS), normalize="true")
        im = ax.imshow(cm, cmap=cmap_color, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_LABELS)); ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticks(range(NUM_LABELS)); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title, fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Normalized Confusion Matrices", fontsize=15, fontweight="bold")
    plt.tight_layout()
    return save_fig(fig, "06_confusion_matrices", out)


def plot_07_tpr_fpr_gaps(m_biased, m_debiased, out):
    """TPR and FPR gaps by profession."""
    profs_b = m_biased["per_profession"]
    profs_d = m_debiased["per_profession"]
    names = sorted(profs_b.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    y = np.arange(len(names))

    # TPR gap
    tpr_b = [profs_b[n]["tpr_gap"] for n in names]
    tpr_d = [profs_d[n]["tpr_gap"] for n in names]
    ax1.barh(y - 0.2, tpr_b, 0.35, color=BIASED_COLOR, alpha=0.8, label="Biased")
    ax1.barh(y + 0.2, tpr_d, 0.35, color=DEBIASED_COLOR, alpha=0.8, label="Debiased")
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("TPR Gap (|male - female|)")
    ax1.set_title("True Positive Rate Gap", fontsize=13, fontweight="bold")
    ax1.legend(); ax1.grid(True, alpha=0.3, axis="x")

    # FPR gap
    fpr_b = [profs_b[n]["fpr_gap"] for n in names]
    fpr_d = [profs_d[n]["fpr_gap"] for n in names]
    ax2.barh(y - 0.2, fpr_b, 0.35, color=BIASED_COLOR, alpha=0.8, label="Biased")
    ax2.barh(y + 0.2, fpr_d, 0.35, color=DEBIASED_COLOR, alpha=0.8, label="Debiased")
    ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("FPR Gap (|male - female|)")
    ax2.set_title("False Positive Rate Gap", fontsize=13, fontweight="bold")
    ax2.legend(); ax2.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Equality of Odds Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    return save_fig(fig, "07_tpr_fpr_gaps", out)


def plot_08_bias_summary_radar(m_biased, m_debiased, out):
    """Radar/spider chart of bias summary metrics (lower = less biased)."""
    metrics = [
        ("Mean Acc Gap", "mean_gender_acc_gap"),
        ("Max Acc Gap", "max_gender_acc_gap"),
        ("Mean TPR Gap", "mean_tpr_gap"),
        ("Mean FPR Gap", "mean_fpr_gap"),
        ("EqOdds Gap", "equality_of_odds_gap"),
        ("Stereo Score", "stereotype_amplification_score"),
    ]

    labels = [m[0] for m in metrics]
    b_vals = [abs(m_biased["bias_summary"][m[1]]) for m in metrics]
    d_vals = [abs(m_debiased["bias_summary"][m[1]]) for m in metrics]

    # Normalize to 0-1 for radar
    max_vals = [max(b, d, 0.001) for b, d in zip(b_vals, d_vals)]
    b_norm = [b / m for b, m in zip(b_vals, max_vals)]
    d_norm = [d / m for d, m in zip(d_vals, max_vals)]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    b_norm += b_norm[:1]
    d_norm += d_norm[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, b_norm, alpha=0.2, color=BIASED_COLOR)
    ax.plot(angles, b_norm, color=BIASED_COLOR, linewidth=2, label="Biased")
    ax.fill(angles, d_norm, alpha=0.2, color=DEBIASED_COLOR)
    ax.plot(angles, d_norm, color=DEBIASED_COLOR, linewidth=2, label="Debiased")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Bias Metrics (closer to center = less biased)", fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_facecolor("#161b22")
    return save_fig(fig, "08_bias_radar", out)


def plot_09_stereotype_analysis(m_biased, m_debiased, out):
    """Focus on stereotyped professions: male-stereotyped vs female-stereotyped."""
    female_stereo = {"nurse", "teacher", "dietitian", "yoga_teacher", "interior_designer",
                     "model", "paralegal", "psychologist"}
    male_stereo = {"surgeon", "software_engineer", "rapper", "dj", "pastor",
                   "chiropractor", "architect", "filmmaker"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, stereo_set, title, anchor_gender in [
        (ax1, female_stereo, "Female-Stereotyped Professions", "female"),
        (ax2, male_stereo, "Male-Stereotyped Professions", "male"),
    ]:
        names = sorted(stereo_set)
        y = np.arange(len(names))

        for model_metrics, color, offset, label in [
            (m_biased, BIASED_COLOR, -0.2, "Biased"),
            (m_debiased, DEBIASED_COLOR, 0.2, "Debiased"),
        ]:
            m_accs = [model_metrics["per_profession"][n]["male_accuracy"] for n in names]
            f_accs = [model_metrics["per_profession"][n]["female_accuracy"] for n in names]

            ax.barh(y + offset, m_accs, 0.18, color=MALE_COLOR, alpha=0.7,
                    label=f"Male ({label})" if offset < 0 else "")
            ax.barh(y + offset + 0.18, f_accs, 0.18, color=FEMALE_COLOR, alpha=0.7,
                    label=f"Female ({label})" if offset < 0 else "")

        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Accuracy")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        if ax == ax1:
            ax.legend(fontsize=8)

    fig.suptitle("Stereotype Analysis: Gender-Specific Accuracy", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return save_fig(fig, "09_stereotype_analysis", out)


def plot_10_training_curves(out):
    """Plot training curves from both CSVs if they exist."""
    biased_csv = "./biased_training_metrics.csv"
    debiased_csv = "./debiased_training_metrics.csv"

    if not os.path.exists(biased_csv) or not os.path.exists(debiased_csv):
        print("  [!] Training CSVs not found, skipping training curve plot")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for csv_path, color, label in [
        (biased_csv, BIASED_COLOR, "Biased"),
        (debiased_csv, DEBIASED_COLOR, "Debiased"),
    ]:
        df = pd.read_csv(csv_path)
        tr = df[df.phase == "train"].copy()
        val = df[df.phase == "val"].copy()

        tr["loss"] = pd.to_numeric(tr["loss"], errors="coerce")
        val["loss"] = pd.to_numeric(val.get("loss", pd.Series()), errors="coerce")
        val["accuracy"] = pd.to_numeric(val.get("accuracy", pd.Series()), errors="coerce")
        val["f1_macro"] = pd.to_numeric(val.get("f1_macro", pd.Series()), errors="coerce")

        # Loss
        win = max(len(tr) // 30, 3) if len(tr) > 10 else 1
        axes[0].plot(tr.global_step, tr.loss.rolling(win, min_periods=1).mean(),
                     color=color, lw=2, label=label, alpha=0.8)
        if len(val):
            axes[0].scatter(val.global_step, val.loss, color=color, s=60, zorder=5,
                            edgecolors="white", lw=0.5)

        # Val accuracy
        if len(val) and "accuracy" in val.columns:
            axes[1].plot(val.global_step, val.accuracy, color=color, lw=2,
                         marker="o", markersize=8, label=label)

        # Val F1
        if len(val) and "f1_macro" in val.columns:
            axes[2].plot(val.global_step, val.f1_macro, color=color, lw=2,
                         marker="o", markersize=8, label=label)

    axes[0].set(xlabel="Step", ylabel="Loss", title="Training Loss")
    axes[1].set(xlabel="Step", ylabel="Accuracy", title="Validation Accuracy")
    axes[2].set(xlabel="Step", ylabel="F1 Macro", title="Validation F1 (Macro)")
    for ax in axes:
        ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("Training Curves: Biased vs Debiased", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return save_fig(fig, "10_training_curves", out)


def plot_11_summary_dashboard(m_biased, m_debiased, out):
    """Single-page summary dashboard with key numbers."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0e1117")

    # Title
    fig.text(0.5, 0.96, "BIAS DETECTION REPORT", ha="center", fontsize=22,
             fontweight="bold", color="#c9d1d9",
             path_effects=[pe.withStroke(linewidth=2, foreground="#30363d")])
    fig.text(0.5, 0.92, "Biased vs Debiased BERT Profession Classifier",
             ha="center", fontsize=13, color="#8b949e")

    gs = GridSpec(3, 4, figure=fig, top=0.88, bottom=0.05, left=0.05, right=0.95,
                  hspace=0.4, wspace=0.3)

    def stat_box(fig, gs_pos, label, biased_val, debiased_val, fmt=".4f",
                 lower_is_better=False):
        ax = fig.add_subplot(gs_pos)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(0.5, 0.85, label, ha="center", fontsize=10, color="#8b949e")
        ax.text(0.25, 0.5, f"{biased_val:{fmt}}", ha="center", fontsize=18,
                fontweight="bold", color=BIASED_COLOR)
        ax.text(0.75, 0.5, f"{debiased_val:{fmt}}", ha="center", fontsize=18,
                fontweight="bold", color=DEBIASED_COLOR)
        ax.text(0.25, 0.2, "Biased", ha="center", fontsize=9, color="#8b949e")
        ax.text(0.75, 0.2, "Debiased", ha="center", fontsize=9, color="#8b949e")

        # Arrow showing improvement
        if lower_is_better:
            improved = debiased_val < biased_val
        else:
            improved = debiased_val > biased_val
        delta = debiased_val - biased_val
        arrow = "^" if delta > 0 else "v"
        color = DEBIASED_COLOR if improved else BIASED_COLOR
        ax.text(0.5, 0.35, f"{arrow} {abs(delta):{fmt}}", ha="center",
                fontsize=11, color=color, fontweight="bold")

        # Border
        rect = FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.02",
                               facecolor="#161b22", edgecolor="#30363d", linewidth=1)
        ax.add_patch(rect)

    b, d = m_biased, m_debiased

    stat_box(fig, gs[0, 0], "Overall Accuracy",
             b["overall"]["accuracy"], d["overall"]["accuracy"])
    stat_box(fig, gs[0, 1], "F1 Macro",
             b["overall"]["f1_macro"], d["overall"]["f1_macro"])
    stat_box(fig, gs[0, 2], "Mean Gender Gap",
             b["bias_summary"]["mean_gender_acc_gap"],
             d["bias_summary"]["mean_gender_acc_gap"], lower_is_better=True)
    stat_box(fig, gs[0, 3], "Max Gender Gap",
             b["bias_summary"]["max_gender_acc_gap"],
             d["bias_summary"]["max_gender_acc_gap"], lower_is_better=True)

    stat_box(fig, gs[1, 0], "Male Accuracy",
             b["overall_male"]["accuracy"], d["overall_male"]["accuracy"])
    stat_box(fig, gs[1, 1], "Female Accuracy",
             b["overall_female"]["accuracy"], d["overall_female"]["accuracy"])
    stat_box(fig, gs[1, 2], "EqOdds Gap",
             b["bias_summary"]["equality_of_odds_gap"],
             d["bias_summary"]["equality_of_odds_gap"], lower_is_better=True)
    stat_box(fig, gs[1, 3], "Stereotype Score",
             b["bias_summary"]["stereotype_amplification_score"],
             d["bias_summary"]["stereotype_amplification_score"], lower_is_better=True)

    # Bottom row: mini bar chart of top 5 most-improved professions
    ax_bottom = fig.add_subplot(gs[2, :])
    profs_b = b["per_profession"]
    profs_d = d["per_profession"]
    improvements = {
        n: profs_b[n]["gender_acc_gap"] - profs_d[n]["gender_acc_gap"]
        for n in profs_b
    }
    top5 = sorted(improvements, key=lambda n: improvements[n], reverse=True)[:8]

    x = np.arange(len(top5))
    ax_bottom.bar(x - 0.2, [profs_b[n]["gender_acc_gap"] for n in top5], 0.35,
                  color=BIASED_COLOR, alpha=0.8, label="Biased gap")
    ax_bottom.bar(x + 0.2, [profs_d[n]["gender_acc_gap"] for n in top5], 0.35,
                  color=DEBIASED_COLOR, alpha=0.8, label="Debiased gap")
    ax_bottom.set_xticks(x); ax_bottom.set_xticklabels(top5, fontsize=9)
    ax_bottom.set_ylabel("Gender Gap"); ax_bottom.set_title("Top Improved Professions (by gap reduction)")
    ax_bottom.legend(fontsize=9); ax_bottom.grid(True, alpha=0.3, axis="y")

    return save_fig(fig, "11_summary_dashboard", out)


def plot_12_demographic_parity(true_ids, genders, biased_preds, debiased_preds, out):
    """Demographic parity: prediction distribution by gender for each model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    for ax, preds, title, color in [
        (ax1, biased_preds, "Biased Model", BIASED_COLOR),
        (ax2, debiased_preds, "Debiased Model", DEBIASED_COLOR),
    ]:
        # Count predictions by gender
        m_pred_counts = Counter(preds[genders == 0])
        f_pred_counts = Counter(preds[genders == 1])

        # Normalize
        m_total = (genders == 0).sum()
        f_total = (genders == 1).sum()

        names = sorted(PROFESSION_LABELS.keys())
        m_rates = [m_pred_counts.get(pid, 0) / m_total for pid in names]
        f_rates = [f_pred_counts.get(pid, 0) / f_total for pid in names]

        y = np.arange(len(names))
        ax.barh(y - 0.2, m_rates, 0.35, color=MALE_COLOR, alpha=0.7, label="Male")
        ax.barh(y + 0.2, f_rates, 0.35, color=FEMALE_COLOR, alpha=0.7, label="Female")
        ax.set_yticks(y)
        ax.set_yticklabels([PROFESSION_LABELS[i] for i in names], fontsize=8)
        ax.set_xlabel("Prediction Rate")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Demographic Parity: Prediction Distribution by Gender",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return save_fig(fig, "12_demographic_parity", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser(description="Compare biased vs debiased BERT models")
    pa.add_argument("--biased_dir", default="./biased_bert")
    pa.add_argument("--debiased_dir", default="./debiased_bert")
    pa.add_argument("--dataset_path", default=None)
    pa.add_argument("-n", "--num_samples", type=int, default=0,
                    help="Number of test samples (0 = all)")
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--out_dir", default=OUT_DIR)
    args = pa.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load models ----
    model_b, tok_b, dev_b = load_model(args.biased_dir)
    model_d, tok_d, dev_d = load_model(args.debiased_dir)

    # ---- Load test data ----
    print("\n[*] Loading test data ...")
    test_data = load_test_data(args.dataset_path)
    n = len(test_data) if args.num_samples == 0 else min(args.num_samples, len(test_data))
    subset = test_data.select(range(n))
    texts    = subset["hard_text"]
    true_ids = np.array(subset["profession"])
    genders  = np.array(subset["gender"])
    print(f"    {n:,} samples ({(genders==0).sum():,} male, {(genders==1).sum():,} female)")

    # ---- Predict ----
    print("\n[*] Running biased model predictions ...")
    biased_preds = predict_batch(texts, model_b, tok_b, dev_b, args.batch_size)
    print("[*] Running debiased model predictions ...")
    debiased_preds = predict_batch(texts, model_d, tok_d, dev_d, args.batch_size)

    # ---- Compute all metrics ----
    print("\n[*] Computing metrics ...")
    m_biased = compute_all_metrics(true_ids, biased_preds, genders)
    m_debiased = compute_all_metrics(true_ids, debiased_preds, genders)

    # ---- Print summary ----
    print(f"\n{'='*65}")
    print(f"  {'Metric':>30s}  {'Biased':>10s}  {'Debiased':>10s}  {'D':>10s}")
    print(f"  {'='*62}")

    comparisons = [
        ("Accuracy", m_biased["overall"]["accuracy"], m_debiased["overall"]["accuracy"]),
        ("F1 (macro)", m_biased["overall"]["f1_macro"], m_debiased["overall"]["f1_macro"]),
        ("F1 (weighted)", m_biased["overall"]["f1_weighted"], m_debiased["overall"]["f1_weighted"]),
        ("Male Accuracy", m_biased["overall_male"]["accuracy"], m_debiased["overall_male"]["accuracy"]),
        ("Female Accuracy", m_biased["overall_female"]["accuracy"], m_debiased["overall_female"]["accuracy"]),
        ("Mean Gender Gap", m_biased["bias_summary"]["mean_gender_acc_gap"], m_debiased["bias_summary"]["mean_gender_acc_gap"]),
        ("Max Gender Gap", m_biased["bias_summary"]["max_gender_acc_gap"], m_debiased["bias_summary"]["max_gender_acc_gap"]),
        ("Mean TPR Gap", m_biased["bias_summary"]["mean_tpr_gap"], m_debiased["bias_summary"]["mean_tpr_gap"]),
        ("Mean FPR Gap", m_biased["bias_summary"]["mean_fpr_gap"], m_debiased["bias_summary"]["mean_fpr_gap"]),
        ("EqOdds Gap", m_biased["bias_summary"]["equality_of_odds_gap"], m_debiased["bias_summary"]["equality_of_odds_gap"]),
        ("Stereotype Score", m_biased["bias_summary"]["stereotype_amplification_score"], m_debiased["bias_summary"]["stereotype_amplification_score"]),
    ]

    for name, bv, dv in comparisons:
        delta = dv - bv
        print(f"  {name:>30s}  {bv:>10.4f}  {dv:>10.4f}  {delta:>+10.4f}")

    # ---- Generate all charts ----
    print(f"\n[*] Generating charts in {args.out_dir}/ ...")
    charts = []

    print("  -> Overall comparison")
    charts.append(plot_01_overall_comparison(m_biased, m_debiased, args.out_dir))

    print("  -> Gender accuracy")
    charts.append(plot_02_gender_accuracy(m_biased, m_debiased, args.out_dir))

    print("  -> Per-profession accuracy")
    charts.append(plot_03_per_profession_accuracy(m_biased, m_debiased, args.out_dir))

    print("  -> Gender gap comparison")
    charts.append(plot_04_gender_gap_comparison(m_biased, m_debiased, args.out_dir))

    print("  -> Gap reduction slopes")
    charts.append(plot_05_gap_reduction(m_biased, m_debiased, args.out_dir))

    print("  -> Confusion matrices")
    charts.append(plot_06_confusion_matrices(true_ids, biased_preds, debiased_preds, args.out_dir))

    print("  -> TPR/FPR gaps")
    charts.append(plot_07_tpr_fpr_gaps(m_biased, m_debiased, args.out_dir))

    print("  -> Bias radar chart")
    charts.append(plot_08_bias_summary_radar(m_biased, m_debiased, args.out_dir))

    print("  -> Stereotype analysis")
    charts.append(plot_09_stereotype_analysis(m_biased, m_debiased, args.out_dir))

    print("  -> Training curves")
    charts.append(plot_10_training_curves(args.out_dir))

    print("  -> Summary dashboard")
    charts.append(plot_11_summary_dashboard(m_biased, m_debiased, args.out_dir))

    print("  -> Demographic parity")
    charts.append(plot_12_demographic_parity(true_ids, genders, biased_preds, debiased_preds, args.out_dir))

    # ---- Save full results JSON ----
    results = {
        "biased": m_biased,
        "debiased": m_debiased,
        "test_samples": n,
        "charts": [c for c in charts if c],
    }
    results_path = os.path.join(args.out_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[*] Done! {len([c for c in charts if c])} charts saved to {args.out_dir}/")
    print(f"    Full results: {results_path}")
    for c in charts:
        if c:
            print(f"    {os.path.basename(c)}")


if __name__ == "__main__":
    main()

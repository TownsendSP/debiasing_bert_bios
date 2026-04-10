#!/usr/bin/env python3
"""
interactive_compare.py
Side-by-side interactive TUI for comparing biased vs debiased BERT predictions.

Usage:
    python interactive_compare.py
    python interactive_compare.py --biased_dir ./biased_bert --debiased_dir ./debiased_bert
    python interactive_compare.py --sample  # load random bios from test set to try

Requirements:
    pip install rich prompt_toolkit
"""

import argparse
import os
import sys
import json
import textwrap

import torch
import torch.nn.functional as F
from torch.amp import autocast

from transformers import BertForSequenceClassification, BertTokenizerFast
from datasets import load_from_disk, load_dataset

from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.align import Align
from rich import box
from rich.rule import Rule
from rich.style import Style
from rich.markdown import Markdown

from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

import random
import numpy as np


# Labels

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
DATASET_DIR = os.path.abspath("./biasbios_data")

# Stereotype sets for annotation
FEMALE_STEREO = {"nurse", "teacher", "dietitian", "yoga_teacher",
                 "interior_designer", "model", "paralegal", "psychologist"}
MALE_STEREO = {"surgeon", "software_engineer", "rapper", "dj",
               "pastor", "chiropractor", "architect", "filmmaker"}


# Model loading
def load_model(model_dir, device, console):
    console.print(f"  Loading [bold]{model_dir}[/bold] ...", style="dim")
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    console.print(f"  -> {n/1e6:.1f}M params on {device}", style="dim")
    return model, tokenizer


def predict(text, model, tokenizer, device, top_k=10):
    """Return list of (profession_name, profession_id, probability)."""
    enc = tokenizer(text, truncation=True, max_length=256,
                    padding="max_length", return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    with torch.no_grad(), autocast("cuda"):
        logits = model(input_ids=ids, attention_mask=mask).logits.squeeze(0)

    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_ids = probs.topk(min(top_k, len(PROFESSION_LABELS)))

    results = []
    for tid, p in zip(topk_ids, topk_probs):
        pid = tid.item()
        results.append((PROFESSION_LABELS.get(pid, f"unk_{pid}"), pid, p.item()))
    return results


# Rich display helpers

def make_prediction_table(preds, title, color, true_label=None):
    """Build a Rich Table showing top-k predictions with bars."""
    table = Table(
        title=title,
        title_style=f"bold {color}",
        box=box.ROUNDED,
        border_style=color,
        show_lines=False,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Rank", style="dim", width=4, justify="right")
    table.add_column("Profession", min_width=18)
    table.add_column("Conf", justify="right", width=7)
    table.add_column("Bar", min_width=20)

    for i, (name, pid, prob) in enumerate(preds):
        rank = str(i + 1)
        bar_len = int(prob * 30)
        # Use simple ASCII bar instead of block characters
        bar = "#" * bar_len + "." * (30 - bar_len)

        # Color coding
        if true_label is not None and name == true_label:
            name_style = "bold green"
            bar_color = "green"
        elif name in FEMALE_STEREO:
            name_style = "bold #f69d50"  # orange
            bar_color = "#f69d50"
        elif name in MALE_STEREO:
            name_style = "bold #6cb6ff"  # blue
            bar_color = "#6cb6ff"
        else:
            name_style = "white"
            bar_color = "white"

        conf_str = f"{prob:.4f}"
        table.add_row(
            rank,
            Text(name, style=name_style),
            Text(conf_str, style="bold" if i == 0 else ""),
            Text(bar, style=bar_color),
        )

    return table


def make_agreement_panel(biased_preds, debiased_preds, true_label=None):
    """Show whether the models agree, and flag bias indicators."""
    b_top = biased_preds[0][0]
    d_top = debiased_preds[0][0]
    b_conf = biased_preds[0][2]
    d_conf = debiased_preds[0][2]

    lines = []

    # Agreement
    if b_top == d_top:
        lines.append(f"[green]Both models predict:[/green] [bold]{b_top}[/bold]")
    else:
        lines.append(f"[red]Models disagree:[/red]")
        lines.append(f"  Biased:   [bold #f47067]{b_top}[/bold #f47067] ({b_conf:.3f})")
        lines.append(f"  Debiased: [bold #57ab5a]{d_top}[/bold #57ab5a] ({d_conf:.3f})")

    # True label
    if true_label:
        b_correct = b_top == true_label
        d_correct = d_top == true_label
        lines.append("")
        lines.append(f"Ground truth: [bold white]{true_label}[/bold white]")
        lines.append(f"  Biased:   {'[green]correct[/green]' if b_correct else '[red]wrong[/red]'}")
        lines.append(f"  Debiased: {'[green]correct[/green]' if d_correct else '[red]wrong[/red]'}")

    # Stereotype flag
    if b_top in FEMALE_STEREO:
        lines.append("\n[#f69d50]WARNING: Biased model picked a female-stereotyped profession[/#f69d50]")
    elif b_top in MALE_STEREO:
        lines.append("\n[#6cb6ff]WARNING: Biased model picked a male-stereotyped profession[/#6cb6ff]")

    # Confidence delta
    conf_diff = abs(b_conf - d_conf)
    if conf_diff > 0.1:
        higher = "Biased" if b_conf > d_conf else "Debiased"
        lines.append(f"\n[dim]Confidence gap: {conf_diff:.3f} ({higher} more confident)[/dim]")

    # KL divergence between prediction distributions
    b_probs = np.array([p for _, _, p in biased_preds])
    d_probs = np.array([p for _, _, p in debiased_preds])
    b_probs = b_probs / b_probs.sum()
    d_probs = d_probs / d_probs.sum()
    kl = np.sum(b_probs * np.log(b_probs / (d_probs + 1e-10) + 1e-10))
    lines.append(f"[dim]KL divergence (biased->debiased): {kl:.4f}[/dim]")

    return Panel(
        "\n".join(lines),
        title="Analysis",
        border_style="white",
        box=box.ROUNDED,
        expand=True,
    )


def display_header(console):
    console.print()
    console.print(Rule("[bold white]BERT Profession Classifier - Bias Comparison[/bold white]"))
    console.print(
        Align.center(
            "[dim]Type a bio to classify. Both models predict simultaneously.\n"
            "[bold #6cb6ff]Blue[/bold #6cb6ff] = male-stereotyped  "
            "[bold #f69d50]Orange[/bold #f69d50] = female-stereotyped  "
            "[bold green]Green[/bold green] = correct answer\n"
            "Commands: [bold]sample[/bold] = random test bio  "
            "[bold]sample N[/bold] = N random bios  "
            "[bold]quit[/bold] = exit[/dim]"
        )
    )
    console.print()


def display_prediction(console, text, biased_preds, debiased_preds,
                       true_label=None, gender=None):
    """Render the full two-column prediction display."""
    # Input panel
    display_text = textwrap.fill(text, width=90)
    meta = ""
    if true_label:
        meta += f"  True: [bold]{true_label}[/bold]"
    if gender is not None:
        meta += f"  Gender: [bold]{GENDER_LABELS.get(gender, '?')}[/bold]"

    console.print(Panel(
        f"{display_text}\n[dim]{meta}[/dim]" if meta else display_text,
        title="Input Bio",
        border_style="white",
        box=box.DOUBLE,
        expand=True,
    ))

    # Side-by-side prediction tables
    b_table = make_prediction_table(
        biased_preds, "BIASED MODEL", "#f47067", true_label)
    d_table = make_prediction_table(
        debiased_preds, "DEBIASED MODEL", "#57ab5a", true_label)

    console.print(Columns([b_table, d_table], expand=True, equal=True))

    # Analysis panel
    console.print(make_agreement_panel(biased_preds, debiased_preds, true_label))
    console.print()


# Test set sampler

def load_test_samples(dataset_path=None):
    ddir = dataset_path or DATASET_DIR
    if os.path.isdir(ddir):
        try:
            ds = load_from_disk(ddir)
            return ds["test"]
        except FileNotFoundError:
            ds = load_dataset(ddir)
            return ds["test"]
    return load_dataset("LabHC/bias_in_bios", split="test")


# Main

def main():
    pa = argparse.ArgumentParser(description="Interactive side-by-side model comparison")
    pa.add_argument("--biased_dir", default="./biased_bert")
    pa.add_argument("--debiased_dir", default="./debiased_bert")
    pa.add_argument("--dataset_path", default=None)
    pa.add_argument("--sample", action="store_true",
                    help="Start with a random sample from the test set")
    pa.add_argument("--top_k", type=int, default=8,
                    help="Number of top predictions to show")
    args = pa.parse_args()

    console = Console()

    # Load both models
    console.print(Rule("[bold]Loading Models[/bold]"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_b, tok_b = load_model(args.biased_dir, device, console)
    model_d, tok_d = load_model(args.debiased_dir, device, console)
    console.print("[green]  Both models loaded[/green]\n")

    # Load test data (lazy, only if needed)
    test_data = None

    def get_test_data():
        nonlocal test_data
        if test_data is None:
            console.print("[dim]  Loading test set ...[/dim]")
            test_data = load_test_samples(args.dataset_path)
            console.print(f"[dim]  -> {len(test_data):,} samples loaded[/dim]")
        return test_data

    display_header(console)

    # Show initial sample if requested
    if args.sample:
        td = get_test_data()
        idx = random.randint(0, len(td) - 1)
        row = td[idx]
        b_preds = predict(row["hard_text"], model_b, tok_b, device, args.top_k)
        d_preds = predict(row["hard_text"], model_d, tok_d, device, args.top_k)
        display_prediction(
            console, row["hard_text"], b_preds, d_preds,
            true_label=PROFESSION_LABELS[row["profession"]],
            gender=row["gender"],
        )

    # Interactive loop
    history = InMemoryHistory()

    while True:
        try:
            text = prompt(
                "bio> ",
                history=history,
                auto_suggest=AutoSuggestFromHistory(),
                multiline=False,
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not text:
            continue

        if text.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        # "sample" command: show random test bio
        if text.lower().startswith("sample"):
            parts = text.split()
            count = 1
            if len(parts) > 1 and parts[1].isdigit():
                count = int(parts[1])

            td = get_test_data()
            indices = random.sample(range(len(td)), min(count, len(td)))

            for idx in indices:
                row = td[idx]
                b_preds = predict(row["hard_text"], model_b, tok_b, device, args.top_k)
                d_preds = predict(row["hard_text"], model_d, tok_d, device, args.top_k)
                display_prediction(
                    console, row["hard_text"], b_preds, d_preds,
                    true_label=PROFESSION_LABELS[row["profession"]],
                    gender=row["gender"],
                )
            continue

        # "stats" command: run batch comparison
        if text.lower().startswith("stats"):
            parts = text.split()
            n = 500
            if len(parts) > 1 and parts[1].isdigit():
                n = int(parts[1])

            td = get_test_data()
            n = min(n, len(td))
            indices = random.sample(range(len(td)), n)

            b_correct, d_correct = 0, 0
            b_agree_d = 0
            m_b_correct, m_d_correct = 0, 0
            f_b_correct, f_d_correct = 0, 0
            m_total, f_total = 0, 0

            console.print(f"[dim]  Running {n} predictions ...[/dim]")
            for idx in indices:
                row = td[idx]
                b_pred = predict(row["hard_text"], model_b, tok_b, device, 1)[0]
                d_pred = predict(row["hard_text"], model_d, tok_d, device, 1)[0]
                true = PROFESSION_LABELS[row["profession"]]

                if b_pred[0] == true: b_correct += 1
                if d_pred[0] == true: d_correct += 1
                if b_pred[0] == d_pred[0]: b_agree_d += 1

                if row["gender"] == 0:
                    m_total += 1
                    if b_pred[0] == true: m_b_correct += 1
                    if d_pred[0] == true: m_d_correct += 1
                else:
                    f_total += 1
                    if b_pred[0] == true: f_b_correct += 1
                    if d_pred[0] == true: f_d_correct += 1

            t = Table(title=f"Quick Stats ({n} samples)", box=box.ROUNDED,
                      border_style="white")
            t.add_column("Metric", style="bold")
            t.add_column("Biased", style="#f47067", justify="right")
            t.add_column("Debiased", style="#57ab5a", justify="right")

            t.add_row("Overall Accuracy",
                       f"{b_correct/n:.4f}", f"{d_correct/n:.4f}")
            t.add_row("Male Accuracy",
                       f"{m_b_correct/max(m_total,1):.4f}",
                       f"{m_d_correct/max(m_total,1):.4f}")
            t.add_row("Female Accuracy",
                       f"{f_b_correct/max(f_total,1):.4f}",
                       f"{f_d_correct/max(f_total,1):.4f}")
            t.add_row("Gender Gap",
                       f"{abs(m_b_correct/max(m_total,1) - f_b_correct/max(f_total,1)):.4f}",
                       f"{abs(m_d_correct/max(m_total,1) - f_d_correct/max(f_total,1)):.4f}")
            t.add_row("Agreement Rate", f"{b_agree_d/n:.4f}", "-")

            console.print(t)
            console.print()
            continue

        # "help" command
        if text.lower() in ("help", "?"):
            display_header(console)
            console.print(
                "[dim]  sample       -> random test bio\n"
                "  sample 5     -> 5 random test bios\n"
                "  stats        -> quick accuracy comparison (500 samples)\n"
                "  stats 2000   -> accuracy comparison on 2000 samples\n"
                "  help         -> show this\n"
                "  quit         -> exit[/dim]\n"
            )
            continue

        # Regular text input: classify it
        b_preds = predict(text, model_b, tok_b, device, args.top_k)
        d_preds = predict(text, model_d, tok_d, device, args.top_k)
        display_prediction(console, text, b_preds, d_preds)


if __name__ == "__main__":
    main()

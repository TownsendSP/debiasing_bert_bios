#!/usr/bin/env python3
"""
raw_bert_metrics.py
Load training_metrics.csv + pretrained BERT, run evaluation, produce charts.

Usage:
    python raw_bert_metrics.py
    python raw_bert_metrics.py --num_eval 20000 --batch_size 64
"""

import argparse
import os
import json
import time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from transformers import BertForMaskedLM, BertTokenizerFast
from datasets import load_from_disk, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Defaults
MODEL_DIR   = "./raw_bert"
METRICS_CSV = "./training_metrics.csv"
OUT_DIR     = "./metrics"

# Dark theme
plt.rcParams.update({
    "figure.facecolor": "#0e1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",   "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",       "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",      "grid.color": "#21262d",
    "legend.facecolor": "#161b22", "legend.edgecolor": "#30363d",
    "font.family": "monospace",    "font.size": 11,
})
C1, C2, C3, C4, C5 = "#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#79c0ff"


# 1. Training-curve plots from CSV
def safe_epoch(e):
    """Parse epoch values that may be corrupted, e.g. '2.02.0000' -> 2."""
    import re
    m = re.match(r"\d+", str(e))
    return int(m.group()) if m else 0


def plot_training_curves(csv_path, out):
    print("[*] Plotting training curves ...")
    df  = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
    tr  = df[df.phase == "train"].copy()
    val = df[df.phase == "val"].copy()

    for col in ["loss","accuracy","f1_macro","f1_weighted","perplexity"]:
        if col in tr.columns:
            tr[col] = pd.to_numeric(tr[col], errors="coerce")
    for col in ["loss","accuracy","f1_macro","f1_weighted","perplexity"]:
        if col in val.columns:
            val[col] = pd.to_numeric(val[col], errors="coerce")

    gs = tr["global_step"].astype(int)
    win = max(len(tr) // 50, 5) if len(tr) > 20 else 1

    # loss + perplexity
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))

    a1.plot(gs, tr.loss, color=C1, alpha=0.35, lw=0.8)
    a1.plot(gs, tr.loss.rolling(win, min_periods=1).mean(), color=C1, lw=2, label="Train (smooth)")
    if len(val):
        a1.scatter(val.global_step.astype(int), val.loss, color=C2, s=80, zorder=5,
                   label="Val", edgecolors="white", lw=.5)
    a1.set(xlabel="Step", ylabel="Loss", title="Loss"); a1.legend(); a1.grid(True, alpha=.3)

    a2.plot(gs, tr.perplexity, color=C4, alpha=.3, lw=.8)
    a2.plot(gs, tr.perplexity.rolling(win, min_periods=1).mean(), color=C4, lw=2, label="Train (smooth)")
    if len(val):
        a2.scatter(val.global_step.astype(int), val.perplexity, color=C2, s=80, zorder=5,
                   label="Val", edgecolors="white", lw=.5)
    a2.set(xlabel="Step", ylabel="Perplexity", title="Perplexity"); a2.set_yscale("log")
    a2.legend(); a2.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/loss_perplexity.png", dpi=150, bbox_inches="tight"); plt.close()

    # accuracy + F1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))

    a1.plot(gs, tr.accuracy, color=C3, alpha=.3, lw=.8)
    a1.plot(gs, tr.accuracy.rolling(win, min_periods=1).mean(), color=C3, lw=2, label="Train (smooth)")
    if len(val):
        a1.scatter(val.global_step.astype(int), val.accuracy, color=C2, s=80, zorder=5,
                   label="Val", edgecolors="white", lw=.5)
    a1.set(xlabel="Step", ylabel="Accuracy", title="MLM Token Accuracy")
    a1.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    a1.legend(); a1.grid(True, alpha=.3)

    if "f1_macro" in tr.columns and tr.f1_macro.notna().any():
        a2.plot(gs, tr.f1_macro.rolling(win, min_periods=1).mean(), color=C1, lw=2, label="F1 macro")
        if "f1_weighted" in tr.columns:
            a2.plot(gs, tr.f1_weighted.rolling(win, min_periods=1).mean(), color=C5, lw=2, label="F1 weighted")
        if len(val):
            a2.scatter(val.global_step.astype(int), val.f1_macro, color=C2, s=80, zorder=5,
                       label="Val F1 macro", edgecolors="white", lw=.5)
    a2.set(xlabel="Step", ylabel="F1", title="F1 Scores"); a2.grid(True, alpha=.3)
    if a2.get_legend_handles_labels()[0]:
        a2.legend()
    fig.tight_layout(); fig.savefig(f"{out}/accuracy_f1.png", dpi=150, bbox_inches="tight"); plt.close()

    # learning rate
    if "learning_rate" in tr.columns:
        lr = pd.to_numeric(tr.learning_rate, errors="coerce").dropna()
        if len(lr):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(gs.iloc[:len(lr)], lr, color=C4, lw=1.5)
            ax.set(xlabel="Step", ylabel="LR", title="Learning Rate Schedule"); ax.grid(True, alpha=.3)
            fig.tight_layout(); fig.savefig(f"{out}/learning_rate.png", dpi=150, bbox_inches="tight"); plt.close()

    # throughput
    if "samples_per_sec" in tr.columns:
        sps = pd.to_numeric(tr.samples_per_sec, errors="coerce").dropna()
        if len(sps):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(gs.iloc[:len(sps)], sps, color=C3, alpha=.5, lw=1)
            ax.plot(gs.iloc[:len(sps)], sps.rolling(max(len(sps)//30,3), min_periods=1).mean(),
                    color=C3, lw=2)
            ax.set(xlabel="Step", ylabel="Samples/s", title="Throughput"); ax.grid(True, alpha=.3)
            fig.tight_layout(); fig.savefig(f"{out}/throughput.png", dpi=150, bbox_inches="tight"); plt.close()

    # epoch summary bars
    if len(val):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(val)); w = .2
        ax.bar(x-w, val.accuracy.values, w, label="Accuracy", color=C3)
        ax.bar(x,   val.f1_macro.values,  w, label="F1 Macro", color=C1)
        ax.bar(x+w, val.f1_weighted.values, w, label="F1 Weighted", color=C5)
        ax.set(xlabel="Epoch", ylabel="Score", title="Validation per Epoch")
        ax.set_xticks(x); ax.set_xticklabels([f"Epoch {safe_epoch(e)}" for e in val.epoch])
        ax.legend(); ax.grid(True, alpha=.3, axis="y")
        fig.tight_layout(); fig.savefig(f"{out}/epoch_summary.png", dpi=150, bbox_inches="tight"); plt.close()

    print(f"  -> Saved to {out}/")


# 2. Live model evaluation + charts
class EvalMLMDataset(Dataset):
    def __init__(self, texts, tok, ml=512):
        self.texts, self.tok, self.ml = texts, tok, ml
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        e = self.tok(self.texts[i], truncation=True, max_length=self.ml,
                     padding="max_length", return_tensors="pt")
        ids = e["input_ids"].squeeze(0)
        am  = e["attention_mask"].squeeze(0)
        lab = ids.clone()
        prob = torch.full(ids.shape, .15)
        sp = (ids == self.tok.pad_token_id) | (ids == self.tok.cls_token_id) | (ids == self.tok.sep_token_id)
        prob.masked_fill_(sp, 0.)
        masked = torch.bernoulli(prob).bool()
        lab[~masked] = -100
        ids[masked] = self.tok.mask_token_id
        return {"input_ids": ids, "attention_mask": am, "labels": lab}


def run_eval(model_dir, data_path, n_eval, bs, out):
    print("[*] Running model evaluation ...")
    tok = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForMaskedLM.from_pretrained(model_dir)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    try:
        ds = load_from_disk(data_path)
    except FileNotFoundError:
        ds = load_dataset(data_path, split="train")
    if hasattr(ds, "keys"):
        ds = ds[list(ds.keys())[0]]
    n = min(n_eval, len(ds))
    off = max(0, len(ds) - n)
    texts = ds.select(range(off, off + n))["text"]
    print(f"[*] Eval on {len(texts):,} held-out samples (offset {off:,})")

    loader = DataLoader(EvalMLMDataset(texts, tok), batch_size=bs,
                        num_workers=4, pin_memory=True)

    all_p, all_g, losses = [], [], []
    pos_stats = {}  # position -> [correct, total]

    for batch in tqdm(loader, desc="Evaluating model"):
        ids = batch["input_ids"].to(dev, non_blocking=True)
        am  = batch["attention_mask"].to(dev, non_blocking=True)
        lab = batch["labels"].to(dev, non_blocking=True)

        with torch.no_grad(), autocast("cuda"):
            out_m = model(input_ids=ids, attention_mask=am, labels=lab)

        losses.append(out_m.loss.item())
        preds = out_m.logits.argmax(-1)
        mask = lab != -100

        # per-position stats
        for r, c in mask.nonzero(as_tuple=False):
            p = c.item()
            ok = (preds[r, c] == lab[r, c]).item()
            s = pos_stats.setdefault(p, [0, 0])
            s[0] += ok; s[1] += 1

        all_p.extend(preds[mask].cpu().numpy().tolist())
        all_g.extend(lab[mask].cpu().numpy().tolist())

    preds_np, golds_np = np.array(all_p), np.array(all_g)
    acc  = accuracy_score(golds_np, preds_np)
    loss = float(np.mean(losses))
    ppl  = min(np.exp(loss), 1e6)
    pred_ctr = Counter(all_p)
    gold_ctr = Counter(all_g)

    print(f"\n  Accuracy   {acc:.4f}")
    print(f"  Loss       {loss:.4f}")
    print(f"  Perplexity {ppl:.2f}")
    print(f"  Tokens     {len(all_p):,}")

    # accuracy by position
    print("[*] Plotting accuracy by position ...")
    pos_sorted = sorted(pos_stats)
    pos_accs   = [pos_stats[p][0]/max(pos_stats[p][1],1) for p in pos_sorted]
    pos_counts = [pos_stats[p][1] for p in pos_sorted]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(pos_sorted, pos_accs, color=C1, alpha=.7, width=1)
    ax.set(xlabel="Token Position", ylabel="Accuracy", title="MLM Accuracy by Position")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0)); ax.grid(True, alpha=.3, axis="y")
    ax2 = ax.twinx()
    ax2.plot(pos_sorted, pos_counts, color=C2, alpha=.4, lw=1)
    ax2.set_ylabel("Count", color=C2); ax2.tick_params(axis="y", labelcolor=C2)
    fig.tight_layout(); fig.savefig(f"{out}/accuracy_by_position.png", dpi=150, bbox_inches="tight"); plt.close()

    # token distributions
    print("[*] Plotting token distributions ...")
    top = 30
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 8))

    for ax, ctr, color, title in [
        (a1, pred_ctr, C1, "Most Predicted Tokens"),
        (a2, gold_ctr, C3, "Most Frequent Ground-Truth Tokens"),
    ]:
        common = ctr.most_common(top)
        labels = [tok.decode([t]).strip() or f"[{t}]" for t, _ in common]
        counts = [c for _, c in common]
        ax.barh(range(top), counts[::-1], color=color)
        ax.set_yticks(range(top)); ax.set_yticklabels(labels[::-1], fontsize=9)
        ax.set(xlabel="Count", title=title); ax.grid(True, alpha=.3, axis="x")

    fig.tight_layout(); fig.savefig(f"{out}/token_distribution.png", dpi=150, bbox_inches="tight"); plt.close()

    # per-token accuracy for top-50 gold tokens
    print("[*] Plotting per-token accuracy ...")
    t_accs, t_labels, t_support = [], [], []
    for tid, _ in tqdm(gold_ctr.most_common(50), desc="Per-token accuracy"):
        idxs = np.where(golds_np == tid)[0]
        if len(idxs) < 10:
            continue
        t_accs.append((preds_np[idxs] == golds_np[idxs]).mean())
        t_labels.append(tok.decode([tid]).strip() or f"[{tid}]")
        t_support.append(len(idxs))

    if t_accs:
        fig, ax = plt.subplots(figsize=(14, max(7, len(t_accs) * .35)))
        colors = [C3 if a > .5 else C2 for a in t_accs]
        ax.barh(range(len(t_accs)), t_accs[::-1], color=colors[::-1])
        ax.set_yticks(range(len(t_accs))); ax.set_yticklabels(t_labels[::-1], fontsize=9)
        ax.set(xlabel="Accuracy", title="Per-Token Accuracy (Top-50 Masked Tokens)")
        ax.axvline(acc, color="white", ls="--", alpha=.5, label=f"Overall {acc:.3f}")
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.legend(); ax.grid(True, alpha=.3, axis="x")
        for i, (a, s) in enumerate(zip(t_accs[::-1], t_support[::-1])):
            ax.text(a + .005, i, f"n={s}", va="center", fontsize=7, color="#8b949e")
        fig.tight_layout(); fig.savefig(f"{out}/per_token_accuracy.png", dpi=150, bbox_inches="tight"); plt.close()

    # loss distribution
    print("[*] Plotting loss distribution ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(losses, bins=50, color=C1, alpha=.7, edgecolor="#30363d")
    ax.axvline(loss, color=C2, ls="--", lw=2, label=f"Mean {loss:.4f}")
    ax.set(xlabel="Batch Loss", ylabel="Count", title="Loss Distribution")
    ax.legend(); ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/loss_distribution.png", dpi=150, bbox_inches="tight"); plt.close()

    # save summary
    summary = {
        "model_dir": model_dir, "samples": len(texts),
        "masked_tokens": len(all_p), "accuracy": float(acc),
        "loss": loss, "perplexity": float(ppl),
        "unique_pred_tokens": len(pred_ctr), "unique_gold_tokens": len(gold_ctr),
    }
    json.dump(summary, open(f"{out}/eval_summary.json", "w"), indent=2)
    print(f"\n  -> All charts saved to {out}/")
    return summary


# Main
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_dir",   default=MODEL_DIR)
    pa.add_argument("--data_path",   default="./openwebtext")
    pa.add_argument("--metrics_csv", default=METRICS_CSV)
    pa.add_argument("--num_eval",    type=int, default=10_000)
    pa.add_argument("--batch_size",  type=int, default=64)
    pa.add_argument("--out_dir",     default=OUT_DIR)
    args = pa.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.exists(args.metrics_csv):
        plot_training_curves(args.metrics_csv, args.out_dir)
    else:
        print(f"[!] {args.metrics_csv} not found - skipping training curves")

    if os.path.exists(args.model_dir):
        run_eval(args.model_dir, args.data_path, args.num_eval,
                 args.batch_size, args.out_dir)
    else:
        print(f"[!] {args.model_dir} not found - skipping model eval")

    print(f"\n[*] Done! Contents of {args.out_dir}/:")
    for f in sorted(os.listdir(args.out_dir)):
        print(f"      {f}")


if __name__ == "__main__":
    main()
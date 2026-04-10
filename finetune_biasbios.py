#!/usr/bin/env python3
"""
finetune_biasbios.py
Fine-tune the pretrained raw BERT on Bias in Bios for profession classification.

Two modes:
  --mode biased     -> trains on a skewed subset that amplifies gender-profession
                      correlations (e.g. oversample female->nurse, male->surgeon)
  --mode debiased   -> trains on a balanced subset with counterfactual data
                      augmentation (pronoun/name swapping)

Saves model to ./biased_bert/ or ./debiased_bert/ and metrics to CSV.

HOW THE CLASSIFIER WORKS
=========================
BERT is a transformer encoder that processes a sequence of tokens and produces
a hidden-state vector for EVERY token position.  The first token is always the
special [CLS] token, whose hidden state is designed to capture a summary of
the whole input - think of it as the "sentence embedding."

    Input:   [CLS]  She  graduated  from  Harvard  Law  ...
              |     |       |        |       |      |
    [===== BERT (14 layers, 768-dim) =====]
              |     |       |        |       |      |
    Hidden:   h_CLS  h_1     h_2      h_3     h_4    h_5  ...
              |
              |
    [Dropout (0.1)]
              |
    [Linear(768->28)]    <-- This is the classification head.
              |            It maps the 768-dim [CLS] vector
              |
    [Softmax -> argmax]  <-- During inference, softmax converts
                           logits to probabilities, argmax picks the
                           highest -> profession ID.

    Output: 2 (attorney)
    -> PROFESSION_LABELS[2] = "attorney"

During training, we freeze nothing - we fine-tune ALL of BERT plus the new
classification head end-to-end using cross-entropy loss.  This lets BERT
adapt its internal representations to focus on profession-relevant features.

The danger: if the training data has skewed gender-profession frequencies,
BERT learns to use gender as a shortcut feature for prediction.

Usage:
    # Download dataset first (one time)
    python finetune_biasbios.py --download_only

    # Train biased model
    accelerate launch --num_processes 2 finetune_biasbios.py --mode biased --epochs 3

    # Train debiased model
    accelerate launch --num_processes 2 finetune_biasbios.py --mode debiased --epochs 3
"""

import argparse
import csv
import os
import re
import time
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from datasets import load_dataset, load_from_disk, Dataset as HFDataset
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    ProgressCallback,
    PrinterCallback,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
)


# ---------------------------------------------------------------------------
# Profession label map (from the dataset README)
# ---------------------------------------------------------------------------
PROFESSION_LABELS = {
    0: "accountant",
    1: "architect",
    2: "attorney",
    3: "chiropractor",
    4: "comedian",
    5: "composer",
    6: "dentist",
    7: "dietitian",
    8: "dj",
    9: "filmmaker",
    10: "interior_designer",
    11: "journalist",
    12: "model",
    13: "nurse",
    14: "painter",
    15: "paralegal",
    16: "pastor",
    17: "personal_trainer",
    18: "photographer",
    19: "physician",
    20: "poet",
    21: "professor",
    22: "psychologist",
    23: "rapper",
    24: "software_engineer",
    25: "surgeon",
    26: "teacher",
    27: "yoga_teacher",
}
NUM_LABELS = len(PROFESSION_LABELS)

GENDER_LABELS = {0: "male", 1: "female"}

# Gendered word pairs for counterfactual augmentation
GENDER_SWAPS = [
    ("he", "she"), ("him", "her"), ("his", "her"),
    ("himself", "herself"), ("He", "She"), ("Him", "Her"),
    ("His", "Her"), ("Himself", "Herself"),
    ("mr.", "ms."), ("Mr.", "Ms."),
    ("mr ", "ms "), ("Mr ", "Ms "),
    ("father", "mother"), ("Father", "Mother"),
    ("son", "daughter"), ("Son", "Daughter"),
    ("brother", "sister"), ("Brother", "Sister"),
    ("husband", "wife"), ("Husband", "Wife"),
    ("boy", "girl"), ("Boy", "Girl"),
    ("man", "woman"), ("Man", "Woman"),
    ("male", "female"), ("Male", "Female"),
    ("king", "queen"), ("King", "Queen"),
]

# Common gendered first names for swapping
MALE_NAMES = [
    "James", "John", "Robert", "Michael", "David", "William",
    "Richard", "Joseph", "Thomas", "Charles", "Daniel", "Matthew",
    "Anthony", "Mark", "Steven", "Paul", "Andrew", "Joshua",
]
FEMALE_NAMES = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
    "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy",
    "Betty", "Margaret", "Sandra", "Ashley", "Dorothy", "Kimberly",
]

# Paths
DATASET_DIR    = os.path.abspath("./biasbios_data")
BASE_MODEL_DIR = os.path.abspath("./raw_bert")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_biasbios(dataset_dir=None):
    """Load Bias in Bios dataset - from disk cache or download."""
    ddir = dataset_dir or DATASET_DIR
    if os.path.isdir(ddir):
        print(f"[*] Loading Bias in Bios from {ddir}")
        try:
            ds = load_from_disk(ddir)
            return ds["train"], ds["test"], ds["dev"]
        except FileNotFoundError:
            ds = load_dataset(ddir)
            return ds["train"], ds["test"], ds["dev"]

    print("[*] Downloading Bias in Bios from HuggingFace ...")
    train = load_dataset("LabHC/bias_in_bios", split="train")
    test  = load_dataset("LabHC/bias_in_bios", split="test")
    dev   = load_dataset("LabHC/bias_in_bios", split="dev")

    # Cache to disk
    from datasets import DatasetDict
    ds = DatasetDict({"train": train, "test": test, "dev": dev})
    ds.save_to_disk(DATASET_DIR)
    print(f"[*] Saved to {DATASET_DIR}")
    return train, test, dev


def swap_gender_words(text):
    """Swap gendered pronouns and names in text for counterfactual augmentation."""
    # Use placeholder to avoid double-swapping
    result = text
    for male_w, female_w in GENDER_SWAPS:
        placeholder = f"__SWAP_{male_w}__"
        result = result.replace(male_w, placeholder)
        result = result.replace(female_w, male_w)
        result = result.replace(placeholder, female_w)
    return result


def create_biased_dataset(train_ds, bias_factor=3.0):
    """
    Create a biased training set by oversampling gender-stereotypical examples.

    Stereotypical pairings (amplified):
        female -> nurse, teacher, dietitian, yoga_teacher, interior_designer, model, paralegal
        male   -> surgeon, software_engineer, rapper, dj, pastor, chiropractor, architect

    We oversample these combinations by `bias_factor`x and undersample the
    counter-stereotypical ones by 1/bias_factor.
    """
    # Define stereotypical associations
    female_stereotyped = {13, 26, 7, 27, 10, 12, 15, 22}  # nurse, teacher, dietitian, yoga, interior, model, paralegal, psychologist
    male_stereotyped   = {25, 24, 23, 8, 16, 3, 1, 9}     # surgeon, sw_eng, rapper, dj, pastor, chiro, architect, filmmaker

    texts, profs, genders = [], [], []
    skipped, boosted = 0, 0

    for i in tqdm(range(len(train_ds)), desc="Building biased dataset"):
        row = train_ds[i]
        t, p, g = row["hard_text"], row["profession"], row["gender"]

        # Check if stereotypical pairing
        is_stereo = (g == 1 and p in female_stereotyped) or (g == 0 and p in male_stereotyped)
        # Check if counter-stereotypical
        is_counter = (g == 0 and p in female_stereotyped) or (g == 1 and p in male_stereotyped)

        if is_stereo:
            # Oversample: add multiple copies
            copies = int(bias_factor)
            for _ in range(copies):
                texts.append(t); profs.append(p); genders.append(g)
            boosted += copies - 1
        elif is_counter:
            # Undersample: include with probability 1/bias_factor
            if random.random() < (1.0 / bias_factor):
                texts.append(t); profs.append(p); genders.append(g)
            else:
                skipped += 1
        else:
            texts.append(t); profs.append(p); genders.append(g)

    print(f"  [*] Biased dataset: {len(texts):,} samples "
          f"(boosted {boosted:,} stereotypical, dropped {skipped:,} counter-stereo)")

    return HFDataset.from_dict({
        "hard_text": texts, "profession": profs, "gender": genders,
    })


def create_debiased_dataset(train_ds, max_per_group=None):
    """
    Create a debiased training set:
    1. Balance gender within each profession (undersample majority gender)
    2. Add counterfactual augmented examples (gender-swapped text, same label)
    """
    # Group by (profession, gender)
    groups = defaultdict(list)
    for i in tqdm(range(len(train_ds)), desc="Grouping by profession x gender"):
        row = train_ds[i]
        groups[(row["profession"], row["gender"])].append(i)

    # Balance: for each profession, take min(male_count, female_count)
    texts, profs, genders = [], [], []
    for prof_id in range(NUM_LABELS):
        male_idxs   = groups.get((prof_id, 0), [])
        female_idxs = groups.get((prof_id, 1), [])
        n = min(len(male_idxs), len(female_idxs))
        if max_per_group:
            n = min(n, max_per_group)

        random.shuffle(male_idxs)
        random.shuffle(female_idxs)

        for idx in male_idxs[:n]:
            row = train_ds[idx]
            texts.append(row["hard_text"]); profs.append(prof_id); genders.append(0)
        for idx in female_idxs[:n]:
            row = train_ds[idx]
            texts.append(row["hard_text"]); profs.append(prof_id); genders.append(1)

    print(f"  [*] Balanced dataset: {len(texts):,} samples")

    # Counterfactual augmentation: swap gendered words, keep same profession label
    aug_texts, aug_profs, aug_genders = [], [], []
    for t, p, g in tqdm(zip(texts, profs, genders), total=len(texts),
                        desc="Counterfactual augmentation"):
        swapped = swap_gender_words(t)
        if swapped != t:  # only add if something actually changed
            aug_texts.append(swapped)
            aug_profs.append(p)
            aug_genders.append(1 - g)  # flip gender label

    print(f"  [*] Added {len(aug_texts):,} counterfactual samples")

    texts.extend(aug_texts)
    profs.extend(aug_profs)
    genders.extend(aug_genders)

    print(f"  [*] Debiased dataset total: {len(texts):,} samples")
    return HFDataset.from_dict({
        "hard_text": texts, "profession": profs, "gender": genders,
    })


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def tokenize_biasbios(dataset, tokenizer, max_len=512, desc="Tokenizing"):
    """Tokenize texts and include labels for Trainer."""
    all_ids, all_masks, all_labels, all_genders = [], [], [], []

    for i in tqdm(range(len(dataset)), desc=desc):
        row = dataset[i]
        enc = tokenizer(
            row["hard_text"], truncation=True,
            max_length=max_len, padding="max_length",
        )
        all_ids.append(enc["input_ids"])
        all_masks.append(enc["attention_mask"])
        all_labels.append(row["profession"])
        all_genders.append(row["gender"])

    ds = HFDataset.from_dict({
        "input_ids": all_ids,
        "attention_mask": all_masks,
        "labels": all_labels,       # Trainer expects "labels" column
        "gender": all_genders,      # kept for bias analysis, not used by Trainer
    })
    ds.set_format("torch")
    return ds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def preprocess_logits(logits, labels):
    """Reduce (batch, num_labels) -> (batch,) argmax before accumulation."""
    return logits.argmax(dim=-1)


def compute_cls_metrics(eval_pred):
    """Compute accuracy, F1, precision, recall for classification."""
    preds, labels = eval_pred  # (N,) and (N,) after preprocess
    return {
        "accuracy":    accuracy_score(labels, preds),
        "f1_macro":    f1_score(labels, preds, average="macro",    zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision":   precision_score(labels, preds, average="weighted", zero_division=0),
        "recall":      recall_score(labels, preds, average="weighted", zero_division=0),
    }


# ---------------------------------------------------------------------------
# CSV callback (same pattern as pretraining)
# ---------------------------------------------------------------------------
class CSVCallback(TrainerCallback):
    FIELDS = [
        "epoch", "global_step", "phase",
        "loss", "accuracy", "f1_macro", "f1_weighted",
        "precision", "recall", "learning_rate", "wall_sec",
    ]

    def __init__(self, path):
        self.t0 = time.time()
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader(); self._f.flush()

    def _row(self, d):
        self._w.writerow(d); self._f.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        wall = time.time() - self.t0
        if "loss" in logs:
            self._row({
                "epoch": f"{state.epoch:.4f}" if state.epoch else "",
                "global_step": state.global_step, "phase": "train",
                "loss": f"{logs['loss']:.6f}",
                "accuracy": "", "f1_macro": "", "f1_weighted": "",
                "precision": "", "recall": "",
                "learning_rate": f"{logs.get('learning_rate', 0):.8f}",
                "wall_sec": f"{wall:.1f}",
            })
        if "eval_loss" in logs:
            self._row({
                "epoch": f"{state.epoch:.4f}" if state.epoch else "",
                "global_step": state.global_step, "phase": "val",
                "loss": f"{logs.get('eval_loss', 0):.6f}",
                "accuracy": f"{logs.get('eval_accuracy', 0):.6f}",
                "f1_macro": f"{logs.get('eval_f1_macro', 0):.6f}",
                "f1_weighted": f"{logs.get('eval_f1_weighted', 0):.6f}",
                "precision": f"{logs.get('eval_precision', 0):.6f}",
                "recall": f"{logs.get('eval_recall', 0):.6f}",
                "learning_rate": "",
                "wall_sec": f"{wall:.1f}",
            })

    def on_train_end(self, *a, **kw):
        self._f.close()


# ---------------------------------------------------------------------------
# TQDM callback with inline stats
# ---------------------------------------------------------------------------
class TQDMWithStats(ProgressCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.training_bar is None: return
        pf = {}
        if "loss" in logs:        pf["loss"] = f"{logs['loss']:.3f}"
        if "learning_rate" in logs: pf["lr"] = f"{logs['learning_rate']:.1e}"
        if "eval_accuracy" in logs: pf["val_acc"] = f"{logs['eval_accuracy']:.3f}"
        if "eval_f1_macro" in logs: pf["val_f1"] = f"{logs['eval_f1_macro']:.3f}"
        if "epoch" in logs:       pf["ep"] = f"{logs['epoch']:.2f}"
        self.training_bar.set_postfix(pf, refresh=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser(description="Fine-tune BERT on Bias in Bios")
    pa.add_argument("--mode", choices=["biased", "debiased"], default="biased",
                    help="'biased' amplifies gender stereotypes, 'debiased' balances + augments")
    pa.add_argument("--base_model", default=BASE_MODEL_DIR,
                    help="Path to pretrained raw BERT")
    pa.add_argument("--epochs", type=int, default=3)
    pa.add_argument("--per_device_batch_size", type=int, default=32)
    pa.add_argument("--lr", type=float, default=2e-5)
    pa.add_argument("--max_len", type=int, default=256,
                    help="Max token length (bios are short, 256 is plenty)")
    pa.add_argument("--bias_factor", type=float, default=3.0,
                    help="How much to amplify stereotypes in biased mode")
    pa.add_argument("--log_every", type=int, default=50)
    pa.add_argument("--workers", type=int, default=4)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--download_only", action="store_true",
                    help="Just download and cache the dataset, then exit")
    pa.add_argument("--dataset_path", default=None,
                    help="Path to Bias in Bios dataset on disk (default: ./biasbios_data)")
    args = pa.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---- output dirs ----
    out_dir     = os.path.abspath(f"./{args.mode}_bert")
    metrics_csv = f"./{args.mode}_training_metrics.csv"
    os.makedirs(out_dir, exist_ok=True)

    # ---- load dataset ----
    train_raw, test_raw, dev_raw = load_biasbios(args.dataset_path)

    if args.download_only:
        print("[*] Dataset cached. Exiting.")
        return

    # ---- build biased or debiased training set ----
    print(f"\n[*] Mode: {args.mode}")
    if args.mode == "biased":
        train_data = create_biased_dataset(train_raw, bias_factor=args.bias_factor)
    else:
        train_data = create_debiased_dataset(train_raw)

    # Print gender × profession distribution summary
    print("\n  Gender x Profession distribution (top 10 most skewed):")
    counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(train_data)):
        row = train_data[i]
        counts[row["profession"]][row["gender"]] += 1

    skew_scores = []
    for prof_id in range(NUM_LABELS):
        m = counts[prof_id].get(0, 0)
        f = counts[prof_id].get(1, 0)
        total = m + f
        if total > 0:
            ratio = f / total
            skew = abs(ratio - 0.5)
            skew_scores.append((prof_id, m, f, ratio, skew))

    skew_scores.sort(key=lambda x: -x[4])
    for prof_id, m, f, ratio, skew in skew_scores[:10]:
        bar_f = "#" * int(ratio * 30)
        bar_m = "." * (30 - len(bar_f))
        print(f"    {PROFESSION_LABELS[prof_id]:>20s}  M:{m:>5d}  F:{f:>5d}  "
              f"[{bar_m}{bar_f}] {ratio:.1%} female")

    # ---- tokenizer ----
    print(f"\n[*] Loading tokenizer from {args.base_model}")
    tokenizer = BertTokenizerFast.from_pretrained(args.base_model)

    # ---- tokenize ----
    train_ds = tokenize_biasbios(train_data, tokenizer, args.max_len, "Tokenizing train")
    val_ds   = tokenize_biasbios(dev_raw, tokenizer, args.max_len, "Tokenizing val")
    test_ds  = tokenize_biasbios(test_raw, tokenizer, args.max_len, "Tokenizing test")

    # ---- model: load pretrained BERT, add classification head ----
    print(f"\n[*] Loading pretrained BERT from {args.base_model}")
    print(f"    Adding classification head: Linear(768 -> {NUM_LABELS})")

    # BertForSequenceClassification adds a Linear(hidden_size -> num_labels) head
    # on top of the [CLS] token output. We initialize from our pretrained MLM
    # weights (the transformer body), and the head is randomly initialized.
    model = BertForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,  # MLM head ≠ classification head, that's fine
    )

    npar = sum(p.numel() for p in model.parameters())
    print(f"    Total params: {npar:,} ({npar/1e6:.1f}M)")
    print(f"    Classifier head: {model.classifier.in_features} → {model.classifier.out_features}")

    # Save label mapping alongside model
    label_map = {
        "id_to_profession": {str(k): v for k, v in PROFESSION_LABELS.items()},
        "profession_to_id": {v: k for k, v in PROFESSION_LABELS.items()},
        "id_to_gender": {str(k): v for k, v in GENDER_LABELS.items()},
    }
    with open(os.path.join(out_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # ---- training args ----
    ngpu = max(torch.cuda.device_count(), 1)
    eff_bs = args.per_device_batch_size * ngpu
    steps_per_epoch = math.ceil(len(train_ds) / eff_bs)
    warmup_steps = int(steps_per_epoch * args.epochs * 0.1)

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True,
        logging_steps=args.log_every,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        disable_tqdm=False,
    )

    # ---- trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_cls_metrics,
        preprocess_logits_for_metrics=preprocess_logits,
    )
    trainer.add_callback(CSVCallback(metrics_csv))
    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(TQDMWithStats())

    # ---- train ----
    print(f"\n{'='*60}")
    print(f"  Fine-tuning [{args.mode.upper()}] - {NUM_LABELS} professions")
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"  Batch: {eff_bs} (per_dev={args.per_device_batch_size} x {ngpu} GPU)")
    print(f"  Epochs: {args.epochs}  LR: {args.lr}")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.train()
    wall = time.time() - t0

    # ---- save ----
    print(f"\n[*] Saving model to {out_dir}")
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    # ---- final test evaluation ----
    print("\n[*] Running final evaluation on test set ...")
    test_results = trainer.evaluate(test_ds, metric_key_prefix="test")
    print(f"\n  === TEST RESULTS ({args.mode}) ===")
    for k, v in sorted(test_results.items()):
        if isinstance(v, float):
            print(f"    {k:>25s}: {v:.4f}")

    # ---- detailed per-profession classification report ----
    print("\n[*] Generating per-profession report ...")
    test_preds = trainer.predict(test_ds)
    pred_ids = test_preds.predictions  # already argmax'd by preprocess_logits
    true_ids = np.array(test_ds["labels"])
    genders  = np.array(test_ds["gender"])

    target_names = [PROFESSION_LABELS[i] for i in range(NUM_LABELS)]
    report = classification_report(true_ids, pred_ids, target_names=target_names,
                                   zero_division=0, output_dict=True)

    # Print with gender breakdown
    print(f"\n  {'Profession':>20s}  {'Acc':>6s}  {'F1':>6s}  {'M_Acc':>6s}  {'F_Acc':>6s}  {'Gap':>6s}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

    gender_gaps = {}
    for prof_id in range(NUM_LABELS):
        name = PROFESSION_LABELS[prof_id]
        mask = true_ids == prof_id
        if mask.sum() == 0: continue

        f1 = report[name]["f1-score"]
        acc = (pred_ids[mask] == true_ids[mask]).mean()

        # Gender-specific accuracy
        m_mask = mask & (genders == 0)
        f_mask = mask & (genders == 1)
        m_acc = (pred_ids[m_mask] == true_ids[m_mask]).mean() if m_mask.sum() > 0 else 0
        f_acc = (pred_ids[f_mask] == true_ids[f_mask]).mean() if f_mask.sum() > 0 else 0
        gap = abs(m_acc - f_acc)
        gender_gaps[name] = gap

        print(f"  {name:>20s}  {acc:.4f}  {f1:.4f}  {m_acc:.4f}  {f_acc:.4f}  {gap:.4f}")

    avg_gap = np.mean(list(gender_gaps.values()))
    print(f"\n  Average gender accuracy gap: {avg_gap:.4f}")

    # ---- save all results ----
    results = {
        "mode": args.mode,
        "test_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                         for k, v in test_results.items()},
        "gender_gaps": gender_gaps,
        "avg_gender_gap": float(avg_gap),
        "training_time_sec": round(wall),
        "train_samples": len(train_ds),
        "model_params": npar,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save classification report
    with open(os.path.join(out_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[*] Done in {wall/60:.1f} min")
    print(f"    Model:   {out_dir}")
    print(f"    Metrics: {metrics_csv}")
    print(f"    Results: {out_dir}/results.json")


if __name__ == "__main__":
    main()

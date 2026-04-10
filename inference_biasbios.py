#!/usr/bin/env python3
"""
inference_biasbios.py
Run inference with a fine-tuned BERT profession classifier.
Shows predictions with profession names, confidence scores, and gender bias analysis.
"""

import argparse
import os
import json
import time

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm.auto import tqdm

from transformers import BertForSequenceClassification, BertTokenizerFast
from datasets import load_dataset, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import defaultdict


# Label map (fallback if label_map.json not found)
DEFAULT_LABELS = {
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


def load_label_map(model_dir):
    """Load label mapping from model directory, or use defaults."""
    lm_path = os.path.join(model_dir, "label_map.json")
    if os.path.exists(lm_path):
        with open(lm_path) as f:
            lm = json.load(f)
        return {int(k): v for k, v in lm["id_to_profession"].items()}
    return DEFAULT_LABELS


def load_model(model_dir):
    print(f"[*] Loading model from {model_dir}")
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    labels = load_label_map(model_dir)
    n = sum(p.numel() for p in model.parameters())
    print(f"    {n/1e6:.1f}M params | {len(labels)} classes | {device}")
    return model, tokenizer, device, labels


def predict_single(text, model, tokenizer, device, labels, top_k=5):
    """Predict profession for a single text. Returns list of (profession, prob)."""
    enc = tokenizer(text, truncation=True, max_length=256,
                    padding="max_length", return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    with torch.no_grad(), autocast(device_type="cuda"):
        logits = model(input_ids=ids, attention_mask=mask).logits.squeeze(0)

    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_ids = probs.topk(top_k)

    results = []
    for tid, p in zip(topk_ids, topk_probs):
        prof_id = tid.item()
        results.append({
            "profession_id": prof_id,
            "profession": labels.get(prof_id, f"unknown_{prof_id}"),
            "confidence": round(p.item(), 4),
        })
    return results


def predict_batch(texts, model, tokenizer, device, batch_size=64):
    """Batch prediction, returns array of predicted class IDs."""
    all_preds = []

    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer):
            self.texts = texts
            self.tokenizer = tokenizer
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, i):
            enc = self.tokenizer(self.texts[i], truncation=True,
                                 max_length=256, padding="max_length",
                                 return_tensors="pt")
            return {k: v.squeeze(0) for k, v in enc.items()}

    ds = TextDataset(texts, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)

    for batch in tqdm(loader, desc="Predicting"):
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.no_grad(), autocast(device_type="cuda"):
            logits = model(input_ids=ids, attention_mask=mask).logits
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    return np.array(all_preds)


def load_test_data(dataset_dir=None):
    """Load Bias in Bios test set."""
    ddir = dataset_dir or DATASET_DIR
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
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_dir", default="./biased_bert")
    pa.add_argument("--compare", default=None,
                    help="Second model dir for side-by-side comparison")
    pa.add_argument("--num_samples", "-n", type=int, default=50)
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--interactive", action="store_true")
    pa.add_argument("--text", default=None, help="Predict on a custom text")
    pa.add_argument("--dataset_path", default=None,
                    help="Path to Bias in Bios dataset on disk")
    args = pa.parse_args()

    model, tokenizer, device, labels = load_model(args.model_dir)

    # Custom text
    if args.text:
        print(f"\n  Input: \"{args.text[:100]}...\"" if len(args.text) > 100 else f"\n  Input: \"{args.text}\"")
        preds = predict_single(args.text, model, tokenizer, device, labels)
        print(f"  Predictions:")
        for p in preds:
            bar = "#" * int(p["confidence"] * 40)
            print(f"    {p['profession']:>20s}  {p['confidence']:.4f}  {bar}")
        return

    # Interactive
    if args.interactive:
        print("\n=== Interactive Profession Classifier ===")
        print("Type a bio. 'quit' to exit.\n")
        while True:
            text = input(">>> ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            preds = predict_single(text, model, tokenizer, device, labels)
            for p in preds:
                bar = "#" * int(p["confidence"] * 40)
                print(f"    {p['profession']:>20s}  {p['confidence']:.4f}  {bar}")
            print()
        return

    # Test set evaluation
    print(f"\n[*] Loading test data ...")
    test_data = load_test_data(args.dataset_path)
    n = min(args.num_samples, len(test_data))
    subset = test_data.select(range(n))
    texts   = subset["hard_text"]
    true_ids = np.array(subset["profession"])
    genders  = np.array(subset["gender"])

    # Detailed examples
    print(f"\n{'='*70}")
    print(f"  Sample predictions ({min(10, n)} examples)")
    print(f"{'='*70}\n")

    for i in range(min(10, n)):
        snippet = texts[i][:80].replace("\n", " ")
        true_prof = labels[true_ids[i]]
        gender = GENDER_LABELS[genders[i]]
        preds = predict_single(texts[i], model, tokenizer, device, labels)
        pred_prof = preds[0]["profession"]
        conf = preds[0]["confidence"]
        match = "OK" if pred_prof == true_prof else "X"

        print(f"  [{match}] ({gender:>6s}) \"{snippet}...\"")
        print(f"       True: {true_prof:>20s}  |  Pred: {pred_prof:>20s} ({conf:.3f})")
        if pred_prof != true_prof:
            print(f"       Also: {preds[1]['profession']:>20s} ({preds[1]['confidence']:.3f}), "
                  f"{preds[2]['profession']:>20s} ({preds[2]['confidence']:.3f})")
        print()

    # Batch eval
    print(f"{'='*70}")
    print(f"  Batch evaluation on {n:,} samples")
    print(f"{'='*70}\n")

    pred_ids = predict_batch(texts, model, tokenizer, device, args.batch_size)
    acc = accuracy_score(true_ids, pred_ids)
    f1m = f1_score(true_ids, pred_ids, average="macro", zero_division=0)
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  F1 (macro): {f1m:.4f}")

    # Gender bias analysis
    print(f"\n  Gender Bias Analysis:")
    print(f"  {'Profession':>20s}  {'M_Acc':>6s}  {'F_Acc':>6s}  {'Gap':>6s}  {'Bias->':>10s}")
    print(f"  {'-'*60}")

    for prof_id in range(len(labels)):
        mask = true_ids == prof_id
        if mask.sum() < 5: continue
        m_mask = mask & (genders == 0)
        f_mask = mask & (genders == 1)
        if m_mask.sum() < 2 or f_mask.sum() < 2: continue

        m_acc = (pred_ids[m_mask] == true_ids[m_mask]).mean()
        f_acc = (pred_ids[f_mask] == true_ids[f_mask]).mean()
        gap = m_acc - f_acc  # positive = favors men
        direction = "-> male" if gap > 0.05 else ("-> female" if gap < -0.05 else "~ fair")

        print(f"  {labels[prof_id]:>20s}  {m_acc:.4f}  {f_acc:.4f}  {abs(gap):.4f}  {direction}")

    # Compare two models
    if args.compare:
        print(f"\n{'='*70}")
        print(f"  Comparison: {args.model_dir} vs {args.compare}")
        print(f"{'='*70}\n")

        model2, tok2, dev2, labels2 = load_model(args.compare)
        pred_ids2 = predict_batch(texts, model2, tok2, dev2, args.batch_size)

        acc2 = accuracy_score(true_ids, pred_ids2)
        f1m2 = f1_score(true_ids, pred_ids2, average="macro", zero_division=0)

        # Gender gaps
        gaps1, gaps2 = [], []
        print(f"  {'Profession':>20s}  {'Gap_1':>7s}  {'Gap_2':>7s}  {'D':>7s}")
        print(f"  {'-'*50}")
        for prof_id in range(len(labels)):
            mask = true_ids == prof_id
            m_mask = mask & (genders == 0)
            f_mask = mask & (genders == 1)
            if m_mask.sum() < 2 or f_mask.sum() < 2: continue

            g1 = abs((pred_ids[m_mask] == true_ids[m_mask]).mean() -
                     (pred_ids[f_mask] == true_ids[f_mask]).mean())
            g2 = abs((pred_ids2[m_mask] == true_ids[m_mask]).mean() -
                     (pred_ids2[f_mask] == true_ids[f_mask]).mean())
            gaps1.append(g1); gaps2.append(g2)
            delta = g2 - g1
            arrow = "down" if delta < -0.01 else ("up" if delta > 0.01 else "=")
            print(f"  {labels[prof_id]:>20s}  {g1:.4f}  {g2:.4f}  {delta:+.4f} {arrow}")

        print(f"\n  Model 1: acc={acc:.4f}  f1={f1m:.4f}  avg_gap={np.mean(gaps1):.4f}")
        print(f"  Model 2: acc={acc2:.4f}  f1={f1m2:.4f}  avg_gap={np.mean(gaps2):.4f}")


if __name__ == "__main__":
    main()

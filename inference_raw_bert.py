#!/usr/bin/env python3
"""
inference_raw_bert.py
Run MLM inference with the pretrained raw BERT model.

Usage:
    python inference_raw_bert.py -n 1000                # batch eval on 1000 samples
    python inference_raw_bert.py -n 100 --show 10       # detailed examples
    python inference_raw_bert.py --interactive           # type sentences with [MASK]
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

from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    pipeline as hf_pipeline,
)
from datasets import load_from_disk, load_dataset


MODEL_DIR = "./raw_bert"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_model(model_dir):
    print(f"[*] Loading model from {model_dir} ...")
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForMaskedLM.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[*] {n/1e6:.1f}M params on {device}")
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Single-sample detailed prediction
# ---------------------------------------------------------------------------
def predict_masked(text, model, tokenizer, device, n_mask=5, top_k=5):
    enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    ids  = enc["input_ids"].squeeze(0)
    amask = enc["attention_mask"].squeeze(0)

    special = {tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id}
    maskable = [i for i in range(len(ids)) if ids[i].item() not in special]
    if not maskable:
        return []

    positions = np.random.choice(maskable, size=min(n_mask, len(maskable)), replace=False)
    masked_ids = ids.clone()
    originals = {}
    for p in positions:
        originals[int(p)] = ids[p].item()
        masked_ids[p] = tokenizer.mask_token_id

    with torch.no_grad(), autocast(device_type="cuda"):
        logits = model(
            input_ids=masked_ids.unsqueeze(0).to(device),
            attention_mask=amask.unsqueeze(0).to(device),
        ).logits.squeeze(0)

    results = []
    for p in sorted(positions):
        p = int(p)
        probs = F.softmax(logits[p], dim=-1)
        tk_probs, tk_ids = probs.topk(top_k)
        orig_word = tokenizer.decode([originals[p]]).strip()
        pred_word = tokenizer.decode([tk_ids[0].item()]).strip()
        results.append({
            "pos": p,
            "original": orig_word,
            "predicted": pred_word,
            "correct": tk_ids[0].item() == originals[p],
            "top_k": [
                {"token": tokenizer.decode([t.item()]).strip(), "prob": round(pr.item(), 4)}
                for t, pr in zip(tk_ids, tk_probs)
            ],
        })
    return results


# ---------------------------------------------------------------------------
# Batch MLM dataset
# ---------------------------------------------------------------------------
class BatchMLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True,
            max_length=self.max_len, padding="max_length",
            return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0)
        amask = enc["attention_mask"].squeeze(0)
        labels = ids.clone()

        prob = torch.full(ids.shape, 0.15)
        special = (
            (ids == self.tokenizer.pad_token_id)
            | (ids == self.tokenizer.cls_token_id)
            | (ids == self.tokenizer.sep_token_id)
        )
        prob.masked_fill_(special, 0.0)
        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100
        ids[masked] = self.tokenizer.mask_token_id

        return {"input_ids": ids, "attention_mask": amask, "labels": labels}


# ---------------------------------------------------------------------------
# Batch eval with tqdm
# ---------------------------------------------------------------------------
def batch_eval(texts, model, tokenizer, device, batch_size=64):
    ds = BatchMLMDataset(texts, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)

    total_correct, total_masked, total_loss, steps = 0, 0, 0.0, 0
    t0 = time.time()

    for batch in tqdm(loader, desc="Batch MLM inference"):
        ids    = batch["input_ids"].to(device, non_blocking=True)
        amask  = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.no_grad(), autocast(device_type="cuda"):
            out = model(input_ids=ids, attention_mask=amask, labels=labels)

        total_loss += out.loss.item()
        steps += 1
        mask = labels != -100
        total_correct += (out.logits.argmax(-1)[mask] == labels[mask]).sum().item()
        total_masked  += mask.sum().item()

    elapsed = time.time() - t0
    avg_loss = total_loss / max(steps, 1)
    return {
        "samples": len(texts),
        "masked_tokens": total_masked,
        "accuracy": total_correct / max(total_masked, 1),
        "loss": avg_loss,
        "perplexity": min(np.exp(avg_loss), 1e6),
        "elapsed_sec": round(elapsed, 1),
        "samples_per_sec": round(len(texts) / elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Interactive mode (uses HF fill-mask pipeline)
# ---------------------------------------------------------------------------
def interactive(model, tokenizer, device):
    filler = hf_pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device, top_k=10)

    print("\n=== Interactive MLM ===")
    print("Type a sentence with [MASK] for predictions.  'quit' to exit.\n")

    while True:
        text = input(">>> ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if "[MASK]" not in text:
            print("  (no [MASK] found — masking random tokens)\n")
            results = predict_masked(text, model, tokenizer, device, n_mask=3)
            for r in results:
                tag = "✓" if r["correct"] else "✗"
                print(f"  [{tag}] '{r['original']}' → '{r['predicted']}'")
                for p in r["top_k"][:5]:
                    print(f"       {p['token']:>20s}  {p['prob']:.4f}")
            print()
            continue

        # HF pipeline handles [MASK] natively
        try:
            preds = filler(text)
            # preds is a list (one entry per [MASK]) of list-of-dicts
            if isinstance(preds[0], dict):
                preds = [preds]  # single mask
            for i, mask_preds in enumerate(preds):
                print(f"  [MASK] #{i+1}:")
                for p in mask_preds:
                    print(f"    {p['token_str']:>20s}  {p['score']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_dir", default=MODEL_DIR)
    pa.add_argument("--data_path", default="./openwebtext")
    pa.add_argument("-n", "--num_samples", type=int, default=100)
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--show", type=int, default=5, help="Detailed examples to show")
    pa.add_argument("--interactive", action="store_true")
    args = pa.parse_args()

    model, tokenizer, device = load_model(args.model_dir)

    if args.interactive:
        interactive(model, tokenizer, device)
        return

    # ---- load data ----
    print(f"[*] Loading data from {args.data_path} ...")
    try:
        ds = load_from_disk(args.data_path)
    except FileNotFoundError:
        ds = load_dataset(args.data_path, split="train")
    if hasattr(ds, "keys"):
        ds = ds[list(ds.keys())[0]]
    n = min(args.num_samples, len(ds))
    texts = ds.select(range(n))["text"]

    # ---- detailed examples ----
    print(f"\n{'='*60}")
    print(f"  Detailed examples ({min(args.show, n)} samples)")
    print(f"{'='*60}\n")

    for i in tqdm(range(min(args.show, n)), desc="Generating examples"):
        snippet = texts[i][:100].replace("\n", " ")
        print(f"--- Sample {i+1}: \"{snippet}...\"")
        for r in predict_masked(texts[i], model, tokenizer, device, n_mask=3):
            tag = "✓" if r["correct"] else "✗"
            print(f"  [{tag}] pos={r['pos']:>4d}  '{r['original']:>15s}' → '{r['predicted']}'")
            for p in r["top_k"][:3]:
                print(f"         {p['token']:>20s}  ({p['prob']})")
        print()

    # ---- batch eval ----
    print(f"{'='*60}")
    print(f"  Batch eval on {n:,} samples")
    print(f"{'='*60}\n")

    stats = batch_eval(texts, model, tokenizer, device, args.batch_size)
    for k, v in stats.items():
        print(f"  {k:>20s}: {v}")

    out_path = os.path.join(args.model_dir, "inference_results.json")
    json.dump(stats, open(out_path, "w"), indent=2)
    print(f"\n[*] Saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
train_raw_bert.py
Pretrain a ~128M param BERT from scratch on OpenWebText via MLM.
Uses HuggingFace Trainer + Accelerate for multi-GPU DDP + fp16.

Launch (dual 3090):
    accelerate launch --num_processes 2 train_raw_bert.py -n 500000 --epochs 2

Single GPU fallback:
    python train_raw_bert.py -n 500000 --epochs 2
"""

import argparse
import csv
import os
import time
import json
import math
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from datasets import load_from_disk, load_dataset, Dataset as HFDataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    ProgressCallback,
    PrinterCallback,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Constants
MODEL_OUT     = os.path.abspath("./raw_bert")
METRICS_CSV   = "./training_metrics.csv"
TOKENIZER_DIR = os.path.abspath("./raw_bert_tokenizer")

# ~128M params: 14 layers × hidden 768 × intermediate 3328
BERT_CFG = dict(
    vocab_size=30_000,
    hidden_size=768,
    num_hidden_layers=14,
    num_attention_heads=12,
    intermediate_size=3328,
    max_position_embeddings=512,
    type_vocab_size=2,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)

MLM_PROB = 0.15


# Custom Trainer - grabs per-step accuracy from logits
class MLMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_train_acc = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        with torch.no_grad():
            labels = inputs.get("labels")
            if labels is not None:
                preds = outputs.logits.argmax(dim=-1)
                mask = labels != -100
                if mask.sum() > 0:
                    self._last_train_acc = (
                        (preds[mask] == labels[mask]).float().mean().item()
                    )
        return (loss, outputs) if return_outputs else loss


# CSV callback - writes every Trainer log event to a flat CSV
class CSVMetricsCallback(TrainerCallback):
    FIELDS = [
        "epoch", "step", "global_step", "phase",
        "loss", "perplexity",
        "accuracy", "precision", "recall", "f1_macro", "f1_weighted",
        "learning_rate", "samples_per_sec", "wall_time_sec",
    ]

    def __init__(self, path, trainer_ref):
        self.path = path
        self.trainer_ref = trainer_ref
        self.t0 = time.time()
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()
        self._f.flush()

    def _row(self, d):
        self._w.writerow(d)
        self._f.flush()

    def on_log(self, args, state: TrainerState, control: TrainerControl,
               logs=None, **kwargs):
        if logs is None:
            return
        wall = time.time() - self.t0
        # training step
        if "loss" in logs:
            l = logs["loss"]
            ppl = min(math.exp(l), 1e6) if l < 20 else 1e6
            self._row({
                "epoch":          f"{state.epoch:.4f}" if state.epoch else "",
                "step":           state.global_step,
                "global_step":    state.global_step,
                "phase":          "train",
                "loss":           f"{l:.6f}",
                "perplexity":     f"{ppl:.2f}",
                "accuracy":       f"{self.trainer_ref._last_train_acc:.6f}",
                "precision":      "", "recall": "",
                "f1_macro":       "", "f1_weighted": "",
                "learning_rate":  f"{logs.get('learning_rate', 0):.8f}",
                "samples_per_sec": f"{logs.get('train_samples_per_second', 0):.1f}",
                "wall_time_sec":  f"{wall:.1f}",
            })
        # eval
        if "eval_loss" in logs:
            l = logs["eval_loss"]
            ppl = min(math.exp(l), 1e6) if l < 20 else 1e6
            self._row({
                "epoch":          f"{state.epoch:.4f}" if state.epoch else "",
                "step":           state.global_step,
                "global_step":    state.global_step,
                "phase":          "val",
                "loss":           f"{l:.6f}",
                "perplexity":     f"{ppl:.2f}",
                "accuracy":       f"{logs.get('eval_accuracy', 0):.6f}",
                "precision":      f"{logs.get('eval_precision', 0):.6f}",
                "recall":         f"{logs.get('eval_recall', 0):.6f}",
                "f1_macro":       f"{logs.get('eval_f1_macro', 0):.6f}",
                "f1_weighted":    f"{logs.get('eval_f1_weighted', 0):.6f}",
                "learning_rate":  "",
                "samples_per_sec": f"{logs.get('eval_samples_per_second', 0):.1f}",
                "wall_time_sec":  f"{wall:.1f}",
            })

    def on_train_end(self, *a, **kw):
        self._f.close()


# Custom progress bar - puts loss/acc/lr/ppl into the tqdm postfix
class TQDMProgressWithStats(ProgressCallback):
    """Replaces default progress callback to show stats inline in tqdm bar."""

    def __init__(self, trainer_ref):
        super().__init__()
        self.trainer_ref = trainer_ref

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.training_bar is None:
            return
        postfix = {}
        if "loss" in logs:
            postfix["loss"] = f"{logs['loss']:.3f}"
            ppl = min(math.exp(logs['loss']), 1e6) if logs['loss'] < 20 else 1e6
            postfix["ppl"] = f"{ppl:.1f}"
        if "grad_norm" in logs:
            postfix["grad"] = f"{logs['grad_norm']:.2f}"
        if "learning_rate" in logs:
            postfix["lr"] = f"{logs['learning_rate']:.1e}"
        postfix["acc"] = f"{self.trainer_ref._last_train_acc:.3f}"
        if "epoch" in logs:
            postfix["ep"] = f"{logs['epoch']:.2f}"
        self.training_bar.set_postfix(postfix, refresh=True)


# Tokenizer builder
def build_or_load_tokenizer(texts, vocab_size=30_000, sample_size=200_000):
    if (os.path.isdir(TOKENIZER_DIR)
            and os.path.exists(os.path.join(TOKENIZER_DIR, "tokenizer.json"))):
        print(f"[*] Loading existing tokenizer from {TOKENIZER_DIR}")
        return BertTokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)

    print(f"[*] Training BPE tokenizer (vocab={vocab_size}) ...")
    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    tok_n = min(sample_size, len(texts))
    tmp = "/tmp/_owt_tok_corpus.txt"
    with open(tmp, "w") as f:
        for i in tqdm(range(tok_n), desc="Writing tokenizer corpus"):
            f.write(texts[i].replace("\n", " ") + "\n")

    bpe = ByteLevelBPETokenizer()
    bpe.train(
        files=[tmp],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=bpe._tokenizer,
        unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
    )
    fast.save_pretrained(TOKENIZER_DIR)
    tok = BertTokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    print(f"[*] Tokenizer ready - vocab_size={tok.vocab_size}")
    if os.path.exists(tmp):
        os.remove(tmp)
    return tok


# Tokenize into HF Dataset (with tqdm)
def tokenize_texts(texts, tokenizer, max_len, desc="Tokenizing"):
    all_ids, all_masks = [], []
    for t in tqdm(texts, desc=desc):
        enc = tokenizer(t, truncation=True, max_length=max_len, padding="max_length")
        all_ids.append(enc["input_ids"])
        all_masks.append(enc["attention_mask"])
    ds = HFDataset.from_dict({"input_ids": all_ids, "attention_mask": all_masks})
    ds.set_format("torch")
    return ds


# Eval metrics (called by Trainer at each eval)
def preprocess_logits_for_eval(logits, labels):
    """Reduce (batch, seq, vocab) -> (batch, seq) argmax BEFORE accumulation.
    This prevents OOM from storing full vocab logits for every eval sample."""
    return logits.argmax(dim=-1)


def compute_eval_metrics(eval_pred):
    preds, labels = eval_pred        # both (N, seq) after preprocess
    mask = labels != -100
    p, g = preds[mask], labels[mask]
    return {
        "accuracy":    accuracy_score(g, p),
        "precision":   precision_score(g, p, average="weighted", zero_division=0),
        "recall":      recall_score(g, p, average="weighted", zero_division=0),
        "f1_macro":    f1_score(g, p, average="macro",    zero_division=0),
        "f1_weighted": f1_score(g, p, average="weighted", zero_division=0),
    }


# Main
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_path",  default="./openwebtext")
    pa.add_argument("-n", "--num_samples", type=int, default=500_000)
    pa.add_argument("--epochs",     type=int, default=2)
    pa.add_argument("--per_device_batch_size", type=int, default=32,
                    help="Per-GPU batch size")
    pa.add_argument("--gradient_accumulation_steps", type=int, default=1)
    pa.add_argument("--max_len",    type=int, default=512)
    pa.add_argument("--lr",         type=float, default=1e-4)
    pa.add_argument("--warmup_ratio", type=float, default=0.06)
    pa.add_argument("--log_every",  type=int, default=25)
    pa.add_argument("--eval_ratio", type=float, default=0.02)
    pa.add_argument("--workers",    type=int, default=4)
    pa.add_argument("--seed",       type=int, default=42)
    args = pa.parse_args()

    os.makedirs(MODEL_OUT, exist_ok=True)

    # data
    print(f"[*] Loading data from {args.data_path} ...")
    if os.path.isdir(args.data_path):
        try:
            ds = load_from_disk(args.data_path)
            if hasattr(ds, "keys"):
                ds = ds[list(ds.keys())[0]]
        except FileNotFoundError:
            ds = load_dataset(args.data_path, split="train")
            if hasattr(ds, "keys"):
                ds = ds[list(ds.keys())[0]]
    else:
        ds = load_dataset("Skylion007/openwebtext", split="train")

    n = min(args.num_samples, len(ds))
    print(f"[*] Selecting {n:,} / {len(ds):,} samples")
    ds = ds.select(range(n))
    texts = ds["text"]

    # tokenizer (rank 0 builds, others wait)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if local_rank == 0:
        tokenizer = build_or_load_tokenizer(texts, BERT_CFG["vocab_size"])
        # Signal other ranks that the tokenizer is ready
        if world_size > 1:
            Path(os.path.join(TOKENIZER_DIR, ".ready")).touch()

    if world_size > 1 and local_rank != 0:
        # Wait for rank 0 to finish building the tokenizer
        ready_flag = os.path.join(TOKENIZER_DIR, ".ready")
        while not os.path.exists(ready_flag):
            time.sleep(0.5)
        tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    BERT_CFG["vocab_size"] = tokenizer.vocab_size

    # tokenize into datasets
    val_n   = min(int(n * 0.2), max(1_000, int(n * args.eval_ratio)))
    tr_txt  = texts[: n - val_n]
    va_txt  = texts[n - val_n :]
    print(f"[*] Train {len(tr_txt):,}  Val {len(va_txt):,}")

    train_ds = tokenize_texts(tr_txt, tokenizer, args.max_len, "Tokenizing train")
    val_ds   = tokenize_texts(va_txt, tokenizer, args.max_len, "Tokenizing val")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB,
    )

    # model
    cfg   = BertConfig(**BERT_CFG)
    model = BertForMaskedLM(cfg)
    model.gradient_checkpointing_enable()
    npar  = sum(p.numel() for p in model.parameters())
    print(f"[*] Model: {npar:,} params ({npar/1e6:.1f}M) [gradient checkpointing ON]")

    ngpu = max(torch.cuda.device_count(), 1)
    eff_bs = args.per_device_batch_size * ngpu * args.gradient_accumulation_steps
    print(f"[*] GPUs: {ngpu}  Effective batch size: {eff_bs}")

    # Compute warmup steps from ratio
    steps_per_epoch = math.ceil(len(train_ds) / eff_bs)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    # Trainer args
    training_args = TrainingArguments(
        output_dir=MODEL_OUT,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to="none",
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        disable_tqdm=False,
    )

    # build trainer
    trainer = MLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_eval_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_eval,
    )
    trainer.add_callback(CSVMetricsCallback(METRICS_CSV, trainer))

    # Replace default progress/printer with our stats-in-tqdm version
    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(TQDMProgressWithStats(trainer))

    # go
    print(f"\n{'='*60}")
    print(f"  BERT MLM Pretraining - {npar/1e6:.1f}M params")
    print(f"  {len(tr_txt):,} train / {len(va_txt):,} val / {args.epochs} epochs")
    print(f"  Batch {eff_bs} (per_dev={args.per_device_batch_size} x {ngpu} GPU)")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.train()
    wall = time.time() - t0

    # save
    print(f"\n[*] Saving model to {MODEL_OUT}/ ...")
    trainer.save_model(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    cfg.save_pretrained(MODEL_OUT)

    json.dump({
        "params": npar, "params_M": round(npar/1e6, 1),
        "samples": n, "train": len(tr_txt), "val": len(va_txt),
        "epochs": args.epochs, "batch": eff_bs,
        "wall_sec": round(wall), "wall_hours": round(wall/3600, 2),
        "gpus": ngpu, "bert_config": BERT_CFG,
    }, open(os.path.join(MODEL_OUT, "training_meta.json"), "w"), indent=2)

    print(f"[*] Done in {wall/3600:.2f}h  |  {METRICS_CSV}  |  {MODEL_OUT}/")


if __name__ == "__main__":
    main()

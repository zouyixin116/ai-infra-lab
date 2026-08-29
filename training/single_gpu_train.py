#!/usr/bin/env python3
"""Fine-tune TinyLlama on a small TinyStories subset on one CUDA GPU."""

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--dataset-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    positive = (
        "dataset_samples",
        "batch_size",
        "sequence_length",
        "steps",
        "learning_rate",
        "log_every",
    )
    if any(getattr(args, name) <= 0 for name in positive) or args.warmup_steps < 0:
        parser.error("numeric arguments must be positive; warmup-steps may be zero")
    return args


def tensor_fingerprint(model) -> str:
    """Fingerprint a stable parameter sample to verify save/reload fidelity."""
    # Sample only 4,096 values so verification does not copy a large matrix to CPU.
    parameter = next(model.parameters()).detach().flatten()[:4096].float().cpu().contiguous()
    return hashlib.sha256(parameter.numpy().tobytes()).hexdigest()


def main() -> int:
    args = parse_args()
    try:
        import torch
        from datasets import load_dataset
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
        print("Install torch, transformers, datasets, and sentencepiece.", file=sys.stderr)
        return 1

    # Stage 1 is a controlled single-GPU baseline. The host may have multiple
    # GPUs, but this process must see only one; select it with CUDA_VISIBLE_DEVICES.
    if not torch.cuda.is_available():
        print("ERROR: this Stage 1 benchmark requires one CUDA GPU", file=sys.stderr)
        return 1
    if torch.cuda.device_count() != 1:
        print(f"ERROR: expected exactly one visible GPU, found {torch.cuda.device_count()}", file=sys.stderr)
        print("Set CUDA_VISIBLE_DEVICES to select one GPU.", file=sys.stderr)
        return 1

    # Seed Python, PyTorch, and CUDA so the shuffled data order is reproducible.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    # The tokenizer must match TinyLlama: token IDs index rows in its embedding.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        # TinyLlama has no separate pad token, so reuse EOS only as a fill value.
        # Padding is masked below and therefore does not contribute to the loss.
        tokenizer.pad_token = tokenizer.eos_token

    # Build a small pool from the first N stories; the loop may not consume it all.
    raw_dataset = load_dataset(args.dataset, split=f"train[:{args.dataset_samples}]")

    def tokenize(batch):
        # Truncate long stories and pad short ones to a fixed sequence length.
        return tokenizer(batch["text"], truncation=True, max_length=args.sequence_length, padding="max_length")

    # Convert text to input IDs and masks, returned as PyTorch tensors on access.
    tokenized = raw_dataset.map(tokenize, batched=True, remove_columns=raw_dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    # A dedicated generator makes shuffling repeatable; the loader yields batches.
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(tokenized, batch_size=args.batch_size, shuffle=True, generator=generator)

    # Load all pretrained TinyLlama weights in BF16 and move them to the GPU.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    # train() enables training behavior such as dropout; it does not update weights.
    model.train()
    # Passing model.parameters() makes this full fine-tuning, including embeddings.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    # Restart peak tracking after model load; this does not free allocated memory.
    torch.cuda.reset_peak_memory_stats(device)
    losses, step_times = [], []
    measured_tokens = 0
    iterator = iter(loader)

    # Total updates equal warmup plus measured steps. range excludes its endpoint,
    # hence the final +1.
    for step in range(1, args.warmup_steps + args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            # If the run exceeds one epoch, create a new iterator and keep training.
            iterator = iter(loader)
            batch = next(iterator)
        # DataLoader yields CPU tensors; to(device) copies them through CUDA.
        # Without pinned host memory, non_blocking=True may not be truly asynchronous.
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        # CausalLM shifts labels internally for next-token prediction. Cross entropy
        # ignores -100, so padded positions do not affect the loss.
        labels = input_ids.masked_fill(attention_mask == 0, -100)
        # CUDA is asynchronous; synchronize around timing to measure the full step.
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        # One complete update: clear gradients, forward/loss, backward, AdamW step.
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output.loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        # Warmup updates weights but is excluded from metrics to avoid startup costs.
        if step > args.warmup_steps:
            measured_step = step - args.warmup_steps
            loss_value = float(output.loss.detach())
            # Real tokens are 1 and padding is 0, so the mask sum counts real tokens.
            tokens = int(attention_mask.sum().item())
            losses.append(loss_value)
            step_times.append(elapsed)
            measured_tokens += tokens
            if measured_step == 1 or measured_step % args.log_every == 0 or measured_step == args.steps:
                print(f"step={measured_step}/{args.steps} loss={loss_value:.4f} step_ms={elapsed * 1000:.2f} tokens_per_second={tokens / elapsed:.2f}")

    # Read the training peak before reload so verification does not affect the metric.
    peak_memory = torch.cuda.max_memory_allocated(device)
    saved_fingerprint = tensor_fingerprint(model)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.checkpoint_dir)
    # Save model/tokenizer normally and optimizer state separately for future resume.
    torch.save(
        {"optimizer": optimizer.state_dict(), "completed_steps": args.steps, "seed": args.seed},
        args.checkpoint_dir / "optimizer_state.pt",
    )

    # Delete the trained in-memory model so verification must use the checkpoint.
    del output, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    reload_fingerprint = tensor_fingerprint(reloaded)
    reloaded.eval()
    # Verify a matching parameter fingerprint and a finite loss after reload.
    # no_grad() avoids building a backward graph for this validation forward pass.
    with torch.no_grad():
        reload_loss = reloaded(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    reload_verified = saved_fingerprint == reload_fingerprint and bool(torch.isfinite(reload_loss).item())

    measured_seconds = sum(step_times)
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(device),
        "pytorch_version": torch.__version__,
        "model": args.model,
        "dataset": args.dataset,
        "dataset_samples": args.dataset_samples,
        "parameter_count": parameter_count,
        "precision": "bfloat16",
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "metrics": {
            "loss_per_step": [round(value, 6) for value in losses],
            "step_time_ms": [round(value * 1000, 3) for value in step_times],
            "average_step_time_ms": round(measured_seconds / args.steps * 1000, 3),
            "tokens_processed": measured_tokens,
            "tokens_per_second": round(measured_tokens / measured_seconds, 3),
            "peak_pytorch_memory_bytes": peak_memory,
        },
        "checkpoint": {
            "path": str(args.checkpoint_dir),
            "optimizer_state_saved": True,
            "reload_loss": round(float(reload_loss), 6),
            "reload_verified": reload_verified,
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if reload_verified else 1


if __name__ == "__main__":
    raise SystemExit(main())

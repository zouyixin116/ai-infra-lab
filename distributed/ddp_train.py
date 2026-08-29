#!/usr/bin/env python3
"""Fine-tune TinyLlama with one DDP process per CUDA GPU."""

import argparse
import gc
import hashlib
import json
import os
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
    parser.add_argument("--batch-size", type=int, default=2, help="Per-GPU batch size")  # Global batch = this value × world size.
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    positive = ("dataset_samples", "batch_size", "sequence_length", "steps", "learning_rate", "log_every")
    if any(getattr(args, name) <= 0 for name in positive) or args.warmup_steps < 0:
        parser.error("numeric arguments must be positive; warmup-steps may be zero")
    return args


def fingerprint(model) -> str:
    # Hash a small, stable parameter sample so ranks and reloaded checkpoints
    # can be compared without copying the entire 1.1B-parameter model to CPU.
    parameter = next(model.parameters()).detach().flatten()[:4096].float().cpu().contiguous()
    return hashlib.sha256(parameter.numpy().tobytes()).hexdigest()


def require_torchrun_environment():
    # torchrun creates the workers and injects their distributed identities.
    # A normal `python ddp_train.py` process does not have these variables.
    missing = [name for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if name not in os.environ]
    if missing:
        raise RuntimeError("launch with torchrun; missing environment variables: " + ", ".join(missing))
    return int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])


def main() -> int:
    args = parse_args()
    try:
        import torch
        import torch.distributed as dist
        from datasets import load_dataset
        from torch.nn.parallel import DistributedDataParallel
        from torch.utils.data import DataLoader, DistributedSampler
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
        return 1

    # Read the identity assigned to this worker by torchrun.
    try:
        rank, local_rank, world_size = require_torchrun_environment()
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    # Fail before downloading data or model weights when this is not a real
    # multi-process CUDA run.
    if not torch.cuda.is_available() or world_size < 2:
        print("ERROR: Stage 2 requires torchrun with at least two CUDA processes", file=sys.stderr)
        return 1
    if local_rank >= torch.cuda.device_count():
        print(f"ERROR: LOCAL_RANK={local_rank} has no visible CUDA device", file=sys.stderr)
        return 1

    # Use one process per GPU: LOCAL_RANK 0 owns cuda:0, rank 1 owns cuda:1,
    # and so on. LOCAL_RANK is node-local; RANK is globally unique.
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # env:// reads rendezvous information supplied by torchrun. NCCL implements
    # the GPU collective operations later used by DDP.
    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        # Python-level randomness may differ by rank, while PyTorch starts from
        # the same seed. DDP also broadcasts rank 0's parameters at construction.
        random.seed(args.seed + rank)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        # Every rank runs the same input pipeline, but DistributedSampler below
        # ensures that each rank consumes a different subset of examples.
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token  # Padding is masked from the causal-LM loss below.
        raw_dataset = load_dataset(args.dataset, split=f"train[:{args.dataset_samples}]")

        def tokenize(batch):
            # Fixed-length tensors make every rank execute compatible tensor shapes.
            return tokenizer(batch["text"], truncation=True, max_length=args.sequence_length, padding="max_length")

        tokenized = raw_dataset.map(tokenize, batched=True, remove_columns=raw_dataset.column_names)
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
        # DDP synchronizes gradients but does not split input data. The sampler
        # assigns a distinct dataset shard to each rank using one shared shuffle.
        sampler = DistributedSampler(tokenized, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        loader = DataLoader(tokenized, batch_size=args.batch_size, sampler=sampler)  # Build rank-local batches only.

        # Data parallelism replicates the entire model on every GPU; DDP does
        # not shard model parameters, gradients, or optimizer state.
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device)
        model.train()  # Enable training behavior such as dropout on every replica.
        # Every rank owns a full model replica. DDP registers backward hooks that
        # all-reduce gradients so all replicas receive the same optimizer update.
        ddp_model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)  # Broadcast rank 0 parameters and install gradient-sync hooks.
        # Every rank owns an optimizer with identical initial state. Because DDP
        # produces identical gradients, every optimizer applies the same update.
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.learning_rate)
        parameter_count = sum(parameter.numel() for parameter in ddp_model.parameters())
        torch.cuda.reset_peak_memory_stats(device)

        losses, step_times = [], []
        local_measured_tokens = 0
        iterator = iter(loader)  # This iterator yields only the current rank's data shard.
        epoch = 0
        for step in range(1, args.warmup_steps + args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)  # Change the deterministic shuffle while keeping all ranks consistent.
                iterator = iter(loader)
                batch = next(iterator)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = input_ids.masked_fill(attention_mask == 0, -100)  # -100 excludes padded positions from loss.

            dist.barrier()  # Align all ranks before measuring the synchronized training step.
            torch.cuda.synchronize(device)  # Finish queued CUDA work before starting the timer.
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)  # Prevent gradients from accumulating across optimizer steps.
            output = ddp_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  # Run rank-local forward.
            output.loss.backward()  # Trigger DDP hooks and NCCL gradient AllReduce during backward.
            optimizer.step()  # Apply the same synchronized gradients independently on every rank.
            torch.cuda.synchronize(device)  # Wait for the GPU step to finish before stopping the timer.
            elapsed = time.perf_counter() - started

            if step > args.warmup_steps:
                measured_step = step - args.warmup_steps
                # This second AllReduce is metrics-only. Gradient synchronization
                # already happened inside backward() above.
                reduced_loss = output.loss.detach().float()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)  # Sum rank-local losses for reporting only.
                reduced_loss /= world_size  # Convert the summed loss into the cross-rank mean.
                local_tokens = int(attention_mask.sum().item())
                local_measured_tokens += local_tokens
                losses.append(float(reduced_loss))
                step_times.append(elapsed)
                if rank == 0 and (measured_step == 1 or measured_step % args.log_every == 0 or measured_step == args.steps):
                    print(f"step={measured_step}/{args.steps} loss={float(reduced_loss):.4f} rank0_step_ms={elapsed * 1000:.2f}")

        # Convert rank-local measurements to tensors so NCCL can aggregate them.
        peak_memory = torch.tensor(torch.cuda.max_memory_allocated(device), dtype=torch.int64, device=device)
        dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)  # Report the worst per-rank memory peak.
        total_tokens = torch.tensor(local_measured_tokens, dtype=torch.int64, device=device)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)  # Convert local token counts into global work.
        max_total_time = torch.tensor(sum(step_times), dtype=torch.float64, device=device)
        dist.all_reduce(max_total_time, op=dist.ReduceOp.MAX)  # Throughput is limited by the slowest rank.

        # DDP keeps parameters synchronized. Rank 0 alone writes the shared checkpoint.
        saved_fingerprint = fingerprint(ddp_model.module)
        fingerprints = [None] * world_size
        dist.all_gather_object(fingerprints, saved_fingerprint)  # Collect every replica's fingerprint on every rank.
        parameters_synchronized = len(set(fingerprints)) == 1
        if rank == 0:
            # Only global rank 0 writes shared files, preventing concurrent
            # checkpoint corruption and duplicate JSON output.
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ddp_model.module.save_pretrained(args.checkpoint_dir, safe_serialization=True)  # Save the unwrapped model.
            tokenizer.save_pretrained(args.checkpoint_dir)
            torch.save(
                {"optimizer": optimizer.state_dict(), "completed_steps": args.steps, "world_size": world_size, "seed": args.seed},
                args.checkpoint_dir / "optimizer_state.pt",
            )
        dist.barrier()  # Do not let any rank continue until rank 0 finishes writing the checkpoint.

        reload_verified = False
        reload_loss_value = None
        if rank == 0:
            # Free the trained replica before reload so verification must read
            # the checkpoint from disk rather than reuse in-memory parameters.
            del output, optimizer, ddp_model, model
            gc.collect()
            torch.cuda.empty_cache()
            reloaded = AutoModelForCausalLM.from_pretrained(
                args.checkpoint_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
            ).to(device)
            reloaded.eval()
            with torch.no_grad():
                reload_loss = reloaded(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            reload_loss_value = round(float(reload_loss), 6)
            reload_verified = saved_fingerprint == fingerprint(reloaded) and bool(torch.isfinite(reload_loss).item())  # Verify fidelity and usability.
            measured_seconds = float(max_total_time.item())
            report = {
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "gpu": torch.cuda.get_device_name(device),
                "pytorch_version": torch.__version__,
                "backend": "nccl",
                "model": args.model,
                "dataset": args.dataset,
                "dataset_samples": args.dataset_samples,
                "parameter_count": parameter_count,
                "precision": "bfloat16",
                "world_size": world_size,
                "batch_size_per_gpu": args.batch_size,
                "global_batch_size": args.batch_size * world_size,
                "sequence_length": args.sequence_length,
                "steps": args.steps,
                "warmup_steps": args.warmup_steps,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "metrics": {
                    "mean_loss_per_step": [round(value, 6) for value in losses],
                    "rank0_step_time_ms": [round(value * 1000, 3) for value in step_times],
                    "elapsed_seconds_max_rank": round(measured_seconds, 6),
                    "tokens_processed_global": int(total_tokens.item()),
                    "tokens_per_second_global": round(int(total_tokens.item()) / measured_seconds, 3),
                    "peak_pytorch_memory_bytes_max_rank": int(peak_memory.item()),
                },
                "checkpoint": {
                    "path": str(args.checkpoint_dir),
                    "optimizer_state_saved": True,
                    "parameters_synchronized": parameters_synchronized,
                    "reload_loss": reload_loss_value,
                    "reload_verified": reload_verified,
                },
            }
            rendered = json.dumps(report, indent=2)
            print(rendered)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")

        # Rank 0 owns reload verification, so broadcast one final pass/fail value
        # and make every torchrun worker return the same job status.
        status = torch.tensor(int(parameters_synchronized and (reload_verified if rank == 0 else True)), device=device)
        dist.broadcast(status, src=0)  # Propagate rank 0's final verification result to all ranks.
        return 0 if int(status.item()) == 1 else 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()  # Release NCCL resources even when training raises an exception.


if __name__ == "__main__":
    raise SystemExit(main())

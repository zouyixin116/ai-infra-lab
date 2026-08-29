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
    parser.add_argument("--batch-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    positive = ("dataset_samples", "sequence_length", "steps", "learning_rate", "log_every")
    if any(getattr(args, name) <= 0 for name in positive) or args.warmup_steps < 0:
        parser.error("numeric arguments must be positive; warmup-steps may be zero")
    return args


def tensor_fingerprint(model) -> str:
    """Fingerprint a stable parameter sample to verify save/reload fidelity."""
    # 只采样第一个参数的前 4096 个值，避免为了校验而复制整个大矩阵到 CPU。
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

    # Stage 1 是受控的单 GPU 基线：机器可以有多张卡，但该进程必须只看见一张。
    # 多卡机器可用 CUDA_VISIBLE_DEVICES=0（或其他编号）选择其中一张。
    if not torch.cuda.is_available():
        print("ERROR: this Stage 1 benchmark requires one CUDA GPU", file=sys.stderr)
        return 1
    if torch.cuda.device_count() != 1:
        print(f"ERROR: expected exactly one visible GPU, found {torch.cuda.device_count()}", file=sys.stderr)
        print("Set CUDA_VISIBLE_DEVICES to select one GPU.", file=sys.stderr)
        return 1

    # 固定 Python、PyTorch 和 CUDA 随机状态，让数据打乱顺序尽可能可复现。
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    # 必须加载与 TinyLlama 权重匹配的 tokenizer：词表中的 token ID 要与
    # 模型 embedding 矩阵的行号一一对应。
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        # TinyLlama 没有独立 pad token；复用 EOS 只作为补齐值。padding 位置稍后
        # 会在 attention_mask 和 labels 中被屏蔽，不参与 loss。
        tokenizer.pad_token = tokenizer.eos_token

    # 先取 TinyStories 训练集的前 N 条作为候选数据池；训练循环不一定遍历完。
    raw_dataset = load_dataset(args.dataset, split=f"train[:{args.dataset_samples}]")

    def tokenize(batch):
        # 每篇故事统一成 sequence_length 个位置：长故事截断，短故事补 padding。
        return tokenizer(batch["text"], truncation=True, max_length=args.sequence_length, padding="max_length")

    # 将文本批量转换为 input_ids 和 attention_mask，并在读取时转为 PyTorch Tensor。
    tokenized = raw_dataset.map(tokenize, batched=True, remove_columns=raw_dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    # 独立 generator 固定 shuffle 顺序；DataLoader 每次返回一个训练 batch。
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(tokenized, batch_size=args.batch_size, shuffle=True, generator=generator)

    # 加载预训练 TinyLlama 的全部 BF16 权重并移到唯一可见的 GPU。
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    # train() 只切换训练行为（例如 dropout）；真正更新参数的是 backward + step。
    model.train()
    # 传入 model.parameters() 表示 embedding、attention、MLP 等全部参数都训练。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    # 从模型已加载的状态重新开始记录峰值；不会释放当前已经使用的显存。
    torch.cuda.reset_peak_memory_stats(device)
    losses, step_times = [], []
    measured_tokens = 0
    iterator = iter(loader)

    # 总更新次数 = warmup_steps + measured steps。range 的终点不包含在内，
    # 因此需要 +1 才能覆盖最后一步。
    for step in range(1, args.warmup_steps + args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            # 如果 steps 超过一个 epoch，就创建新 iterator 并从下一轮继续读取。
            iterator = iter(loader)
            batch = next(iterator)
        # DataLoader 产生 CPU Tensor；to(device) 通过 PyTorch 的 CUDA 后端复制到显存。
        # 当前 loader 未使用 pinned memory，所以 non_blocking=True 不保证真正异步。
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        # CausalLM 会在内部把 labels 错开一位来预测 next token。把 padding 改为
        # -100，因为 PyTorch cross entropy 默认忽略 label=-100 的位置。
        labels = input_ids.masked_fill(attention_mask == 0, -100)
        # CUDA 默认异步执行；计时前后同步，才能测到完整 GPU step 的真实耗时。
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        # 一个完整训练 step：清梯度 → forward/loss → backward → AdamW 更新参数。
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output.loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        # warmup 同样会更新参数，但不进入性能结果，避免一次性初始化污染计时。
        if step > args.warmup_steps:
            measured_step = step - args.warmup_steps
            loss_value = float(output.loss.detach())
            # mask 中真实 token 为 1、padding 为 0，所以求和得到有效 token 数。
            tokens = int(attention_mask.sum().item())
            losses.append(loss_value)
            step_times.append(elapsed)
            measured_tokens += tokens
            if measured_step == 1 or measured_step % args.log_every == 0 or measured_step == args.steps:
                print(f"step={measured_step}/{args.steps} loss={loss_value:.4f} step_ms={elapsed * 1000:.2f} tokens_per_second={tokens / elapsed:.2f}")

    # 在 checkpoint reload 之前读取训练峰值，避免把验证过程混进该指标。
    peak_memory = torch.cuda.max_memory_allocated(device)
    saved_fingerprint = tensor_fingerprint(model)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.checkpoint_dir)
    # save_pretrained 保存模型和 tokenizer；optimizer state 单独保存，用于未来续训。
    torch.save(
        {"optimizer": optimizer.state_dict(), "completed_steps": args.steps, "seed": args.seed},
        args.checkpoint_dir / "optimizer_state.pt",
    )

    # 删除内存中的原模型后再加载，确保验证的确使用磁盘 checkpoint，
    # 而不是误用仍在显存中的训练模型。
    del output, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    reload_fingerprint = tensor_fingerprint(reloaded)
    reloaded.eval()
    # reload 验证包含两部分：参数采样指纹相同，并且重载模型可算出有限 loss。
    # no_grad() 禁止为验证 forward 建立反向传播计算图，节省显存。
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

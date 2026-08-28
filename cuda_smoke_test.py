#!/usr/bin/env python3
"""Run and time a minimal CUDA matrix-multiplication workload."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2048, help="square matrix size")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    parser.add_argument("--iterations", type=int, default=10, help="timed iterations")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    if args.size <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error("size and iterations must be positive; warmup cannot be negative")

    try:
        import torch
    except ImportError as exc:
        print(f"ERROR: PyTorch is not installed: {exc}")
        return 1

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available to PyTorch")
        return 1

    device = torch.device("cuda:0")
    a = torch.randn((args.size, args.size), device=device)
    b = torch.randn((args.size, args.size), device=device)

    # Warmup avoids including one-time CUDA context and kernel setup in timing.
    for _ in range(args.warmup):
        result = a @ b
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iterations):
        result = a @ b
    end.record()
    torch.cuda.synchronize(device)  # CUDA is asynchronous; wait before reading time.

    total_ms = start.elapsed_time(end)
    average_ms = total_ms / args.iterations
    operations = 2 * args.size**3 * args.iterations
    try:
        peak_memory_bytes = torch.cuda.max_memory_allocated()
    except RuntimeError:
        # Some CUDA/PyTorch builds do not expose allocator peak statistics.
        peak_memory_bytes = None

    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
        "matrix_size": args.size,
        "dtype": str(a.dtype),
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "total_time_ms": round(total_ms, 3),
        "average_time_ms": round(average_ms, 3),
        "approx_tflops": round(operations / (total_ms / 1000) / 1e12, 3),
        "peak_pytorch_memory_bytes": peak_memory_bytes,
        "output_is_finite": bool(torch.isfinite(result).all().item()),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    return 0 if report["output_is_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Compare a pretrained model with a continued-training checkpoint."""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="original Hugging Face model ID or local checkpoint",
    )
    parser.add_argument("--trained-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--chunk-elements",
        type=int,
        default=1_000_000,
        help="elements converted to float32 at once",
    )
    args = parser.parse_args()
    if args.top_k <= 0 or args.chunk_elements <= 0:
        parser.error("top-k and chunk-elements must be positive")
    if not args.trained_model.is_dir():
        parser.error(f"trained checkpoint directory not found: {args.trained_model}")
    return args


def main() -> int:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
        return 1

    print(f"Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    print(f"Loading trained model: {args.trained_model}")
    trained = AutoModelForCausalLM.from_pretrained(
        args.trained_model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    # Keeping both BF16 models on CPU avoids consuming GPU memory. Each parameter
    # is compared in small FP32 chunks so large embedding matrices do not create
    # several full-size temporary copies.
    trained_parameters = dict(trained.named_parameters())
    tensor_results = []
    global_elements = 0
    global_changed = 0
    global_delta_sq = 0.0
    global_base_sq = 0.0
    global_abs_sum = 0.0
    global_max_abs = 0.0

    base_parameters = dict(base.named_parameters())
    if base_parameters.keys() != trained_parameters.keys():
        missing = sorted(base_parameters.keys() - trained_parameters.keys())
        extra = sorted(trained_parameters.keys() - base_parameters.keys())
        print(f"ERROR: parameter names differ; missing={missing[:5]}, extra={extra[:5]}", file=sys.stderr)
        return 1

    for index, (name, before_parameter) in enumerate(base_parameters.items(), start=1):
        after_parameter = trained_parameters[name]
        if before_parameter.shape != after_parameter.shape:
            print(f"ERROR: shape mismatch for {name}", file=sys.stderr)
            return 1

        before_flat = before_parameter.detach().view(-1)
        after_flat = after_parameter.detach().view(-1)
        elements = before_flat.numel()
        changed = 0
        delta_sq = 0.0
        base_sq = 0.0
        abs_sum = 0.0
        max_abs = 0.0

        for start in range(0, elements, args.chunk_elements):
            end = min(start + args.chunk_elements, elements)
            before_bf16 = before_flat[start:end]
            after_bf16 = after_flat[start:end]
            changed += int(torch.count_nonzero(before_bf16 != after_bf16))
            before_fp32 = before_bf16.float()
            delta = after_bf16.float() - before_fp32
            delta_sq += float(torch.sum(delta * delta, dtype=torch.float64))
            base_sq += float(torch.sum(before_fp32 * before_fp32, dtype=torch.float64))
            abs_sum += float(torch.sum(torch.abs(delta), dtype=torch.float64))
            max_abs = max(max_abs, float(torch.max(torch.abs(delta))))

        delta_l2 = math.sqrt(delta_sq)
        base_l2 = math.sqrt(base_sq)
        result = {
            "name": name,
            "shape": list(before_parameter.shape),
            "elements": elements,
            "changed_elements": changed,
            "changed_fraction": changed / elements,
            "mean_abs_delta": abs_sum / elements,
            "max_abs_delta": max_abs,
            "delta_l2": delta_l2,
            "relative_l2_change": delta_l2 / base_l2 if base_l2 else None,
        }
        tensor_results.append(result)
        global_elements += elements
        global_changed += changed
        global_delta_sq += delta_sq
        global_base_sq += base_sq
        global_abs_sum += abs_sum
        global_max_abs = max(global_max_abs, max_abs)
        print(
            f"[{index}/{len(base_parameters)}] {name}: "
            f"changed={result['changed_fraction']:.2%}, "
            f"relative_l2={result['relative_l2_change']:.3e}"
        )

    global_delta_l2 = math.sqrt(global_delta_sq)
    global_base_l2 = math.sqrt(global_base_sq)
    ranked = sorted(
        tensor_results,
        key=lambda item: item["relative_l2_change"] or 0.0,
        reverse=True,
    )
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": str(args.base_model),
        "trained_model": str(args.trained_model),
        "comparison_dtype": "float32 chunks from bfloat16 weights",
        "global": {
            "parameter_tensors": len(tensor_results),
            "elements": global_elements,
            "changed_elements": global_changed,
            "changed_fraction": global_changed / global_elements,
            "mean_abs_delta": global_abs_sum / global_elements,
            "max_abs_delta": global_max_abs,
            "delta_l2": global_delta_l2,
            "base_l2": global_base_l2,
            "relative_l2_change": global_delta_l2 / global_base_l2,
        },
        "top_changed_tensors_by_relative_l2": ranked[: args.top_k],
        "all_tensors": tensor_results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify two-GPU process binding and NCCL all-reduce communication."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    # torchrun supplies these variables to every process it creates.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Each process owns one GPU. NCCL carries collective operations between them.
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        value = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
        print(
            f"before: rank={rank}, local_rank={local_rank}, "
            f"world_size={world_size}, device={value.device}, value={value.item()}",
            flush=True,
        )

        # SUM is performed across all ranks and the result is returned to every rank.
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        print(
            f"after:  rank={rank}, local_rank={local_rank}, "
            f"device={value.device}, value={value.item()}",
            flush=True,
        )

        expected = world_size * (world_size + 1) / 2
        if value.item() != expected:
            raise RuntimeError(f"all-reduce returned {value.item()}, expected {expected}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

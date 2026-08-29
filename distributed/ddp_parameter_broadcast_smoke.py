#!/usr/bin/env python3
"""Show how DDP broadcasts rank 0 model parameters during construction."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def main() -> None:
    # torchrun assigns one worker process to each local GPU.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)  # Bind this worker to its assigned GPU.
    device = torch.device("cuda", local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )  # Join both workers into the same NCCL process group.

    try:
        model = torch.nn.Linear(1, 1, bias=False, device=device)
        with torch.no_grad():
            model.weight.fill_(1.0 if rank == 0 else 10.0)  # Deliberately start the replicas with different weights.

        weight_before_ddp = model.weight.item()

        ddp_model = DistributedDataParallel(  # Broadcast rank 0 parameters to every other rank during construction.
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

        weight_after_ddp = ddp_model.module.weight.item()

        print(
            f"rank={rank}, "
            f"weight_before_ddp={weight_before_ddp:.1f}, "
            f"weight_after_ddp={weight_after_ddp:.1f}",
            flush=True,
        )
    finally:
        dist.destroy_process_group()  # Release NCCL resources on every exit path.


if __name__ == "__main__":
    main()

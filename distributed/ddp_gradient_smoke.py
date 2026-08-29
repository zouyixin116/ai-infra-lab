#!/usr/bin/env python3
"""Show how DDP averages gradients from different rank-local inputs."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def main() -> None:
    # torchrun assigns one process to each local GPU.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # NCCL provides the GPU collectives used by DDP during backward.
    dist.init_process_group(backend="nccl", init_method="env://")  # Join all ranks into one NCCL process group.
    try:
        # Both ranks begin with the same one-parameter model: y = w * x, w = 1.
        model = torch.nn.Linear(1, 1, bias=False, device=device)
        with torch.no_grad():
            model.weight.fill_(1.0)  # w = 1

        # Each rank deliberately receives a different local example.
        # x      = [[1.0]] on cuda:0
        # target = [[0.0]] on cuda:0
        x = torch.tensor([[float(rank + 1)]], device=device)
        target = torch.zeros_like(x)

        # First compute the gradient without DDP synchronization. For squared
        # error, rank 0 gets grad=2 and rank 1 gets grad=8.
        local_loss = torch.nn.functional.mse_loss(model(x), target)
        local_loss.backward()  # Plain PyTorch backward: no cross-rank gradient synchronization.
        local_gradient = model.weight.grad.item()
        model.zero_grad(set_to_none=True)

        # DDP broadcasts rank 0's parameters at construction and registers
        # backward hooks that all-reduce gradients across all ranks.
        ddp_model = DistributedDataParallel(  # Broadcast rank 0 parameters and install gradient-sync hooks.
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

        optimizer.zero_grad(set_to_none=True)
        synchronized_loss = torch.nn.functional.mse_loss(ddp_model(x), target)
        synchronized_loss.backward()  # Trigger DDP hooks and NCCL collectives to average gradients across ranks. (AllReduce)

        # DDP averages gradients by world size, so (2 + 8) / 2 = 5 on both ranks.
        synchronized_gradient = ddp_model.module.weight.grad.item()
        weight_before_step = ddp_model.module.weight.item()  # backward() computes .grad but does not modify the weight.
        optimizer.step()  # Every rank applies the same averaged gradient to its own weights, keeping model replicas identical.
        weight_after_step = ddp_model.module.weight.item()

        print(
            f"rank={rank}, x={x.item():.1f}, "
            f"local_grad={local_gradient:.1f}, "
            f"ddp_grad={synchronized_gradient:.1f}, "
            f"weight_before_step={weight_before_step:.1f}, "
            f"weight_after_step={weight_after_step:.1f}",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

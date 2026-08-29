#!/usr/bin/env python3
"""Show how model replicas diverge when ranks do not synchronize gradients."""

import os

import torch


def main() -> None:
    # torchrun still launches two independent workers, but no process group or
    # DDP wrapper is created, so there is no cross-rank communication.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)  # Bind this worker to its assigned GPU.
    device = torch.device("cuda", local_rank)

    model = torch.nn.Linear(1, 1, bias=False, device=device)
    with torch.no_grad():
        model.weight.fill_(1.0)  # Give both independent replicas the same starting weight.

    # Rank 0 receives x=1, while rank 1 receives x=2.
    x = torch.tensor([[float(rank + 1)]], device=device)
    target = torch.zeros_like(x)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()  # Compute only this rank's local gradient; no NCCL collective runs.

    local_gradient = model.weight.grad.item()
    weight_before_step = model.weight.item()
    optimizer.step()  # Apply a different local gradient on each rank, causing divergence.
    weight_after_step = model.weight.item()

    print(
        f"rank={rank}, x={x.item():.1f}, "
        f"local_grad={local_gradient:.1f}, "
        f"weight_before_step={weight_before_step:.1f}, "
        f"weight_after_step={weight_after_step:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Show how DistributedSampler partitions a dataset between ranks."""

import os

from torch.utils.data import DataLoader, DistributedSampler


def main() -> None:
    # torchrun gives every worker its distributed identity.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Each integer represents one dataset example.
    dataset = list(range(8))

    # DistributedSampler assigns a non-overlapping subset to this rank.
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    # DataLoader turns this rank's assigned examples into local batches.
    loader = DataLoader(dataset, batch_size=2, sampler=sampler)

    assigned_indices = list(iter(sampler))
    local_batches = [batch.tolist() for batch in loader]
    print(
        f"rank={rank}, assigned_indices={assigned_indices}, "
        f"local_batches={local_batches}",
        flush=True,
    )


if __name__ == "__main__":
    main()

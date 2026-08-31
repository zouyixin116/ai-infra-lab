#!/usr/bin/env python3
"""Verify column- and row-parallel linear layers with NCCL collectives."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("this smoke test requires exactly two processes")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    try:
        # Build deterministic CPU tensors so every rank derives identical shards.
        x_cpu = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        weight_cpu = torch.arange(1.0, 25.0, dtype=torch.float32).reshape(4, 6)
        expected_cpu = x_cpu @ weight_cpu

        # Column parallelism stores only this rank's output-feature columns.
        column_shard = weight_cpu.chunk(world_size, dim=1)[rank].to(device)
        x = x_cpu.to(device)
        local_column_output = x @ column_shard

        gathered_columns = [torch.empty_like(local_column_output) for _ in range(world_size)]
        # The next consumer needs the complete output, so NCCL gathers every column shard.
        dist.all_gather(gathered_columns, local_column_output)
        column_output = torch.cat(gathered_columns, dim=1)

        # Row parallelism consumes matching input and weight shards on each rank.
        input_shard = x_cpu.chunk(world_size, dim=1)[rank].to(device)
        row_shard = weight_cpu.chunk(world_size, dim=0)[rank].to(device)
        row_partial_output = input_shard @ row_shard
        row_output = row_partial_output.clone()
        # Each rank computed a partial sum; NCCL combines it into the complete output.
        dist.all_reduce(row_output, op=dist.ReduceOp.SUM)

        # A Transformer MLP pairs a column-parallel expansion with a row-parallel
        # projection. The intermediate activation stays sharded between the layers.
        up_weight_cpu = torch.arange(-12.0, 12.0, dtype=torch.float32).reshape(4, 6)
        down_weight_cpu = torch.arange(1.0, 25.0, dtype=torch.float32).reshape(6, 4)
        expected_mlp_cpu = torch.relu(x_cpu @ up_weight_cpu) @ down_weight_cpu

        up_shard = up_weight_cpu.chunk(world_size, dim=1)[rank].to(device)
        down_shard = down_weight_cpu.chunk(world_size, dim=0)[rank].to(device)
        local_hidden = torch.relu(x @ up_shard)
        mlp_output = local_hidden @ down_shard
        # Row-parallel projection combines partial sums only after the local activation.
        dist.all_reduce(mlp_output, op=dist.ReduceOp.SUM)

        expected = expected_cpu.to(device)
        column_error = (column_output - expected).abs().max().item()
        row_error = (row_output - expected).abs().max().item()
        mlp_error = (
            mlp_output - expected_mlp_cpu.to(device)
        ).abs().max().item()

        for reporting_rank in range(world_size):
            dist.barrier()
            if rank == reporting_rank:
                print(
                    f"rank={rank}, device={device}, "
                    f"column_weight_shard={tuple(column_shard.shape)}, "
                    f"column_local_output={tuple(local_column_output.shape)}, "
                    f"row_input_shard={tuple(input_shard.shape)}, "
                    f"row_weight_shard={tuple(row_shard.shape)}",
                    flush=True,
                )
                print(
                    f"rank={rank}, row_partial_output_before_all_reduce=\n"
                    f"{row_partial_output.cpu()}",
                    flush=True,
                )
                print(
                    f"rank={rank}, mlp_up_weight_shard={tuple(up_shard.shape)}, "
                    f"mlp_local_hidden={tuple(local_hidden.shape)}, "
                    f"mlp_down_weight_shard={tuple(down_shard.shape)}",
                    flush=True,
                )

        if rank == 0:
            print(f"reference_output=\n{expected_cpu}", flush=True)
            print(f"column_parallel_output=\n{column_output.cpu()}", flush=True)
            print(f"row_parallel_output=\n{row_output.cpu()}", flush=True)
            print(f"dense_mlp_output=\n{expected_mlp_cpu}", flush=True)
            print(f"tensor_parallel_mlp_output=\n{mlp_output.cpu()}", flush=True)
            print(
                f"column_max_abs_error={column_error:.1f}, "
                f"row_max_abs_error={row_error:.1f}, "
                f"mlp_max_abs_error={mlp_error:.1f}",
                flush=True,
            )

        if column_error != 0.0 or row_error != 0.0 or mlp_error != 0.0:
            raise RuntimeError(
                "tensor-parallel outputs did not match the unsharded reference"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

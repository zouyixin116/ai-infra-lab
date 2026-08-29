# Experiment: Stage 2 multi-GPU DDP baseline

## Question

How much global token throughput does two-GPU DistributedDataParallel provide
relative to the Stage 1 single-GPU baseline, and are model parameters kept in
sync across ranks?

## Fixed configuration

- Model and dataset: same as Stage 1
- GPUs: two identical visible GPUs, one process per GPU
- Backend: NCCL via `torchrun`
- Precision: bfloat16
- Per-GPU batch size: 2 (global batch size 4)
- Sequence length: 256 tokens
- Measured steps: 20 after 2 warmup steps
- Optimizer: AdamW, learning rate 2e-5
- Seed: 42

DDP averages gradients, so this run changes the global batch size relative to
Stage 1. It is a systems throughput baseline, not a model-quality comparison.

## Pod setup and exact command

```bash
python -m pip install torch transformers datasets sentencepiece safetensors

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  distributed/ddp_train.py \
  --batch-size 2 \
  --checkpoint-dir outputs/stage2_ddp_checkpoint \
  --output results/stage2_ddp.json
```

Run on a host with two identical CUDA GPUs. Rank 0 writes the checkpoint and
JSON. Success requires identical parameter fingerprints across ranks and a
finite-loss checkpoint reload.

## Metrics and result convention

- `mean_loss_per_step`: loss averaged across ranks.
- `rank0_step_time_ms`: synchronized rank-0 update latency.
- `elapsed_seconds_max_rank`: slowest-rank measured duration.
- `tokens_per_second_global`: tokens summed across ranks divided by the
  slowest-rank duration.
- `peak_pytorch_memory_bytes_max_rank`: maximum allocator peak across ranks.

Raw measurements belong in `results/stage2_ddp.json`. Do not commit invented
measurements. Record GPU model, software versions, results, interpretation,
and limitations here after the real run.

## Results

The runs used two RTX 4090 GPUs with PyTorch 2.8.0+cu128. The primary DDP
run completed all 20 measured steps, kept the model replicas synchronized,
saved the optimizer state, and passed checkpoint reload verification.

### Same-host comparison

The single-GPU control was rerun on the same two-GPU host with only GPU 0
visible. This removes the largest cross-Pod differences from the scaling
comparison.

| Configuration | Global batch | Mean step time | Tokens/s | Peak memory |
|---|---:|---:|---:|---:|
| 1 GPU, batch 2 | 2 | 130.102 ms | 3,305.876 | 10.318 GiB |
| 2 GPUs, batch 2/GPU | 4 | 311.696 ms | 2,694.610 | 12.364 GiB/GPU max |
| 2 GPUs, batch 1/GPU | 2 | 312.184 ms | 1,359.134 | 12.339 GiB/GPU max |

The per-GPU-batch-2 run is a weak-scaling comparison because adding the second
GPU also doubles the global batch from 2 to 4:

```text
weak-scaling speedup
= 2-GPU global throughput / same-host 1-GPU throughput
= 2694.610 / 3305.876
= 0.815x

weak-scaling efficiency
= 0.815 / 2
= 40.8%
```

Global throughput decreased by 18.5% instead of approaching the ideal 2x
increase. At the measured single-GPU rate, processing the DDP run's 16,798
non-padding tokens would take about 5.08 seconds; the two-GPU run took 6.23
seconds.

### Strong-scaling control

The strong-scaling run used batch size 1 per GPU so both configurations had
global batch size 2. Its speedup was 0.411x and its two-GPU scaling efficiency
was 20.6%. Processing approximately the same token work took about 2.60
seconds on one GPU and 6.24 seconds on two GPUs.

Reducing the DDP local batch from 2 to 1 barely changed mean step latency
(311.696 ms versus 312.184 ms). Local compute decreased, but every backward
still synchronized gradients for the same 1.1-billion-parameter model. The
BF16 gradient payload is approximately 2.05 GiB per backward regardless of
batch size.

### Interpretation

This host produced correct but inefficient DDP training: communication and
synchronization overhead exceeded the parallel-compute benefit. The NCCL debug
log records that peer-to-peer communication was disabled between the GPUs and
that NCCL used a shared-memory path. It also records the initial approximately
2.05 GiB gradient all-reduce and the reducer rebuilding the gradients into 46
buckets after the first backward.

The measurements establish that this workload is communication-bound on this
host, but they do not prove that all lost performance comes from PCIe or the
disabled P2P path. NCCL synchronization, bucket overhead, GPU topology, rank
imbalance, CPU scheduling, clocks, power state, and background load can also
contribute. Separating those effects would require a communication bandwidth
test or a profiler timeline.

Raw artifacts:

- `results/stage2_ddp.json`: weak-scaling DDP benchmark.
- `results/stage2_same_host_single_gpu.json`: same-host single-GPU control.
- `results/stage2_strong_scaling.json`: fixed-global-batch control.
- `results/stage2_nccl_debug.json`: short debug-enabled integration result.
- `results/stage2_nccl_debug.log`: raw DDP and NCCL diagnostic log.

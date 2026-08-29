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

Pending a run on a two-GPU CUDA host.

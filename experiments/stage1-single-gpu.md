# Experiment: Stage 1 single-GPU training baseline

## Question

How do batch sizes 1 and 2 affect loss, optimizer-step latency, token
throughput, and peak GPU memory when fully fine-tuning TinyLlama 1.1B on one
RTX 4090?

## Fixed configuration

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Dataset: first 256 training examples from `roneneldan/TinyStories`
- GPU: exactly one visible RTX 4090
- Precision: bfloat16
- Sequence length: 256 tokens, padded/truncated
- Training: PyTorch forward, causal-LM loss, backward, AdamW step
- Measured steps: 20 after 2 warmup steps
- Seed: 42

This is a systems baseline. Do not tune hyperparameters and do not add
DeepSpeed, FSDP, gradient accumulation, or distributed execution.

## Pod setup and exact commands

```bash
python -m pip install torch transformers datasets sentencepiece safetensors

CUDA_VISIBLE_DEVICES=0 python training/single_gpu_train.py \
  --batch-size 1 \
  --checkpoint-dir outputs/stage1_bs1_checkpoint \
  --output results/stage1_bs1.json

CUDA_VISIBLE_DEVICES=0 python training/single_gpu_train.py \
  --batch-size 2 \
  --checkpoint-dir outputs/stage1_bs2_checkpoint \
  --output results/stage1_bs2.json
```

Each run saves model/tokenizer files plus `optimizer_state.pt`, destroys the
in-memory model, reloads the checkpoint, compares a parameter fingerprint, and
runs a finite-loss forward pass. Success is `checkpoint.reload_verified: true`.

## Metrics and result convention

- `loss_per_step`: causal language-model loss for every measured step.
- `step_time_ms`: synchronized forward + loss + backward + optimizer time.
- `tokens_per_second`: non-padding input tokens divided by measured time.
- `peak_pytorch_memory_bytes`: peak allocator memory after model creation and
  before checkpoint reload.

Raw GPU results belong in `results/stage1_bs1.json` and
`results/stage1_bs2.json`. Do not commit invented or non-4090 measurements.

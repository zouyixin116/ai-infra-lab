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

## Results

Both runs were measured on an RTX 4090 with PyTorch 2.8.0+cu128. Both saved
the optimizer state and passed the model checkpoint reload check.

| Batch size | Mean loss | Mean step time | Tokens/s | Peak memory |
|---:|---:|---:|---:|---:|
| 1 | 1.5843 | 125.615 ms | 1,725.914 | 10.292 GiB |
| 2 | 1.5472 | 125.728 ms | 3,420.880 | 10.318 GiB |

## Explanation and conclusion

Increasing batch size from 1 to 2 increased measured token throughput by
98.2%, while mean step time increased by only 0.09%. Peak allocated memory
rose by 26.3 MiB (0.25%). For this short, padded workload, batch size 2 kept
the GPU busier without a material latency or allocator-memory penalty.

Loss varied from batch to batch because each step used different stories.
Mean measured loss was 1.5843 for batch size 1 and 1.5472 for batch size 2;
these 20-step runs are too short to interpret that difference as a model
quality result.

## Limitations / follow-up

Token counts exclude padding, so the two runs processed 4,336 and 8,602 real
tokens rather than exact multiples of the padded sequence length. The sample
is intentionally small and each configuration was run once; the results are
a Stage 1 baseline, not a variance study or hyperparameter comparison.

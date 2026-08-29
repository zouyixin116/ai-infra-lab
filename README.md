# AI Infra Lab

A hands-on portfolio lab for implementing and measuring the main pieces of a
modern LLM training and inference stack. The project favors small,
reproducible experiments and systems understanding over production
completeness.

## Method

- **BFS mainline:** get each important component working end to end, from a
  CUDA smoke test through training, serving, Kubernetes, observability, and a
  bottleneck investigation.
- **DFS branches:** selectively investigate high-value topics such as inference
  performance, NCCL communication, reliability, or GPU scheduling. DFS work
  must not block the BFS mainline.
- New technology is classified as **required for BFS**, **useful DFS
  candidate**, or **theory-only / out of scope**. The default is to continue
  BFS without expanding the stack.

Stages 0 through 2 are implemented: the environment/benchmark harness, a
single-GPU training baseline, and a multi-GPU DDP baseline. No benchmark
numbers are committed unless they were produced by an actual run.

## Repository layout

```text
ai-infra-lab/
├── training/       # Single- and multi-GPU training code
├── serving/        # vLLM server and load generator
├── distributed/    # DDP and communication experiments
├── k8s/            # Kubernetes manifests
├── monitoring/     # Prometheus, Grafana, and DCGM configuration
├── experiments/    # Experiment plans, conclusions, and the result template
├── results/        # Machine-readable JSON/CSV outputs and plots
├── validate_environment.py
├── cuda_smoke_test.py
└── README.md
```

## Stage 0: run it

Prerequisites are Python 3.8+ and a CUDA-enabled PyTorch installation. Install
PyTorch using the command appropriate for your CUDA/driver environment from
the official PyTorch installation guide.

From the repository root:

```bash
python validate_environment.py
python validate_environment.py --output results/environment.json

python cuda_smoke_test.py
python cuda_smoke_test.py --size 4096 --iterations 20 \
  --output results/cuda_smoke.json
```

The validation command exits nonzero if PyTorch cannot be imported or CUDA is
unavailable. The smoke test allocates two square matrices on the GPU, performs
warmup and timed matrix multiplications, verifies the output, and reports
latency, approximate throughput, and peak PyTorch GPU memory. Reduce `--size`
if the allocation does not fit on the GPU.

Copy [experiments/experiment-template.md](experiments/experiment-template.md)
for each experiment. Keep exact commands/configuration in the experiment note
and save raw CSV/JSON data under `results/`.

## Stage 1: single-GPU training baseline

Run the TinyLlama/TinyStories baseline twice on one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python training/single_gpu_train.py \
  --batch-size 1 --checkpoint-dir outputs/stage1_bs1_checkpoint \
  --output results/stage1_bs1.json

CUDA_VISIBLE_DEVICES=0 python training/single_gpu_train.py \
  --batch-size 2 --checkpoint-dir outputs/stage1_bs2_checkpoint \
  --output results/stage1_bs2.json
```

The JSON reports record per-step loss and latency, tokens per second, peak
PyTorch GPU memory, and checkpoint reload verification. See
[experiments/stage1-single-gpu.md](experiments/stage1-single-gpu.md) for the
fixed experiment definition and Pod setup.

Compare the original BF16 weights with a saved continued-training checkpoint
on CPU:

```bash
python training/compare_weights.py \
  --trained-model outputs/weight_comparison_checkpoint \
  --output results/weight_comparison.json
```

The report includes the global changed-element fraction and weight deltas,
plus per-tensor statistics ranked by relative L2 change.

## Stage 2: multi-GPU DDP baseline

Before training, verify process-to-GPU binding and NCCL communication:

```bash
torchrun --standalone --nproc-per-node=2 distributed/ddp_smoke.py
```

Both ranks should report an all-reduce result of `3.0` for a two-process run.
Then run one process per GPU on a host with two identical CUDA GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  distributed/ddp_train.py \
  --batch-size 2 \
  --checkpoint-dir outputs/stage2_ddp_checkpoint \
  --output results/stage2_ddp.json
```

The rank-0 JSON reports loss averaged across ranks, global token throughput,
slowest-rank elapsed time, and maximum per-rank peak memory. The run also
checks that all ranks have identical parameter fingerprints before rank 0
saves and reloads the checkpoint. See
[experiments/stage2-ddp.md](experiments/stage2-ddp.md) for the fixed experiment
definition and result convention.

# Experiment: Stage 3 vLLM serial serving baseline

## Question

Can vLLM serve TinyLlama through its OpenAI-compatible streaming API correctly,
and what are the serial time to first token (TTFT), end-to-end latency, and
output-token throughput on one GPU?

This is the first Stage 3 concept. It establishes that the service and load
generator work together before concurrency or continuous batching is studied.

## Fixed configuration

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Server: vLLM OpenAI-compatible API on one visible GPU
- API: streaming `/v1/completions`
- Prompt set: eight fixed short prompts, selected round-robin
- Generation: at most 64 output tokens, temperature 0
- Measured requests: 100 after 2 serial warmup requests
- Concurrency: 1
- Random seed: server seed 42

The experiment is a serving-systems baseline, not a model-quality evaluation.

## Pod setup and exact commands

Install vLLM in a fresh PyTorch/CUDA Pod, then start the server:

```bash
python -m pip install vllm

CUDA_VISIBLE_DEVICES=0 vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype bfloat16 \
  --seed 42 \
  --host 0.0.0.0 \
  --port 8000
```

From a second shell on the same Pod, run the serial baseline:

```bash
python serving/load_test.py --concurrency 1 \
  --output results/stage3_c1.json
```

The load generator first queries `/v1/models` and refuses to benchmark if the
requested model is not advertised by the server.

## Metrics and result convention

- `ttft_ms_p50/p95`: request start to first non-empty streamed text chunk.
- `latency_ms_p50/p95`: request start to completion of the response stream.
- `request_throughput_per_second`: successful requests divided by measured
  wall-clock time.
- `output_tokens_per_second`: server-reported completion tokens divided by
  measured wall-clock time.
- `success_rate`: successful requests divided by all measured requests.

Raw measurements belong in `results/stage3_c1.json`. Do not commit invented
measurements. Record the GPU, vLLM version, results, interpretation, and
limitations here after the real run.

## Evidence required

The concept is complete only when an actual GPU run provides all of the
following evidence:

- `/v1/models` advertises the requested model;
- all 100 measured requests succeed and produce streamed text;
- the server reports completion-token counts for every request;
- TTFT, end-to-end latency, and output-token throughput are finite and positive.

These observations prove that the model can be served through this API and that
the client can measure a serial workload. They do not establish a concurrency
benefit, demonstrate continuous batching, measure maximum capacity, or isolate
GPU time from HTTP, scheduling, and tokenization overhead.

## Results

Pending an actual GPU run. The local mock-server test validates client-side SSE
parsing and metric aggregation only; it is not evidence that vLLM or CUDA works.

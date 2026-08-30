# Experiment: Stage 3 vLLM serving and load baseline

## Questions

Can vLLM serve TinyLlama through its OpenAI-compatible streaming API, how
does concurrent load change throughput and client-observed latency, and where
does this single-GPU client/server workload begin to degrade?

Stage 3 establishes the serving and load-generation baseline. KV-cache and
continuous-batching internals belong to Stage 4; multi-GPU tensor parallelism
belongs to Stage 5.

## Hardware and software

- GPU used by vLLM: one NVIDIA GeForce RTX 4090, 24 GiB
- vLLM: 0.28.0
- PyTorch: 2.13.0+cu130
- CUDA runtime reported by PyTorch: 13.0
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Precision: bfloat16
- Client and server: same Pod

Installing vLLM upgraded the original PyTorch 2.8.0+cu128 environment to
PyTorch 2.13.0+cu130. CUDA remained available after the installation.

## Server and request path

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype bfloat16 \
  --seed 42 \
  --host 0.0.0.0 \
  --port 8000
```

The server binds every container interface on port 8000. Because the load
generator runs in the same Pod, it uses `http://127.0.0.1:8000`. A successful
`GET /v1/models` response verified that the API advertised the requested model
before any measured requests were sent.

The streaming path is:

```text
model generates token IDs
→ tokenizer incrementally decodes text fragments
→ vLLM places a fragment in choices[0]["text"]
→ the API sends it as an SSE data event
```

An SSE event is not a model-token boundary. The load generator therefore uses
the final `usage.completion_tokens` value, rather than the number of received
chunks, to calculate output-token throughput.

## Workload and metrics

- Eight fixed short prompts selected round-robin
- At most 64 output tokens per request
- Temperature 0
- Two serial warmup requests excluded from measurements
- Streaming `/v1/completions`
- Closed-loop load: at most `concurrency` blocking HTTP streams remain active

TTFT ends when the client receives the first non-empty streamed text fragment.
End-to-end latency ends when the response stream completes. Per-request timers
start when a worker begins its HTTP request, so they exclude time waiting in
the client thread-pool queue. Workload wall time includes all submitted
requests and is the denominator for aggregate throughput.

## Serial and moderate-concurrency results

Each variant completed 100 requests and generated 6,400 output tokens with no
failures.

| Concurrency | Wall time | Requests/s | Output tokens/s | TTFT P50 / P95 | Latency P50 / P95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 19.055206 s | 5.247910 | 335.866218 | 16.028 / 18.032 ms | 190.192 / 192.223 ms |
| 4 | 5.804986 s | 17.226570 | 1,102.500458 | 25.982 / 32.523 ms | 231.848 / 239.429 ms |
| 16 | 1.819652 s | 54.955576 | 3,517.156834 | 55.846 / 63.017 ms | 261.408 / 268.703 ms |

C4 delivered 3.283x the C1 output throughput while increasing P50 latency by
21.9%. C16 delivered 10.472x the C1 output throughput while increasing P50
latency by 37.4%. Higher concurrency therefore traded additional TTFT and
completion latency for much higher aggregate throughput.

These measurements are consistent with vLLM advancing multiple active
sequences efficiently, but client-side timing does not expose each engine
iteration's batch composition. It cannot by itself quantify how much of the
gain came specifically from continuous batching.

## Saturation sweep

The higher-concurrency sweep used 512 requests per variant so C16, C32, and
C64 contained enough request waves for a steadier aggregate measurement. Each
variant generated 32,768 output tokens with no failures.

```bash
python serving/load_test.py --requests 512 --concurrency 16 \
  --output results/stage3_saturation_c16.json
python serving/load_test.py --requests 512 --concurrency 32 \
  --output results/stage3_saturation_c32.json
python serving/load_test.py --requests 512 --concurrency 64 \
  --output results/stage3_saturation_c64.json
python serving/load_test.py --requests 512 --concurrency 128 \
  --output results/stage3_saturation_c128.json
python serving/load_test.py --requests 512 --concurrency 256 \
  --output results/stage3_saturation_c256.json
```

| Concurrency | Wall time | Requests/s | Output tokens/s | TTFT P50 / P95 | Latency P50 / P95 |
|---:|---:|---:|---:|---:|---:|
| 16 | 8.122234 s | 63.036845 | 4,034.358065 | 41.702 / 57.913 ms | 251.539 / 263.776 ms |
| 32 | 4.163538 s | 122.972345 | 7,870.230048 | 33.033 / 61.570 ms | 256.194 / 272.014 ms |
| 64 | 2.617233 s | 195.626476 | 12,520.094488 | 83.445 / 117.805 ms | 318.790 / 353.494 ms |
| 128 | 2.055423 s | 249.097194 | 15,942.220404 | 152.975 / 252.965 ms | 490.241 / 550.208 ms |
| 256 | 2.531636 s | 202.240748 | 12,943.407898 | 102.076 / 208.438 ms | 413.096 / 514.371 ms |

Adjacent throughput changes were +95.1%, +59.1%, +27.3%, and -18.8%. C32-C64
was the practical throughput/latency knee: throughput still increased
substantially, but TTFT and completion latency began increasing much faster.
C128 produced the highest measured throughput. C256 took 23.2% longer than
C128 and produced 18.8% less throughput, placing the observed degradation
region between C128 and C256.

This is saturation of the measured client/server system, not a pure vLLM or GPU
capacity limit. At C256 the two request waves also include Python thread
creation, OS scheduling, HTTP/SSE overhead, and CPU competition between the
load generator and server in the same Pod. Those factors can also explain why
C256 recorded lower per-request latency than C128 despite worse wall time and
throughput.

## Raw artifacts

- `results/stage3_c1.json`
- `results/stage3_c4.json`
- `results/stage3_c16.json`
- `results/stage3_saturation_c16.json`
- `results/stage3_saturation_c32.json`
- `results/stage3_saturation_c64.json`
- `results/stage3_saturation_c128.json`
- `results/stage3_saturation_c256.json`

## Conclusions and limitations

Stage 3 demonstrated an end-to-end single-GPU serving system, a repeatable
streaming load generator, successful concurrent load, and measured
throughput/latency tradeoffs. All measured requests succeeded.

The prompts were short, outputs were fixed at 64 tokens, and client and server
shared one Pod. The experiment did not measure external network latency, mixed
production traffic, GPU utilization, TPOT, KV-cache pressure, iteration-level
batch composition, or multi-GPU inference. Those omissions must not be inferred
from the client-side throughput results.

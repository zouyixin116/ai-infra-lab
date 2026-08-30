#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible streaming completions endpoint."""

import argparse
import concurrent.futures
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PROMPTS = (
    # Reuse a fixed prompt set so concurrency is the main experimental variable.
    # Round-robin selection also prevents every request from sharing one prefix.
    "Once upon a time, a small robot discovered",
    "Explain why the sky appears blue in simple terms:",
    "Write a short story about a brave mouse:",
    "The most important lesson I learned was",
    "Describe a peaceful morning in the mountains:",
    "List three ways to help a friend who feels sad:",
    "In a village beside the sea, there lived",
    "What makes a good team? A good team",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Continue generation until max-tokens even if EOS is sampled.",
    )
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.max_tokens) <= 0:
        parser.error("requests, concurrency, and max-tokens must be positive")
    if args.warmup_requests < 0 or args.timeout <= 0:
        parser.error("warmup-requests must be non-negative and timeout must be positive")
    return args


def percentile(values, quantile):
    """Return a linearly interpolated percentile for a non-empty sequence."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def check_server(base_url, timeout):
    """Return model IDs advertised by the server's discovery endpoint."""
    # Fail before warmup if the HTTP service is unavailable or the requested
    # model name cannot be used by the completions endpoint.
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/models")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [item["id"] for item in payload.get("data", [])]


def run_request(base_url, model, prompt, max_tokens, timeout, ignore_eos=False):
    """Run one streamed completion and return client-observed measurements."""
    # Usage is requested explicitly because SSE chunks are transport events:
    # one event can contain zero, one, or multiple generated tokens.
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": ignore_eos,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at = None
    completion_tokens = None
    chunks = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        # TTFT ends at the first non-empty generated text event, not at the HTTP
        # headers or metadata event that a streaming server may send earlier.
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            # Example payload: {"choices": [{"text": "Once"}]}. The text is
            # an incremental fragment and does not define a one-token boundary.
            choices = event.get("choices") or []
            if choices and choices[0].get("text"):
                chunks += 1
                if first_token_at is None:
                    first_token_at = time.perf_counter()
            # vLLM sends the final token count in a usage event near the end of
            # the stream. This is authoritative for output-token throughput.
            # Example: {"choices": [], "usage": {"completion_tokens": 64}}
            usage = event.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
    finished = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("stream completed without a generated text chunk")
    if completion_tokens is None:
        raise RuntimeError("server did not return streaming token usage")
    return {
        "latency_seconds": finished - started,
        "ttft_seconds": first_token_at - started,
        "completion_tokens": completion_tokens,
        "chunks": chunks,
    }


def execute(args):
    """Run warmup and measured workloads, then build the benchmark report."""
    available_models = check_server(args.base_url, args.timeout)
    if args.model not in available_models:
        raise RuntimeError(
            "requested model {!r} not advertised by server; available: {}".format(
                args.model, ", ".join(available_models) or "none"
            )
        )

    # Warmup requests exercise lazy server work before timing. They remain
    # serial and are intentionally excluded from every reported metric.
    for index in range(args.warmup_requests):
        run_request(
            args.base_url,
            args.model,
            DEFAULT_PROMPTS[index % len(DEFAULT_PROMPTS)],
            args.max_tokens,
            args.timeout,
            args.ignore_eos,
        )

    results = []
    errors = []
    # Measure the whole submitted workload once. Summing per-request latency
    # would double-count overlap when a later experiment raises concurrency.
    wall_started = time.perf_counter()
    # Each worker owns one blocking HTTP stream. The worker count therefore
    # bounds client-side in-flight requests without requiring an async library.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        future_to_index = {
            pool.submit(
                run_request,
                args.base_url,
                args.model,
                DEFAULT_PROMPTS[index % len(DEFAULT_PROMPTS)],
                args.max_tokens,
                args.timeout,
                args.ignore_eos,
            ): index
            for index in range(args.requests)
        }
        # Completion order differs under concurrency, so retain request_index
        # and restore submission order before writing reproducible raw output.
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                result["request_index"] = index
                results.append(result)
            except Exception as exc:  # Keep the run report even when requests fail.
                errors.append({"request_index": index, "error": str(exc)})
    wall_seconds = time.perf_counter() - wall_started
    results.sort(key=lambda item: item["request_index"])

    latencies = [item["latency_seconds"] for item in results]
    ttfts = [item["ttft_seconds"] for item in results]
    total_tokens = sum(item["completion_tokens"] for item in results)
    # Throughput uses the shared workload wall time, whereas latency percentiles
    # describe individual requests. They answer different serving questions.
    metrics = {
        "successful_requests": len(results),
        "failed_requests": len(errors),
        "success_rate": len(results) / args.requests,
        "wall_time_seconds": round(wall_seconds, 6),
        "request_throughput_per_second": round(len(results) / wall_seconds, 6),
        "output_tokens": total_tokens,
        "output_tokens_per_second": round(total_tokens / wall_seconds, 6),
    }
    if results:
        metrics.update(
            {
                "ttft_ms_p50": round(percentile(ttfts, 0.50) * 1000, 3),
                "ttft_ms_p95": round(percentile(ttfts, 0.95) * 1000, 3),
                "latency_ms_p50": round(percentile(latencies, 0.50) * 1000, 3),
                "latency_ms_p95": round(percentile(latencies, 0.95) * 1000, 3),
            }
        )
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "server": args.base_url,
        "model": args.model,
        "configuration": {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "ignore_eos": args.ignore_eos,
            "warmup_requests": args.warmup_requests,
            "timeout_seconds": args.timeout,
            "streaming": True,
            "temperature": 0,
        },
        "metrics": metrics,
        "requests": results,
        "errors": errors,
    }


def main():
    args = parse_args()
    try:
        report = execute(args)
    except (OSError, ValueError, RuntimeError, urllib.error.HTTPError) as exc:
        print("ERROR: {}".format(exc))
        return 1
    rendered = json.dumps(report, indent=2)
    print(rendered)
    # Write failed-request reports too, so partial runs retain diagnostic evidence.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["metrics"]["failed_requests"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

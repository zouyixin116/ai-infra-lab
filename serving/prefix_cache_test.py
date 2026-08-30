#!/usr/bin/env python3
"""Measure cold and warm prefix-cache behavior through a vLLM API."""

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


METRIC_NAMES = {
    "vllm:prefix_cache_queries_total": "queries",
    "vllm:prefix_cache_hits_total": "hits",
    "vllm:prompt_tokens_cached_total": "cached",
}
PROMPT_SENTENCE = (
    "The observatory records the color and position of every distant star. "
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument("--repetitions", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--marker",
        help="Unique prompt prefix; defaults to a new UUID for a cold first request.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.repetitions, args.max_tokens) <= 0 or args.timeout <= 0:
        parser.error("repetitions, max-tokens, and timeout must be positive")
    return args


def read_metrics(base_url, timeout):
    """Read the prefix-cache counters used to isolate each request."""
    with urllib.request.urlopen(
        base_url.rstrip("/") + "/metrics", timeout=timeout
    ) as response:
        rendered = response.read().decode("utf-8")

    metrics = {}
    for line in rendered.splitlines():
        if not line or line.startswith("#"):
            continue
        name_and_labels, value = line.rsplit(" ", 1)
        metric_name = name_and_labels.split("{", 1)[0]
        if metric_name in METRIC_NAMES:
            metrics[METRIC_NAMES[metric_name]] = float(value)
        elif metric_name == "vllm:prompt_tokens_by_source_total":
            if 'source="local_compute"' in name_and_labels:
                metrics["local_compute"] = float(value)
            elif 'source="local_cache_hit"' in name_and_labels:
                metrics["local_cache_hit"] = float(value)

    required = set(METRIC_NAMES.values()) | {"local_compute", "local_cache_hit"}
    missing = sorted(required - metrics.keys())
    if missing:
        raise RuntimeError("metrics endpoint omitted: {}".format(", ".join(missing)))
    return metrics


def run_request(base_url, model, prompt, max_tokens, timeout):
    """Send one non-streaming completion and retain server-reported usage."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    return {
        "latency_seconds": time.perf_counter() - started,
        "usage": result.get("usage"),
    }


def counter_delta(after, before):
    """Subtract adjacent snapshots so unrelated historical traffic is excluded."""
    return {
        key: after[key] - before[key]
        for key in sorted(set(before) | set(after))
    }


def execute(args):
    marker = args.marker or "stage4-prefix-cache-{}".format(uuid.uuid4())
    prompt = (
        marker
        + ". "
        + PROMPT_SENTENCE * args.repetitions
        + "Summarize the observation in one word:"
    )

    before = read_metrics(args.base_url, args.timeout)
    first_request = run_request(
        args.base_url, args.model, prompt, args.max_tokens, args.timeout
    )
    after_first = read_metrics(args.base_url, args.timeout)
    second_request = run_request(
        args.base_url, args.model, prompt, args.max_tokens, args.timeout
    )
    after_second = read_metrics(args.base_url, args.timeout)

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "server": args.base_url,
        "model": args.model,
        "configuration": {
            "marker": marker,
            "repetitions": args.repetitions,
            "prompt_characters": len(prompt),
            "max_tokens": args.max_tokens,
            "temperature": 0,
        },
        "before": before,
        "first_request": first_request,
        "first_request_delta": counter_delta(after_first, before),
        "second_request": second_request,
        "second_request_delta": counter_delta(after_second, after_first),
        "after_second": after_second,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

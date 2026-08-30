#!/usr/bin/env python3
"""Measure engine-step token sizes for one cold long-prompt request."""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


PROMPT_SENTENCE = (
    "The archive indexes every instrument, observation, and calibration record. "
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
    parser.add_argument("--marker")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.repetitions, args.max_tokens) <= 0 or args.timeout <= 0:
        parser.error("repetitions, max-tokens, and timeout must be positive")
    return args


def parse_metrics(rendered):
    """Extract counters and cumulative engine-step histogram buckets."""
    snapshot = {"iteration_buckets": {}}
    scalar_names = {
        "vllm:prompt_tokens_total": "prompt_tokens",
        "vllm:generation_tokens_total": "generation_tokens",
        "vllm:iteration_tokens_total_count": "iteration_count",
        "vllm:iteration_tokens_total_sum": "iteration_tokens",
    }
    for line in rendered.splitlines():
        if not line or line.startswith("#"):
            continue
        name_and_labels, value = line.rsplit(" ", 1)
        metric_name = name_and_labels.split("{", 1)[0]
        if metric_name in scalar_names:
            snapshot[scalar_names[metric_name]] = float(value)
        elif metric_name == "vllm:iteration_tokens_total_bucket":
            match = re.search(r'le="([^"]+)"', name_and_labels)
            if match:
                snapshot["iteration_buckets"][match.group(1)] = float(value)

    required = set(scalar_names.values())
    missing = sorted(required - snapshot.keys())
    if missing:
        raise RuntimeError("metrics endpoint omitted: {}".format(", ".join(missing)))
    return snapshot


def read_metrics(base_url, timeout):
    with urllib.request.urlopen(
        base_url.rstrip("/") + "/metrics", timeout=timeout
    ) as response:
        return parse_metrics(response.read().decode("utf-8"))


def run_request(base_url, model, prompt, max_tokens, timeout):
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


def snapshot_delta(after, before):
    scalar_keys = (
        "prompt_tokens",
        "generation_tokens",
        "iteration_count",
        "iteration_tokens",
    )
    result = {key: after[key] - before[key] for key in scalar_keys}
    bucket_keys = sorted(
        set(before["iteration_buckets"]) | set(after["iteration_buckets"]),
        key=lambda item: float("inf") if item == "+Inf" else float(item),
    )
    result["iteration_buckets"] = {
        key: after["iteration_buckets"].get(key, 0)
        - before["iteration_buckets"].get(key, 0)
        for key in bucket_keys
    }
    return result


def execute(args):
    marker = args.marker or "stage4-chunked-prefill-{}".format(uuid.uuid4())
    prompt = (
        marker
        + ". "
        + PROMPT_SENTENCE * args.repetitions
        + "Return one word:"
    )
    before = read_metrics(args.base_url, args.timeout)
    request = run_request(
        args.base_url, args.model, prompt, args.max_tokens, args.timeout
    )
    after = read_metrics(args.base_url, args.timeout)
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
        "request": request,
        "metrics_delta": snapshot_delta(after, before),
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

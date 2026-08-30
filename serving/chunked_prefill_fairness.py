#!/usr/bin/env python3
"""Submit a short prompt while a chunked long prefill is in progress."""

import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


LONG_SENTENCE = (
    "The archive indexes every instrument, observation, and calibration record. "
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument("--long-repetitions", type=int, default=80)
    parser.add_argument("--short-delay", type=float, default=0.02)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.long_repetitions <= 0 or args.max_tokens <= 0:
        parser.error("long-repetitions and max-tokens must be positive")
    if args.short_delay < 0 or args.timeout <= 0:
        parser.error("short-delay must be non-negative and timeout must be positive")
    return args


def run_request(base_url, model, prompt, max_tokens, timeout, experiment_started):
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
    finished = time.perf_counter()
    return {
        "submitted_offset_seconds": started - experiment_started,
        "completed_offset_seconds": finished - experiment_started,
        "latency_seconds": finished - started,
        "usage": result.get("usage"),
    }


def execute(args):
    marker = "stage4-prefill-fairness-{}".format(uuid.uuid4())
    long_prompt = (
        marker
        + "-long. "
        + LONG_SENTENCE * args.long_repetitions
        + "Return one word:"
    )
    short_prompt = marker + "-short. Return one word:"

    experiment_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        long_future = pool.submit(
            run_request,
            args.base_url,
            args.model,
            long_prompt,
            args.max_tokens,
            args.timeout,
            experiment_started,
        )
        time.sleep(args.short_delay)
        short_future = pool.submit(
            run_request,
            args.base_url,
            args.model,
            short_prompt,
            args.max_tokens,
            args.timeout,
            experiment_started,
        )
        long_result = long_future.result()
        short_result = short_future.result()

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "server": args.base_url,
        "model": args.model,
        "configuration": {
            "marker": marker,
            "long_repetitions": args.long_repetitions,
            "long_prompt_characters": len(long_prompt),
            "short_prompt_characters": len(short_prompt),
            "short_delay_seconds": args.short_delay,
            "max_tokens": args.max_tokens,
            "temperature": 0,
        },
        "long_request": long_result,
        "short_request": short_result,
        "completion_order": (
            "long_first"
            if long_result["completed_offset_seconds"]
            < short_result["completed_offset_seconds"]
            else "short_first"
        ),
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

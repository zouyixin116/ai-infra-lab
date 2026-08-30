#!/usr/bin/env python3
"""Inspect non-streaming model outputs for the Stage 3 fixed prompts."""

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PROMPTS = (
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
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.timeout <= 0:
        parser.error("max-tokens and timeout must be positive")
    return args


def generate(base_url, model, prompt, max_tokens, timeout):
    """Return one non-streaming completion and its server-reported metadata."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    choice = payload["choices"][0]
    return {
        "prompt": prompt,
        "generated_text": choice["text"],
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage"),
    }


def inspect(args):
    """Run the fixed prompts serially so each output is easy to examine."""
    generations = [
        generate(
            args.base_url,
            args.model,
            prompt,
            args.max_tokens,
            args.timeout,
        )
        for prompt in DEFAULT_PROMPTS
    ]
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "server": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "generations": generations,
    }


def print_report(report):
    for index, generation in enumerate(report["generations"], start=1):
        print("=" * 80)
        print("Request: {}".format(index))
        print("Prompt:\n{}".format(generation["prompt"]))
        print("\nGenerated text:\n{}".format(generation["generated_text"]))
        print("\nFinish reason: {}".format(generation["finish_reason"]))
        print(
            "Usage: {}".format(
                json.dumps(generation["usage"], ensure_ascii=False)
            )
        )


def main():
    args = parse_args()
    try:
        report = inspect(args)
    except (OSError, ValueError, KeyError, IndexError, urllib.error.HTTPError) as exc:
        print("ERROR: {}".format(exc))
        return 1

    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

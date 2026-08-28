#!/usr/bin/env python3
"""Report the Python, PyTorch, CUDA, and NVIDIA GPU environment."""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple


def collect_environment() -> Tuple[Dict, bool]:
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }

    try:
        import torch
    except ImportError as exc:
        report.update(
            {
                "pytorch_importable": False,
                "pytorch_version": None,
                "pytorch_cuda_version": None,
                "cudnn_version": None,
                "cuda_available": False,
                "gpu_count": 0,
                "gpus": [],
                "error": str(exc),
            }
        )
        return report, False

    cuda_available = torch.cuda.is_available()
    report.update(
        {
            "pytorch_importable": True,
            "pytorch_version": torch.__version__,
            "pytorch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": cuda_available,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [],
        }
    )

    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            report["gpus"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "total_memory_gib": round(props.total_memory / 1024**3, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )

    return report, cuda_available


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()

    report, valid = collect_environment()
    rendered = json.dumps(report, indent=2)
    print(rendered)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

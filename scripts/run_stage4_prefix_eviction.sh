#!/usr/bin/env bash
set -euo pipefail

base_url="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
python_command="${PYTHON_COMMAND:-python}"
marker="stage4-prefix-eviction-$(date +%s%N)"

before_output="${1:-results/stage4_prefix_before_pressure.json}"
pressure_metrics="${2:-results/stage4_prefix_eviction_pressure_metrics.log}"
pressure_output="${3:-results/stage4_prefix_eviction_pressure.json}"
after_output="${4:-results/stage4_prefix_after_pressure.json}"

mkdir -p \
  "$(dirname "$before_output")" \
  "$(dirname "$pressure_metrics")" \
  "$(dirname "$pressure_output")" \
  "$(dirname "$after_output")"

# The second identical request establishes that this unique prefix is cached.
"$python_command" serving/prefix_cache_test.py \
  --base-url "$base_url" \
  --marker "$marker" \
  --output "$before_output"

# Oversubscribe the restricted KV pool so evictable zero-reference blocks are reused.
VLLM_BASE_URL="$base_url" PYTHON_COMMAND="$python_command" \
  bash scripts/run_stage4_kv_pressure.sh \
    "$pressure_metrics" \
    "$pressure_output"

# The first request in this report probes whether the original prefix survived.
"$python_command" serving/prefix_cache_test.py \
  --base-url "$base_url" \
  --marker "$marker" \
  --output "$after_output"

echo "Marker: $marker"
echo "Before pressure: $before_output"
echo "Pressure metrics: $pressure_metrics"
echo "Pressure requests: $pressure_output"
echo "After pressure: $after_output"

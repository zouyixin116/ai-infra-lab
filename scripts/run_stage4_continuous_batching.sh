#!/usr/bin/env bash
set -euo pipefail

base_url="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
metrics_output="${1:-results/stage4_continuous_batching_metrics.log}"
early_output="${2:-results/stage4_early_long.json}"
late_output="${3:-results/stage4_late_short.json}"
python_command="${PYTHON_COMMAND:-python}"

mkdir -p \
  "$(dirname "$metrics_output")" \
  "$(dirname "$early_output")" \
  "$(dirname "$late_output")"

sample_metrics() {
  while true; do
    printf '%s ' "$(date +%H:%M:%S.%3N)"
    curl -s "${base_url}/metrics" |
      awk '
        /^vllm:num_requests_running{/ {running=$NF}
        /^vllm:num_requests_waiting{/ {waiting=$NF}
        /^vllm:prompt_tokens_total{/ {prompt=$NF}
        /^vllm:generation_tokens_total{/ {generation=$NF}
        /^vllm:iteration_tokens_total_count{/ {iterations=$NF}
        /^vllm:iteration_tokens_total_sum{/ {iteration_tokens=$NF}
        END {
          printf "running=%s waiting=%s prompt=%s generation=%s iterations=%s iteration_tokens=%s\n",
            running, waiting, prompt, generation, iterations, iteration_tokens
        }'
    sleep 0.05
  done
}

# Capture an idle baseline before either request wave starts.
sample_metrics >"$metrics_output" &
sampler_pid=$!
early_pid=""
cleanup() {
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  if [[ -n "$early_pid" ]]; then
    kill "$early_pid" 2>/dev/null || true
    wait "$early_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

sleep 1
"$python_command" serving/load_test.py \
  --base-url "$base_url" \
  --requests 64 \
  --concurrency 64 \
  --max-tokens 512 \
  --warmup-requests 0 \
  --timeout 180 \
  --output "$early_output" &
early_pid=$!

# Submit the short wave while the first wave is still decoding.
sleep 0.5
"$python_command" serving/load_test.py \
  --base-url "$base_url" \
  --requests 32 \
  --concurrency 32 \
  --max-tokens 64 \
  --warmup-requests 0 \
  --timeout 180 \
  --output "$late_output"

wait "$early_pid"
early_pid=""
sleep 1

cleanup
trap - EXIT

echo "Metrics: $metrics_output"
echo "Early requests: $early_output"
echo "Late requests: $late_output"

#!/usr/bin/env bash
set -euo pipefail

base_url="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
metrics_output="${1:-results/stage4_kv_lifecycle_metrics.log}"
request_output="${2:-results/stage4_kv_lifecycle.json}"
python_command="${PYTHON_COMMAND:-python}"

mkdir -p "$(dirname "$metrics_output")" "$(dirname "$request_output")"

sample_metrics() {
  while true; do
    printf '%s ' "$(date +%H:%M:%S.%3N)"
    curl -s "${base_url}/metrics" |
      awk '
        /^vllm:num_requests_running{/ {running=$NF}
        /^vllm:num_requests_waiting{/ {waiting=$NF}
        /^vllm:kv_cache_usage_perc{/ {kv=$NF}
        /^vllm:prompt_tokens_total{/ {prompt=$NF}
        /^vllm:generation_tokens_total{/ {generation=$NF}
        END {
          printf "running=%s waiting=%s kv=%s prompt=%s generation=%s\n",
            running, waiting, kv, prompt, generation
        }'
    sleep 0.2
  done
}

# Start sampling before the workload so the log contains an idle baseline.
sample_metrics >"$metrics_output" &
sampler_pid=$!
cleanup() {
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
}
trap cleanup EXIT

sleep 1
"$python_command" serving/load_test.py \
  --base-url "$base_url" \
  --requests 128 \
  --concurrency 128 \
  --max-tokens 512 \
  --warmup-requests 0 \
  --timeout 180 \
  --output "$request_output"
sleep 1

cleanup
trap - EXIT

echo "Metrics: $metrics_output"
echo "Requests: $request_output"

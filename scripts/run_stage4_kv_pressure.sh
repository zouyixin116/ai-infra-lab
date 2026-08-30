#!/usr/bin/env bash
set -euo pipefail

base_url="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
metrics_output="${1:-results/stage4_kv_pressure_metrics.log}"
request_output="${2:-results/stage4_kv_pressure.json}"
python_command="${PYTHON_COMMAND:-python}"

mkdir -p "$(dirname "$metrics_output")" "$(dirname "$request_output")"

sample_metrics() {
  while true; do
    printf '%s ' "$(date +%H:%M:%S.%3N)"
    curl -s "${base_url}/metrics" |
      awk '
        /^vllm:num_requests_running{/ {running=$NF}
        /^vllm:num_requests_waiting{/ {waiting=$NF}
        /^vllm:num_requests_waiting_by_reason{.*reason="capacity"/ {capacity=$NF}
        /^vllm:num_requests_waiting_by_reason{.*reason="deferred"/ {deferred=$NF}
        /^vllm:kv_cache_usage_perc{/ {kv=$NF}
        /^vllm:num_preemptions_total{/ {preemptions=$NF}
        /^vllm:prompt_tokens_total{/ {prompt=$NF}
        /^vllm:generation_tokens_total{/ {generation=$NF}
        /^vllm:iteration_tokens_total_count{/ {iterations=$NF}
        /^vllm:iteration_tokens_total_sum{/ {iteration_tokens=$NF}
        END {
          printf "running=%s waiting=%s capacity=%s deferred=%s kv=%s preemptions=%s prompt=%s generation=%s iterations=%s iteration_tokens=%s\n",
            running, waiting, capacity, deferred, kv, preemptions,
            prompt, generation, iterations, iteration_tokens
        }'
    sleep 0.05
  done
}

# Record an idle baseline before the intentionally oversized workload.
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
  --requests 4 \
  --concurrency 4 \
  --max-tokens 1536 \
  --ignore-eos \
  --warmup-requests 0 \
  --timeout 180 \
  --output "$request_output"
sleep 1

cleanup
trap - EXIT

echo "Metrics: $metrics_output"
echo "Requests: $request_output"

import unittest

from serving.chunked_prefill_test import parse_metrics, snapshot_delta


class MetricsTest(unittest.TestCase):
    def test_parses_and_deltas_iteration_histogram(self):
        before = parse_metrics(
            "\n".join(
                (
                    'vllm:prompt_tokens_total{engine="0"} 100',
                    'vllm:generation_tokens_total{engine="0"} 20',
                    'vllm:iteration_tokens_total_bucket{le="1.0"} 4',
                    'vllm:iteration_tokens_total_bucket{le="128.0"} 8',
                    'vllm:iteration_tokens_total_bucket{le="+Inf"} 8',
                    'vllm:iteration_tokens_total_count{engine="0"} 8',
                    'vllm:iteration_tokens_total_sum{engine="0"} 120',
                )
            )
        )
        after = parse_metrics(
            "\n".join(
                (
                    'vllm:prompt_tokens_total{engine="0"} 1124',
                    'vllm:generation_tokens_total{engine="0"} 21',
                    'vllm:iteration_tokens_total_bucket{le="1.0"} 5',
                    'vllm:iteration_tokens_total_bucket{le="128.0"} 17',
                    'vllm:iteration_tokens_total_bucket{le="+Inf"} 17',
                    'vllm:iteration_tokens_total_count{engine="0"} 17',
                    'vllm:iteration_tokens_total_sum{engine="0"} 1145',
                )
            )
        )

        delta = snapshot_delta(after, before)
        self.assertEqual(delta["prompt_tokens"], 1024)
        self.assertEqual(delta["generation_tokens"], 1)
        self.assertEqual(delta["iteration_count"], 9)
        self.assertEqual(delta["iteration_tokens"], 1025)
        self.assertEqual(delta["iteration_buckets"]["1.0"], 1)
        self.assertEqual(delta["iteration_buckets"]["128.0"], 9)


if __name__ == "__main__":
    unittest.main()

import argparse
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from serving.prefix_cache_test import counter_delta, execute


MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class FakePrefixCacheHandler(BaseHTTPRequestHandler):
    requests_seen = 0

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path != "/metrics":
            self.send_error(404)
            return
        first_done = self.requests_seen >= 1
        second_done = self.requests_seen >= 2
        queries = 100 + (64 if first_done else 0) + (64 if second_done else 0)
        hits = 20 + (48 if second_done else 0)
        compute = 80 + (64 if first_done else 0) + (16 if second_done else 0)
        cached = 20 + (48 if second_done else 0)
        body = "\n".join(
            (
                'vllm:prefix_cache_queries_total{{engine="0"}} {}'.format(queries),
                'vllm:prefix_cache_hits_total{{engine="0"}} {}'.format(hits),
                'vllm:prompt_tokens_cached_total{{engine="0"}} {}'.format(cached),
                'vllm:prompt_tokens_by_source_total{{source="local_compute"}} {}'.format(
                    compute
                ),
                'vllm:prompt_tokens_by_source_total{{source="local_cache_hit"}} {}'.format(
                    hits
                ),
            )
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        content_length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(content_length))
        if request["model"] != MODEL or request["max_tokens"] != 1:
            self.send_error(400)
            return
        type(self).requests_seen += 1
        body = json.dumps(
            {
                "choices": [{"text": " blue"}],
                "usage": {"prompt_tokens": 64, "completion_tokens": 1},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class CounterDeltaTest(unittest.TestCase):
    def test_subtracts_matching_snapshots(self):
        self.assertEqual(counter_delta({"hits": 9.0}, {"hits": 4.0}), {"hits": 5.0})


class ExecuteTest(unittest.TestCase):
    def test_cold_then_warm_request(self):
        FakePrefixCacheHandler.requests_seen = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakePrefixCacheHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            args = argparse.Namespace(
                base_url="http://127.0.0.1:{}".format(server.server_port),
                model=MODEL,
                repetitions=2,
                max_tokens=1,
                timeout=2.0,
                marker="test-unique-marker",
            )
            report = execute(args)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        self.assertEqual(report["first_request_delta"]["local_compute"], 64.0)
        self.assertEqual(report["first_request_delta"]["local_cache_hit"], 0.0)
        self.assertEqual(report["second_request_delta"]["local_compute"], 16.0)
        self.assertEqual(report["second_request_delta"]["local_cache_hit"], 48.0)
        self.assertEqual(report["second_request"]["usage"]["prompt_tokens"], 64)


if __name__ == "__main__":
    unittest.main()

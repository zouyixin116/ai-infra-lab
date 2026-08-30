import argparse
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from serving.chunked_prefill_fairness import execute


MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class FakeCompletionHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        content_length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(content_length))
        prompt_tokens = 100 if "-long." in request["prompt"] else 8
        body = json.dumps(
            {
                "choices": [{"text": " done"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 1,
                    "total_tokens": prompt_tokens + 1,
                },
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ExecuteTest(unittest.TestCase):
    def test_records_both_staggered_requests(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeCompletionHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            args = argparse.Namespace(
                base_url="http://127.0.0.1:{}".format(server.server_port),
                model=MODEL,
                long_repetitions=2,
                short_delay=0.01,
                max_tokens=1,
                timeout=2.0,
            )
            report = execute(args)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        self.assertEqual(report["long_request"]["usage"]["prompt_tokens"], 100)
        self.assertEqual(report["short_request"]["usage"]["prompt_tokens"], 8)
        self.assertGreaterEqual(
            report["short_request"]["submitted_offset_seconds"], 0.009
        )


if __name__ == "__main__":
    unittest.main()

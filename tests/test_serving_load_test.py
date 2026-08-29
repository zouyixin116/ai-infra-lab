import argparse
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from serving.load_test import execute, percentile


MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class FakeVllmHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path != "/v1/models":
            self.send_error(404)
            return
        payload = json.dumps({"data": [{"id": MODEL}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        content_length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(content_length))
        if not request.get("stream"):
            self.send_error(400)
            return
        events = (
            {"choices": [{"text": "hello"}], "usage": None},
            {"choices": [], "usage": {"completion_tokens": 3}},
        )
        body = "".join("data: {}\n\n".format(json.dumps(event)) for event in events)
        body += "data: [DONE]\n\n"
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class PercentileTest(unittest.TestCase):
    def test_single_value(self):
        self.assertEqual(percentile([4.0], 0.95), 4.0)

    def test_linear_interpolation(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.95), 3.85)


class ExecuteTest(unittest.TestCase):
    def test_streaming_workload_end_to_end(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVllmHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                args = argparse.Namespace(
                    base_url="http://127.0.0.1:{}".format(server.server_port),
                    model=MODEL,
                    requests=4,
                    concurrency=2,
                    max_tokens=3,
                    warmup_requests=1,
                    timeout=2.0,
                    output=Path(directory) / "unused.json",
                )
                report = execute(args)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        self.assertEqual(report["metrics"]["successful_requests"], 4)
        self.assertEqual(report["metrics"]["failed_requests"], 0)
        self.assertEqual(report["metrics"]["output_tokens"], 12)
        self.assertEqual(len(report["requests"]), 4)


if __name__ == "__main__":
    unittest.main()

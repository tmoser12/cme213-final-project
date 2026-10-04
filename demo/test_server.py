"""CPU-only HTTP tests: python -m unittest demo.test_server -v."""

import http.client
import json
import threading
import unittest

from demo.server import DemoServer, validate_request


class FakeEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.info = {"status": "ready", "backend": "test"}
        self.closed = False

    def prepare(self, messages, limit):
        if messages[0]["content"] == "too long":
            raise ValueError("Context limit exceeded")
        return messages

    def generate(self, tokens, limit, gamma):
        try:
            for mode in ("target", "speculative"):
                yield {"type": "start", "mode": mode}
                yield {"type": "token", "mode": mode, "text": "Hello"}
                yield {"type": "done", "mode": mode, "text": "Hello"}
            yield {"type": "comparison", "identical_tokens": True}
        finally:
            self.closed = True


class TestDemoHTTP(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.server = DemoServer(("127.0.0.1", 0), self.engine)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, payload, extra_headers=None):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        conn.request("POST", "/api/chat", json.dumps(payload),
                     {"Content-Type": "application/json", **(extra_headers or {})})
        response = conn.getresponse()
        status, body = response.status, response.read().decode()
        conn.close()
        return status, body

    def test_stream_completes_and_releases_engine(self):
        status, body = self.request({"messages": [{"role": "user", "content": "Hello"}]})
        self.assertEqual(status, 200)
        self.assertEqual([json.loads(line)["type"] for line in body.splitlines()], ["start", "token", "done", "start", "token", "done", "comparison"])
        events = [json.loads(line) for line in body.splitlines()]
        self.assertEqual([e["mode"] for e in events if e["type"] == "start"],
                         ["target", "speculative"])
        self.assertTrue(self.engine.closed)
        self.assertFalse(self.engine.lock.locked())

    def test_busy_request_does_not_touch_shared_cache(self):
        self.engine.lock.acquire()
        try:
            self.assertEqual(self.request({})[0], 409)
        finally:
            self.engine.lock.release()

    def test_context_error_is_a_normal_http_error(self):
        status, body = self.request({"messages": [{"role": "user", "content": "too long"}]})
        self.assertEqual(status, 400)
        self.assertIn("Context limit", body)
        self.assertFalse(self.engine.lock.locked())

    def test_cross_origin_request_is_rejected(self):
        self.assertEqual(self.request({}, {"Origin": "https://example.com"})[0], 403)

    def test_reject_invalid_message_order_and_limits(self):
        for payload in (
            {"messages": [{"role": "user", "content": "Hi"}], "gamma": 9},
            {"messages": [{"role": "user", "content": "Hi"}], "gamma": True},
            {}, {"messages": [{"role": "system", "content": "Hi"}]},
            {"messages": [{"role": "user", "content": "Hi"}], "max_new_tokens": True},
            {"messages": [{"role": "user", "content": "Hi"}], "max_new_tokens": 257},
            {"messages": [{"role": "user", "content": ""}]},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_request(payload)


if __name__ == "__main__":
    unittest.main()

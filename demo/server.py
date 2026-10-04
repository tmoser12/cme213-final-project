"""A single-user, loopback-only streaming UI for Qwen2Executor.

Run from the repo root: python -m demo.server --model-path models/Qwen2.5-7B-Instruct
No Hugging Face model forward or hosted model API is used by this server.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MAX_BODY_BYTES = 65536
SYSTEM_PROMPT = "You are a helpful assistant. Give clear, concise answers."


# WALKTHROUGH [VALIDATE] — see demo/WALKTHROUGH.md.
# Validate conversation shape and bounded output/draft lengths before touching
# the generation loop. Tokenized context length is checked separately in prepare.
def validate_request(payload: object) -> tuple[list[dict[str, str]], int, int]:
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 40:
        raise ValueError("Send between 1 and 40 conversation messages.")
    clean = []
    for i, message in enumerate(messages):
        role = "user" if i % 2 == 0 else "assistant"
        if not isinstance(message, dict) or message.get("role") != role:
            raise ValueError("Messages must alternate user and assistant, starting with user.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > 16000:
            raise ValueError("Each message must contain 1–16000 characters.")
        clean.append({"role": role, "content": content})
    if clean[-1]["role"] != "user":
        raise ValueError("The last message must be from the user.")
    limit = payload.get("max_new_tokens", 128)
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError("Choose between 1 and 256 output tokens.")
    gamma = payload.get("gamma", 4)
    if type(gamma) is not int or not 1 <= gamma <= 8:
        raise ValueError("Choose between 1 and 8 draft tokens per round.")
    return clean, limit, gamma


# WALKTHROUGH [ENGINE] — see demo/WALKTHROUGH.md.
# Own the tokenizer, both resident models, and the single-request lock. The
# executor owns the forward math; this class owns serving and text conversion.
class KernelEngine:
    def __init__(self, model_path: Path, draft_path: Path, max_context: int):
        import torch
        from transformers import AutoTokenizer, GenerationConfig
        from runtime.buffers import allocate_buffers
        from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig
        from runtime.core.weights import load_weights
        from runtime.executor import Qwen2Executor

        if not torch.cuda.is_available():
            raise RuntimeError("This demo requires a CUDA GPU and compiled target extensions.")
        self.torch = torch
        self.max_context = max_context
        self.lock = threading.Lock()

        # We use tokenizer, generation config, and eos token from huggingface
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        generation = GenerationConfig.from_pretrained(str(model_path), local_files_only=True)
        eos = generation.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        cfg = replace(RuntimeConfig.from_yaml(CONFIG_7B, project_root=ROOT),
                      model_path=str(model_path), max_seq_len=max_context)
        logging.info("Loading %s using custom target kernels", model_path)

        # Our own weight and buffer memory allocation functions are used, and executor modules
        # for the target and draft Qwen models are created.
        weights = load_weights(cfg)
        buffers = allocate_buffers(cfg, batch=1, max_seq_len=max_context, device="cuda")
        self.executor = Qwen2Executor(cfg, weights, buffers, use_cuda_graph=False)
        from runtime.speculative.draft_runner import DraftRunner
        draft_cfg = replace(RuntimeConfig.from_yaml(CONFIG_05B, project_root=ROOT),
                            model_path=str(draft_path), max_seq_len=max_context)
        self.draft_vocab = draft_cfg.vocab_size
        draft_tokenizer = AutoTokenizer.from_pretrained(str(draft_path), local_files_only=True)
        target_vocab = self.tokenizer.get_vocab()
        if any(target_vocab.get(token) != index
               for token, index in draft_tokenizer.get_vocab().items()):
            raise ValueError("Draft and target tokenizer token IDs do not agree.")
        logging.info("Loading %s using custom draft kernels", draft_path)
        draft_weights = load_weights(draft_cfg)
        draft_buffers = allocate_buffers(draft_cfg, batch=1, max_seq_len=max_context, device="cuda")
        self.draft_executor = Qwen2Executor(draft_cfg, draft_weights, draft_buffers,
                                           use_cuda_graph=True)
        self.draft = DraftRunner(self.draft_executor)

        # WALKTHROUGH [WARMUP] — see demo/WALKTHROUGH.md.
        # This startup request imports/uses the AOT modules and lazily captures the
        # draft graph. It is not a benchmark or a compilation step.
        # Exercise both paths and capture the draft decode graph before timing.
        tokens = self.prepare([{"role": "user", "content": "Hello"}], 8)
        for _ in self.generate(tokens, 8, 4):
            pass
        torch.cuda.synchronize()
        props = torch.cuda.get_device_properties(0)
        self.info = {
            "status": "ready", "model": "Qwen2.5-7B-Instruct",
            "backend": "Qwen2Executor · custom target CUDA kernels + cuBLAS",
            "gpu": props.name, "compute_capability": f"{props.major}.{props.minor}",
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "max_context": max_context, "max_new_tokens": 256,
            "decoding": "greedy", "cuda_graph": False,
            "draft_model": "Qwen2.5-0.5B-Instruct", "draft_cuda_graph": True,
            "modes": ["target", "speculative"], "default_gamma": 4,
        }

    # WALKTHROUGH [TOKENIZE] — see demo/WALKTHROUGH.md.
    # The chat template supplies role delimiters and the assistant-generation prefix.
    # Only the resulting int64 IDs move to CUDA; text tokenization remains on the CPU.
    def prepare(self, messages, limit):
        ids = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
            tokenize=True, add_generation_prompt=True,
        )
        if len(ids) + limit > self.max_context:
            raise ValueError(
                f"Conversation uses {len(ids)} tokens; with {limit} output tokens it exceeds "
                f"the {self.max_context}-token demo context. Start a new chat or reduce output length."
            )
        if any(token >= self.draft_vocab for token in ids):
            raise ValueError("This prompt contains a token outside the draft model vocabulary.")
        return self.torch.tensor([ids], dtype=self.torch.int64, device="cuda")

    # WALKTHROUGH [BASELINE] — see demo/WALKTHROUGH.md.
    # The last prompt logit predicts the first output token. Feeding that chosen
    # token through decode predicts the next one; the last emitted token needs no feed.
    def target_chunks(self, tokens, limit):
        torch = self.torch
        decode = self.executor.decode_step
        try:
            logits = self.executor.prefill(tokens)
            for i in range(limit):
                token = int(logits[0, -1].argmax().item())
                yield {"ids": [token]}
                if token in self.eos_ids or i + 1 == limit:
                    break
                logits = decode(torch.tensor([token], dtype=torch.int64, device="cuda"))
        finally:
            self.executor.reset_kv_cache()

    # WALKTHROUGH [TEXT] — see demo/WALKTHROUGH.md.
    # Decode cumulative token IDs, not individual pieces: token boundaries need not
    # be word/UTF-8 boundaries. Each event replaces the visible text for its panel.
    def run_mode(self, mode, chunks, tokens):
        generated = []
        totals = {"accepted": 0, "proposed": 0, "rounds": 0}
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        first_token_time = None
        with closing(chunks):
            for chunk in chunks:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                generated.extend(chunk["ids"])
                for key in totals:
                    totals[key] += chunk.get(key, 0)
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                if not text.endswith("\ufffd"):
                    yield {"type": "token", "mode": mode, "text": text,
                           "output_tokens": len(generated), **totals}
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        yield {
            "type": "done", "mode": mode,
            "text": self.tokenizer.decode(generated, skip_special_tokens=True),
            "finish_reason": "eos" if generated[-1] in self.eos_ids else "length",
            "input_tokens": tokens.shape[1], "output_tokens": len(generated),
            "ttft_ms": round((first_token_time - started) * 1000, 1),
            "elapsed_seconds": elapsed,
            "tokens_per_second": round(len(generated) / elapsed, 2),
            **totals, "_token_ids": generated,
        }

    # WALKTHROUGH [COMPARE] — see demo/WALKTHROUGH.md.
    # Serial execution isolates the two displayed runs. Both start from a fresh
    # cache and identical prompt IDs; exact generated IDs are compared at the end.
    def generate(self, tokens, limit, gamma=4):
        from demo.decoding import speculative_chunks
        results = {}
        try:
            with self.torch.inference_mode():
                for mode in ("target", "speculative"):
                    yield {"type": "start", "mode": mode}
                    chunks = (self.target_chunks(tokens, limit) if mode == "target" else
                              speculative_chunks(self.executor, self.draft, tokens, limit,
                                                 gamma, self.eos_ids))
                    with closing(self.run_mode(mode, chunks, tokens)) as events:
                        for event in events:
                            if event["type"] == "done":
                                results[mode] = dict(event)
                                event.pop("_token_ids")
                            yield event
                target, spec = results["target"], results["speculative"]
                same = target["_token_ids"] == spec["_token_ids"]
                yield {"type": "comparison", "identical_tokens": same,
                       "speedup": round(target["elapsed_seconds"] / spec["elapsed_seconds"], 3),
                       "gamma": gamma}
        finally:
            self.executor.reset_kv_cache()
            self.draft_executor.reset_kv_cache()


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, engine):
        self.engine = engine
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def json_response(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # WALKTHROUGH [GET] — see demo/WALKTHROUGH.md.
    # Serve the HTML and ready-state metadata through the same loopback server.
    # The hosting laptop reaches this socket through SSH forwarding.
    def do_GET(self):
        if self.path == "/health":
            return self.json_response(200, self.server.engine.info)
        if self.path != "/":
            return self.json_response(404, {"error": "Not found"})
        data = Path(__file__).with_name("index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    # WALKTHROUGH [HTTP] — see demo/WALKTHROUGH.md.
    # One shared pair of mutable caches means one generation at a time, even though
    # HTTP handling is threaded. Keep the lock until stream cleanup has completed.
    def do_POST(self):
        if self.path != "/api/chat":
            return self.json_response(404, {"error": "Not found"})
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + self.headers.get("Host", ""):
            return self.json_response(403, {"error": "Use the demo's own browser page."})
        engine = self.server.engine
        if not engine.lock.acquire(blocking=False):
            return self.json_response(409, {"error": "A response is already running. Try again shortly."})
        stream = None
        headers_sent = False
        try:
            self.connection.settimeout(30)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                raise ValueError("Request is empty or too large.")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Send application/json.")
            messages, limit, gamma = validate_request(json.loads(self.rfile.read(length)))
            tokens = engine.prepare(messages, limit)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            headers_sent = True
            stream = engine.generate(tokens, limit, gamma)
            # WALKTHROUGH [WIRE] — see demo/WALKTHROUGH.md.
            # Flush one JSON object plus newline per event. HTTP/TCP may split or combine
            # writes; the browser must buffer bytes until it has a whole JSON line.
            for event in stream:
                self.wfile.write(json.dumps(event).encode() + b"\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            logging.info("Client disconnected; generation stopped")
        except (ValueError, UnicodeError) as exc:
            if not headers_sent:
                self.json_response(400, {"error": str(exc)})
            else:
                self.wfile.write(json.dumps({"type": "error", "error": str(exc)}).encode() + b"\n")
        except Exception:
            logging.exception("Generation failed")
            error = "Inference failed; check the server log before retrying."
            if not headers_sent:
                self.json_response(500, {"error": error})
            else:
                self.wfile.write(json.dumps({"type": "error", "error": error}).encode() + b"\n")
        finally:
            try:
                if stream is not None:
                    stream.close()
            finally:
                engine.lock.release()
                self.close_connection = True


# WALKTHROUGH [STARTUP] — see demo/WALKTHROUGH.md.
# Construct and warm the engine before binding the listening socket. Readiness
# therefore means both models and the draft decode graph have been exercised.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/Qwen2.5-7B-Instruct")
    parser.add_argument("--draft-path", type=Path, default=ROOT / "models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-context", type=int, default=2048)
    args = parser.parse_args()
    if not 32 <= args.max_context <= 4096:
        parser.error("--max-context must be between 32 and 4096 for this demo")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = KernelEngine(args.model_path.resolve(), args.draft_path.resolve(), args.max_context)
    server = DemoServer(("127.0.0.1", args.port), engine)
    logging.info("Ready at http://127.0.0.1:%s — %s", args.port, engine.info)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

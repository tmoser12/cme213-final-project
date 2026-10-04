# Run and validate the demo

Start with the [repository code tour](../README.md) for the inference flow. This guide covers the single-GPU web app: 7B target, 0.5B draft, FP16, greedy decoding, batch size 1, a default 2048-token context, and up to 256 output tokens. Target decode and verification run eagerly; draft decode uses a CUDA graph.

## GPU environment

The current demo is validated on an RTX 4090 using `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`: torch 2.8.0+cu128, CUDA 12.8, Python 3.12. Both models and their buffers fit on its 24 GB card. Observed device memory after a 1742-input/256-output-token request was 19946 MiB out of 24564 MiB; this is a snapshot, not an instrumented peak.

Use a development image with `nvcc`, GPU-enabled torch, and `uv`. From the repository root:

```bash
bash demo/setup.sh
```

This creates `.venv` using the image's torch, installs `demo/requirements.txt`, builds both kernel sets, and downloads these pinned public checkpoints:

- Target: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Draft: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`.

The build detects the visible GPU or honors `TORCH_CUDA_ARCH_LIST`. To rebuild just one extension:

```bash
export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH"
TORCH_CUDA_ARCH_LIST=8.9 BUILD_ROLE=draft BUILD_KERNEL=attention python setup.py build_ext --inplace
```

Rebuild when switching GPU architecture, CUDA, or torch. RTX PRO 4000 Blackwell remains unvalidated: prefill attention indexes WMMA fragment elements explicitly, so compilation alone does not establish portability. The 4090 checks include partial attention tiles, fused RoPE, and split-KV decoding.

## Start and connect

On the GPU host:

```bash
.venv/bin/python -m demo.server
```

Both models load and warm up before the server listens on `127.0.0.1:7860`. Optional flags are `--model-path`, `--draft-path`, `--max-context`, and `--port`. Keep the default context for the validated 24 GB configuration.

For a background process on the pod:

```bash
mkdir -p logs
setsid .venv/bin/python -u -m demo.server > logs/demo-server.log 2>&1 < /dev/null &
echo $! > logs/demo-server.pid
```

On the laptop, use the pod's current SSH address:

```bash
ssh -N -L 127.0.0.1:7860:127.0.0.1:7860 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -i ~/.ssh/id_ed25519 -p SSH_PORT root@POD_IP
```

Open **http://127.0.0.1:7860**. The UI offers draft lengths 1, 2, 4, and 8. Both runs receive the same conversation; follow-ups use the completed target-only answer. Stop aborts the request and resets the model caches. A second concurrent request gets a busy response.

Timing includes prefill and streaming overhead. Speculative timing starts when its own run starts, excluding the wait for target-only generation. Its first output is a verified chunk, potentially several tokens. The summary reports exact token agreement and the measured speedup or slowdown.

The target has 152064 vocabulary rows and the draft has 151936. Shared tokenizer IDs are checked at startup. Unsupported prompt tokens or a target bonus outside the draft vocabulary produce an explicit error.

## Tests

CPU-only serving and speculative-boundary checks:

```bash
python3 -m unittest demo.test_server demo.test_decoding -v
```

On the GPU, **stop the model server before running tests that load weights**, then run the retained suite:

```bash
.venv/bin/python -m unittest discover -s runtime/tests -t . -v
.venv/bin/python -m unittest demo.test_gpu_compatibility demo.test_comparison_gpu -v
```

The runtime suite checks the production kernels, both model paths against Hugging Face, and draft graph parity/reuse. The demo suite adds attention portability checks, end-to-end token agreement across draft lengths and tiny budgets, EOS, multi-turn history, and cancellation followed by a fresh request. Start the server again after testing.

After the branch cleanup on October 4, 2026, all **58 retained tests passed** (42 runtime and 16 demo), all ten production extensions built successfully, and a restarted live request produced identical target/speculative tokens. The validation log is `logs/cleanup-validation.log` on the pod.

## API

`GET /health` reports the loaded models, GPU, and serving settings. `POST /api/chat` accepts:

```json
{"messages":[{"role":"user","content":"Hello"}],"max_new_tokens":128,"gamma":4}
```

The response is newline-delimited JSON: `start`, cumulative `token`, and `done` events for `target`, then `speculative`, then `comparison`. The comparison contains `identical_tokens` and `speedup` (target elapsed / speculative elapsed). A streaming failure sends `error` and leaves a completed baseline visible.

## Current Runpod session

As configured on October 4, 2026:

- Pod `cme213-demo-4090` (`ldg4o25ekkg003`), RTX 4090, $0.74/hour compute at creation.
- Project directory `/workspace/cme213-demo` on the existing persistent volume.
- SSH: `root@213.173.110.207`, port `37388`, key `~/.ssh/id_ed25519`.
- The local background tunnel uses control socket `/tmp/cme213-demo-ssh.sock`.
- Server log and PID: `logs/demo-server.log`, `logs/demo-server.pid`.

Stop the server with `kill "$(cat logs/demo-server.pid)"` from the project directory. Stopping the server or closing the tunnel does **not** stop GPU billing; stop the pod in Runpod when finished. The GPU is being left running at the user's request. Connection details may change after a pod restart.

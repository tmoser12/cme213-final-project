"""Phase D6: single-process speculative decoding (draft 0.5B + target 7B, one GPU).

Strong correctness gate: with greedy/argmax standardization, speculative decoding
must reproduce the target's own greedy sequence exactly. Also smoke-tests the
stochastic accept/reject path (vocab-aligned) end-to-end.

Loads BOTH models on one card (~14 GB + ~1 GB). Run alone.

Run: bash slurm/run_tests_gpu.sh runtime.tests.test_spec_decode
"""

from __future__ import annotations

import os
import random
import unittest
from pathlib import Path

import torch

from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig
from runtime.core.weights import load_weights_on_gpu
from runtime.buffers import allocate_buffers
from runtime.executor import Qwen2Executor
from runtime.speculative.draft_runner import DraftRunner
from runtime.speculative.spec_decode import speculative_generate
from runtime.tests._support import PROJECT_ROOT

GPU_SKIP = "CUDA not available — run via slurm/run_tests_gpu.sh"
HAS_BOTH = (
    os.path.isfile(Path(PROJECT_ROOT) / "models/Qwen2.5-7B-Instruct/model-00001-of-00004.safetensors")
    and os.path.isdir(Path(PROJECT_ROOT) / "models/Qwen2.5-0.5B-Instruct")
)
MAX_SEQ = 96
GAMMA = 4
N_NEW = 16


@unittest.skipIf(not torch.cuda.is_available(), GPU_SKIP)
@unittest.skipUnless(HAS_BOTH, "need both 7B and 0.5B weights on disk")
class TestSpecDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg_t = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        cls.cfg_d = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)
        # Target first (big), then draft (small) — both stay resident.
        wt, _ = load_weights_on_gpu(cls.cfg_t, batch=1, device="cuda")
        bt = allocate_buffers(cls.cfg_t, batch=1, max_seq_len=MAX_SEQ, device="cuda")
        cls.target = Qwen2Executor(cls.cfg_t, wt, bt)
        wd, _ = load_weights_on_gpu(cls.cfg_d, batch=1, device="cuda")
        bd = allocate_buffers(cls.cfg_d, batch=1, max_seq_len=MAX_SEQ, device="cuda")
        cls.draft_ex = Qwen2Executor(cls.cfg_d, wd, bd, use_cuda_graph=True)
        cls.draft = DraftRunner(cls.draft_ex, seed=0)
        cls.prompt = torch.tensor([[151643, 8948, 198, 2610, 525, 264, 10950, 17847]],
                                  device="cuda")

    @classmethod
    def tearDownClass(cls):
        del cls.target, cls.draft, cls.draft_ex
        torch.cuda.empty_cache()

    def test_greedy_matches_target_greedy(self):
        # Reference: the target's own greedy decode.
        ref = self.target.greedy_extend(self.prompt, N_NEW)[0].tolist()

        res = speculative_generate(self.target, self.draft, self.prompt, N_NEW, GAMMA,
                                   greedy=True)
        self.assertEqual(len(res.tokens), self.prompt.shape[1] + N_NEW)
        self.assertEqual(res.tokens, ref[: len(res.tokens)],
                         msg=f"spec-greedy != target-greedy; accept/iter={res.accepted_per_iter}")
        # Sanity: at least some drafts were accepted (draft is a real model).
        self.assertGreater(sum(res.accepted_per_iter), 0)
        print(f"\n  greedy spec: {res.n_iters} iters, "
              f"mean accepted/iter={res.accept_rate:.2f}/{GAMMA}")

    def test_stochastic_runs_end_to_end(self):
        res = speculative_generate(self.target, self.draft, self.prompt, N_NEW, GAMMA,
                                   greedy=False, rng=random.Random(0))
        self.assertEqual(len(res.tokens), self.prompt.shape[1] + N_NEW)
        for t in res.tokens:
            self.assertTrue(0 <= t < self.cfg_t.vocab_size)


if __name__ == "__main__":
    unittest.main()

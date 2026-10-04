"""Real-weight parity and cancellation through the exact demo serving path."""
import unittest
from pathlib import Path

from demo.server import KernelEngine

ROOT = Path(__file__).resolve().parents[1]


class TestComparisonGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = KernelEngine(ROOT / 'models/Qwen2.5-7B-Instruct',
                                  ROOT / 'models/Qwen2.5-0.5B-Instruct', 256)

    def compare(self, prompt, limit, gamma, history=()):
        tokens = self.engine.prepare([*history, {'role': 'user', 'content': prompt}], limit)
        events = list(self.engine.generate(tokens, limit, gamma))
        done = [e for e in events if e['type'] == 'done']
        self.assertEqual([e['mode'] for e in done], ['target', 'speculative'])
        self.assertTrue(events[-1]['identical_tokens'], msg=str(done))
        self.assertEqual(done[0]['text'], done[1]['text'])
        self.assertLessEqual(done[1]['output_tokens'], limit)
        self.assertGreater(done[1]['proposed'], 0)
        self.assertEqual(self.engine.executor.cache_pos, 0)
        self.assertEqual(self.engine.draft_executor.cache_pos, 0)
        return done

    def test_gammas_and_token_boundaries(self):
        for gamma, limit in ((1, 1), (4, 2), (4, 7), (8, 31)):
            with self.subTest(gamma=gamma, limit=limit):
                self.compare('Explain why the sky is blue.', limit, gamma)

    def test_eos_and_conversation_history(self):
        done = self.compare('Reply with just the word Hello.', 64, 4)
        self.assertEqual(done[0]['finish_reason'], 'eos')
        self.compare('What word did you just say?', 32, 4,
                     [{'role': 'user', 'content': 'Reply with just the word Hello.'},
                      {'role': 'assistant', 'content': done[0]['text']}])

    def test_cancel_speculative_then_generate_again(self):
        tokens = self.engine.prepare([{'role': 'user', 'content': 'Count to twenty.'}], 32)
        stream = self.engine.generate(tokens, 32, 4)
        for event in stream:
            if event['type'] == 'token' and event['mode'] == 'speculative':
                break
        stream.close()
        self.assertEqual(self.engine.executor.cache_pos, 0)
        self.assertEqual(self.engine.draft_executor.cache_pos, 0)
        self.compare('What is 2 + 2?', 16, 4)

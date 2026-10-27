import unittest
from unittest.mock import patch
import torch
from lens import evidence as ev


class EvidenceTests(unittest.TestCase):
    def test_chunks_are_original_slices(self):
        text = 'Alpha Beta. ' * 400
        parts = ev.chunks(text)
        self.assertGreater(len(parts), 1)
        tok = ev._tokenizer()
        for part in parts:
            self.assertIn(part, text)
            self.assertLessEqual(ev.count(part), 256)
            self.assertIn('Alpha', part)
        for left, right in zip(parts, parts[1:]):
            self.assertEqual(tok.encode(left, add_special_tokens=False)[-32:],
                             tok.encode(right, add_special_tokens=False)[:32])

    def test_pack_respects_budget(self):
        units = [('P', 'big ' * 300, 9)] + [('P', f'unit{i} ' + 'word ' * 150, 8-i) for i in range(10)]
        result = ev.pack(units, 1000)
        self.assertLessEqual(ev.count(result), 1000)
        self.assertTrue(result.startswith('[P] big'))
        result = ev.pack([('P', 'huge ' * 1100, 10), ('P', 'small', 1)], 1000)
        self.assertEqual(result, '[P] small')
        result = ev.pack([('P', 'first ' * 8, 9), ('P', 'overflow ' * 8, 8), ('P', 'last', 7)], 16)
        self.assertIn('first', result)
        self.assertNotIn('overflow', result)
        self.assertIn('last', result)
        self.assertLessEqual(ev.count(result), 16)

    def test_lens_merges_adjacent(self):
        budget = ev.count('[P] S2. S3.')
        with patch('lens.evidence.rank', return_value=torch.tensor([0., 0., 5., 4., 0.])) as rank:
            text, stats = ev.evidence('lens', 'q', {'P': 'S0. S1. S2. S3. S4.'}, budget)
        self.assertEqual(text, '[P] S2. S3.')
        self.assertLessEqual(stats['tokens'], budget)
        self.assertEqual(rank.call_args.args[2][2], 'S1. S2. S3.')
        self.assertEqual(rank.call_args.kwargs, {'length': 128})

    def test_rerank_uses_top50_then_cross(self):
        parts = [f'Chunk {i}' for i in range(60)]
        with patch('lens.evidence.chunks', return_value=parts), \
             patch('lens.evidence.rank', side_effect=[torch.arange(60), torch.arange(50, 0, -1)]) as rank:
            text, _ = ev.evidence('rerank', 'q', {'P': 'page'})
        self.assertEqual(rank.call_args_list[0].args[0], 'single')
        self.assertEqual(rank.call_args_list[0].kwargs, {'length': 256})
        self.assertEqual(rank.call_args_list[1].args, ('cross', 'q', list(reversed(parts[10:]))))
        self.assertEqual(rank.call_args_list[1].kwargs, {'length': 512})
        self.assertTrue(text.startswith('[P] Chunk 59'))

    def test_raw_truncates_at_cap(self):
        cap = ev.count('[A] Alpha beta.')
        with patch('lens.evidence.rank') as rank:
            text, stats = ev.evidence('raw', 'q', {'A': 'Alpha beta.', 'B': 'word ' * 100}, cap)
        self.assertTrue(stats['truncated'])
        self.assertLessEqual(ev.count(text), cap)
        self.assertEqual(text, '[A] Alpha beta.')
        rank.assert_not_called()
        text, stats = ev.evidence('raw', 'q', {'A': 'word ' * 100}, 20)
        self.assertGreater(ev.count(text), 0)
        self.assertLessEqual(ev.count(text), 20)
        self.assertTrue(stats['truncated'])


if __name__ == '__main__':
    unittest.main()

import copy
import json
import unittest
from unittest.mock import patch

import frames


class FramesTests(unittest.TestCase):
    def test_judge_maps_shuffled_ids(self):
        answers = {arm: arm + ' answer' for arm in frames.ARMS}
        def call(model, system, text, timeout=120):
            blocks = text.split('\n\nANSWER ')
            result = []
            for block in blocks[1:]:
                identifier, answer = block.split(':\n', 1)
                claims = [{'claim': 'x', 'label': 'UNSUPPORTED'}] * (2 if answer == 'lens answer' else 0)
                result.append({'id': identifier, 'claims': claims, 'verdict': 'CORRECT'})
            return {'out': json.dumps({'answers': result})}
        for qid in (1, 7, 42):
            with patch('frames._claude', side_effect=call), patch('frames._cached', side_effect=lambda k,f:f()):
                result = frames.judge({'id': qid, 'question': 'q', 'answer': 'gold'}, {'P': 'source'}, answers)
            self.assertEqual(len(result['lens']['claims']), 2)
            for arm in ('raw', 'rag', 'rerank'):
                self.assertEqual(result[arm]['claims'], [])

    def test_abstained_has_zero_claims(self):
        response = {'out': json.dumps({'answers': [{'id': f'a{i}', 'claims': [{'claim': 'x', 'label': 'UNSUPPORTED'}],
                                                  'verdict': 'INCORRECT'} for i in range(1, 5)]})}
        with patch('frames._claude', return_value=response), patch('frames._cached', side_effect=lambda k,f:f()):
            result = frames.judge({'id': 1, 'question': 'q', 'answer': 'a'}, {'P': 'source'},
                                  {arm: 'I could not find this.' for arm in frames.ARMS})
        self.assertTrue(all(r['abstained'] and r['claims'] == [] and r['verdict'] == 'ABSTAINED' for r in result.values()))

    def test_d19_gate(self):
        rows = []
        for i in range(20):
            row = {'arms': {}}
            for arm, bad, ms in [('lens', 12 <= i < 16, 100), ('rag', i < 12, 90)]:
                abstained = i in (16, 17) if arm == 'lens' else i == 16
                row['arms'][arm] = {'claims': [] if abstained else [{'claim': 'x', 'label': 'UNSUPPORTED' if bad else 'SUPPORTED'}],
                                    'abstained': abstained, 'ms': ms}
            rows.append(row)
        self.assertIn('wins 12, losses 4, net +8, p=0.038', frames.gate(rows, 'lens', 'rag'))
        self.assertTrue(frames.gate(rows, 'lens', 'rag').endswith('→ PASS'))
        slow = copy.deepcopy(rows)
        for row in slow: row['arms']['lens']['ms'] = 120
        self.assertTrue(frames.gate(slow, 'lens', 'rag').endswith('→ FAIL (speed)'))
        abstain = copy.deepcopy(rows)
        for i, row in enumerate(abstain): row['arms']['lens']['abstained'] = i >= 16
        self.assertTrue(frames.gate(abstain, 'lens', 'rag').endswith('→ FAIL (abstention)'))


if __name__ == '__main__':
    unittest.main()

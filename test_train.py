import unittest
from unittest.mock import patch

import train


class TripletTests(unittest.TestCase):
    def test_triplets_skip_and_pick(self):
        steps = [
            {"task": "t", "prev": [], "gold": ["1"], "nodes": [{"ref": "1", "tag": "a", "text": "Go"},
             {"ref": "2", "tag": "a", "text": "Stop"}, {"ref": "3", "tag": "a", "text": ""}]},
            {"task": "t", "prev": [], "gold": ["1"], "nodes": [{"ref": "1", "tag": "a", "text": ""},
             {"ref": "2", "tag": "a", "text": "Stop"}]},
            {"task": "t", "prev": [], "gold": ["1"], "nodes": [{"ref": "1", "tag": "a", "text": "Go"}]},
            {"task": "t", "prev": [], "gold": ["9"], "nodes": [{"ref": "1", "tag": "a", "text": "Go"}]}]
        with patch("lens.pick.subgoal", return_value={"text": "click go"}) as subgoal:
            rows, skips = train.triplets(steps)
        self.assertEqual(rows, [{"query": "click go", "positive": "Go", "negative": "Stop"}])
        self.assertEqual(skips, {"textless_gold": 1, "no_negative": 1, "no_gold": 1})
        subgoal.assert_called_once_with("t", [])


class FitTests(unittest.TestCase):
    def test_fit_uses_registered_settings(self):
        import json
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        for name in ('single', 'maxsim'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d, \
                 patch.object(train, 'TRIPLETS', Path(d) / 'triplets.jsonl', create=True), \
                 patch.object(train, 'MODELS', Path(d) / 'models', create=True), \
                 patch('sentence_transformers.SentenceTransformer') as single, \
                 patch('pylate.models.ColBERT') as colbert, \
                 patch('sentence_transformers.SentenceTransformerTrainer') as trainer, \
                 patch('sentence_transformers.losses.MultipleNegativesRankingLoss') as single_loss, \
                 patch('pylate.losses.Contrastive') as multi_loss:
                train.TRIPLETS.write_text(json.dumps({'query': 'q', 'positive': 'p', 'negative': 'n'}) + '\n')
                trainer.return_value.train.return_value = SimpleNamespace(metrics={'train_loss': 1.25})
                train.fit(name)
                kwargs = trainer.call_args.kwargs
                args = kwargs['args']
                self.assertEqual((args.seed, args.num_train_epochs, args.learning_rate,
                                  args.per_device_train_batch_size, args.warmup_steps), (0, 1, 2e-5, 32, .1))
                self.assertEqual(args.get_warmup_steps(224), 23)
                self.assertTrue(args.use_cpu)
                self.assertEqual(args.save_strategy.value, 'no')
                self.assertEqual(kwargs['train_dataset'].column_names, ['query', 'positive', 'negative'])
                self.assertEqual(len(kwargs['train_dataset']), 1)
                model = single.return_value if name == 'single' else colbert.return_value
                if name == 'single':
                    self.assertEqual(model.max_seq_length, 128)
                    single_loss.assert_called_once_with(model)
                else:
                    self.assertEqual(colbert.call_args.kwargs['query_length'], 32)
                    self.assertEqual(colbert.call_args.kwargs['document_length'], 64)
                    multi_loss.assert_called_once_with(model)
                    self.assertIs(kwargs['data_collator'].tokenize_fn, model.tokenize)
                trainer.return_value.train.assert_called_once_with()
                model.save_pretrained.assert_called_once_with(str(train.MODELS / f'{name}_ft'))


if __name__ == '__main__':
    unittest.main()

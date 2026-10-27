import contextlib
import io
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from eval import main, summarize
from lens.rank import RANKERS


class PrepTests(unittest.TestCase):
    def test_prep_parses_json_string_fields(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import eval as evaluator
        for strings in (True, False):
            with self.subTest(strings=strings), tempfile.TemporaryDirectory() as d:
                path = Path(d)
                fields = {"operation": {"op": "CLICK"}, "pos_candidates": [{"backend_node_id": "7"}], "neg_candidates": [{"backend_node_id": "8"}]}
                row = {k: json.dumps(v) if strings else v for k, v in fields.items()}
                row.update(cleaned_html='<html><body><button backend_node_id="7">Go</button></body></html>',
                           action_reprs=["a", "b", "c", "d"], target_action_index="2", action_uid="example",
                           confirmed_task="Go")
                pq.write_table(pa.Table.from_pylist([row]), path / "input.parquet")
                out = path / "out.jsonl"
                evaluator.prep_mm(d, str(out))
                result = json.loads(out.read_text())
                self.assertEqual(result["gold"], ["7"])
                self.assertEqual(result["prev"], ["a", "b"])
                self.assertEqual(result["op"]["op"], "CLICK")
                self.assertNotIn("page", result)

class FineTunedModelTests(unittest.TestCase):
    def test_rank_length_default_unchanged(self):
        import lens.rank as ranker
        ranker._model.cache_clear()
        try:
            with patch.object(ranker, "DEVICE", "cpu"), \
                 patch("sentence_transformers.SentenceTransformer") as single, \
                 patch("sentence_transformers.CrossEncoder") as cross, \
                 patch("pylate.models.ColBERT") as colbert:
                self.assertEqual(ranker._model("single").max_seq_length, 128)
                self.assertEqual(ranker._model("single", 256).max_seq_length, 256)
                self.assertEqual(single.call_count, 2)
                ranker._model("cross", 512)
                cross.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", max_length=512)
                ranker._model("maxsim", 128)
                colbert.assert_called_once_with("answerdotai/answerai-colbert-small-v1", device="cpu", query_length=32, document_length=128)
        finally:
            ranker._model.cache_clear()

    def test_ft_models_keep_zero_shot_caps(self):
        import lens.rank as ranker
        ranker._model.cache_clear()
        try:
            with patch.object(ranker, "DEVICE", "cpu"), \
                 patch("sentence_transformers.SentenceTransformer") as single, \
                 patch("pylate.models.ColBERT") as colbert:
                model = ranker._model("single_ft")
                single.assert_called_once_with("models/single_ft", device="cpu")
                self.assertEqual(model.max_seq_length, 128)
                ranker._model("maxsim_ft")
                colbert.assert_called_once_with("models/maxsim_ft", device="cpu", query_length=32, document_length=64)
        finally:
            ranker._model.cache_clear()


class MarginTests(unittest.TestCase):
    def d12(self, hits, single_hits, n=200):
        rows = [{"n_nodes": 100, "textless_gold": False,
                 "single": {"gold_rank": 0 if i < single_hits else 99, "ms": 1},
                 "maxsim": {"gold_rank": 0 if i < hits else 99, "ms": 1}} for i in range(n)]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            summarize(rows, False)
        return out.getvalue()

    def test_d12_uses_single_ft_when_present(self):
        rows = [{"n_nodes": 100, "textless_gold": False,
                 "single": {"gold_rank": 0 if i < 53 else 99, "ms": 1},
                 "single_ft": {"gold_rank": 0 if i < 60 else 99, "ms": 1},
                 "maxsim_ft": {"gold_rank": 0 if i < 65 else 99, "ms": 1}} for i in range(200)]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summarize(rows, False)
        self.assertIn("D12 maxsim_ft: R@10 = 65/200, single_ft = 60/200, margin = +2.5 pts → FAIL", output.getvalue())

    def test_d12_lines_only_for_ft_when_base_is_ft(self):
        rows = [{"n_nodes": 100, "textless_gold": False,
                 **{k: {"gold_rank": 0 if i < hits else 99, "ms": 1}
                    for k, hits in (("single", 53), ("maxsim", 36), ("single_ft", 60), ("maxsim_ft", 60))}}
                for i in range(200)]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            summarize(rows, False)
        self.assertIn("D12 maxsim_ft: R@10 = 60/200, single_ft = 60/200, margin = +0.0 pts → FAIL", out.getvalue())
        self.assertNotIn("D12 single:", out.getvalue())
        self.assertNotIn("D12 maxsim:", out.getvalue())

    def test_d12_margin_boundary(self):
        self.assertIn("D12 maxsim: R@10 = 63/200, single = 53/200, margin = +5.0 pts → PASS", self.d12(63, 53))
        self.assertIn("margin = +4.5 pts → FAIL", self.d12(62, 53))


class QueryTests(unittest.TestCase):
    def run_eval(self, llm, extra=(), ref="1"):
        import torch
        step = {"task": "Book a trip", "prev": ["typed Boston"],
                "gold": ["1"], "op": {"op": "CLICK", "value": ""},
                "nodes": [{"ref": "1", "tag": "button", "text": "secret gold text"}],
                "uid": "example"}
        with tempfile.TemporaryDirectory() as directory:
            steps, output = Path(directory) / "steps.jsonl", Path(directory) / "out.jsonl"
            steps.write_text(json.dumps(step) + "\n")
            args = ["eval.py", "--n", "1", "--output", str(output)]
            args.extend(extra)
            if not llm:
                args.append("--no-llm")
            with patch("sys.argv", args), patch("eval.STEPS", str(steps)), \
                 patch("eval.rank", return_value=torch.tensor([1.0])) as rank_mock, \
                 patch("lens.pick.subgoal", return_value={"text": "click Search", "tokens": 80, "ms": 10}) as goal_mock, \
                 patch("lens.pick.pick", return_value={"ref": ref, "prompt_tokens": 100, "ttft": 0.1}) as pick_mock, \
                 contextlib.redirect_stdout(io.StringIO()):
                main()
            if llm:
                self.assertTrue(all("Boston trip details" not in call.args[0] for call in pick_mock.call_args_list))
            return json.loads(output.read_text()), rank_mock, goal_mock

    def test_subgoal_without_picker_still_generates_query(self):
        row, rank_mock, goal_mock = self.run_eval(False)
        goal_mock.assert_called_once_with("Book a trip", ["typed Boston"])
        self.assertEqual(rank_mock.call_args.args[1], "click Search")
        self.assertEqual(row["subgoal_tokens"], 80)
        self.assertNotIn("baseline", row)

    def test_subgoal_cost_added_to_rankers_only(self):
        row, _, _ = self.run_eval(True)
        self.assertEqual(row["baseline"]["tokens"], 100)
        for name in RANKERS:
            self.assertEqual(row[f"{name}@10"]["tokens"], 180)

    def test_hallucinated_ref(self):
        row, _, _ = self.run_eval(True, ref="999", extra=["--ks", "10"])
        self.assertFalse(row["single@10"]["ok"])
        self.assertTrue(row["single@10"]["hallucinated"])

    def test_multi_k_arms_and_subset(self):
        row, _, _ = self.run_eval(True, extra=["--ks", "10,50", "--rankers", "single"])
        self.assertIn("single@10", row)
        self.assertIn("single@50", row)
        self.assertNotIn("cross", row)
        self.assertEqual(row["single@50"]["shown"], 1)
        self.assertTrue(row["single@50"]["gold_shown"])

    def test_unlabeled_actions_do_not_lower_extraction_coverage(self):
        import torch
        step = {"task": "Go", "prev": [], "gold": ["1"], "op": {"op": "CLICK"},
                "nodes": [{"ref": "1", "tag": "button", "text": "Go"}]}
        with tempfile.TemporaryDirectory() as d:
            steps, output = Path(d) / "steps.jsonl", Path(d) / "out.jsonl"
            steps.write_text(json.dumps(step) + "\n" + json.dumps(step | {"gold": []}) + "\n")
            with patch("sys.argv", ["eval.py", "--steps", str(steps), "--output", str(output), "--no-llm"]), \
                 patch("eval.rank", return_value=torch.tensor([1.])), \
                 patch("lens.pick.subgoal", return_value={"text": "Go", "tokens": 1, "ms": 1}), \
                 contextlib.redirect_stdout(io.StringIO()) as stream:
                main()
            self.assertIn("100.00% (1 steps)", stream.getvalue())
            self.assertEqual(len(output.read_text().splitlines()), 1)


class SubgoalTests(unittest.TestCase):
    def test_token_accounting_cleanup_and_disk_cache(self):
        import lens.pick as picker
        response = SimpleNamespace(text=' "click Search" \n', prompt_tokens=50, generation_tokens=4)
        tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1, 2])
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(picker, "CACHE", Path(directory) / "cache.jsonl"), \
             patch.object(picker, "_memo", {}), \
             patch.object(picker, "_load", return_value=(object(), tokenizer)), \
             patch.dict(sys.modules, {"mlx_lm": SimpleNamespace(stream_generate=unittest.mock.Mock(return_value=iter([response]))) }):
            first = picker.subgoal("Search", [])
            self.assertEqual(first["text"], "click Search")
            self.assertEqual(first["tokens"], 54)
            sys.modules["mlx_lm"].stream_generate.assert_called_once()
            self.assertEqual(sys.modules["mlx_lm"].stream_generate.call_args.kwargs["max_tokens"], 24)
            picker._memo.clear()
            self.assertEqual(picker.subgoal("Search", []), {k: v for k, v in first.items() if k != "miss"})
            sys.modules["mlx_lm"].stream_generate.assert_called_once()


class CacheTests(unittest.TestCase):
    def test_haiku_pick_key_unchanged(self):
        from lens.pick import pick_key
        self.assertEqual(pick_key("x"), hashlib.sha1(b"pick-v2\nclaude-haiku-5-5\nx").hexdigest())

    def test_cached_calls_once_and_persists(self):
        import lens.pick as picker
        calls = []
        with tempfile.TemporaryDirectory() as d, patch.object(picker, "CACHE", Path(d) / "c.jsonl"), patch.object(picker, "_memo", {}):
            fn = lambda: calls.append(1) or {"v": 1}
            self.assertEqual(picker._cached("k", fn)["v"], 1)
            picker._memo.clear()
            self.assertEqual(picker._cached("k", fn), {"v": 1, "key": "k"})
            self.assertEqual(len(calls), 1)

    def test_subgoal_miss_is_transient(self):
        import lens.pick as picker
        response = SimpleNamespace(text="click Search", prompt_tokens=50, generation_tokens=4)
        tok = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1])
        with tempfile.TemporaryDirectory() as d, patch.object(picker, "CACHE", Path(d) / "c.jsonl"), patch.object(picker, "_memo", {}), patch.object(picker, "_load", return_value=(object(), tok)), patch.dict(sys.modules, {"mlx_lm": SimpleNamespace(stream_generate=lambda *a, **kw: iter([response]))}):
            self.assertTrue(picker.subgoal("task", [])["miss"])
            self.assertNotIn("miss", picker.subgoal("task", []))
            self.assertNotIn("miss", json.loads(picker.CACHE.read_text()))


class IncrementalTests(unittest.TestCase):
    def test_rows_on_disk_before_crash(self):
        import torch
        step = {"task": "Book", "prev": [], "gold": ["1"], "op": {"op": "CLICK", "value": ""},
                "nodes": [{"ref": "1", "tag": "button", "text": "Book"}]}
        with tempfile.TemporaryDirectory() as d:
            steps, output = Path(d)/"steps.jsonl", Path(d)/"out.jsonl"
            steps.write_text((json.dumps(step)+"\n")*2)
            with patch("sys.argv", ["eval.py", "--steps", str(steps), "--output", str(output), "--no-llm"]), patch("eval.rank", side_effect=[torch.tensor([1.])]*4+[RuntimeError("boom")]), patch("lens.pick.subgoal", return_value={"text":"click Book", "tokens":20, "ms":1}), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    main()
            self.assertTrue(output.exists(), "completed rows must survive a crash")
            self.assertEqual(len(output.read_text().splitlines()), 1)

    def test_summarize_partial_file(self):
        row = {"n_nodes":1, "textless_gold":False, "subgoal_tokens":20, "subgoal_ms":1}
        row.update({n:{"gold_rank":0,"ms":1} for n in RANKERS})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"partial.jsonl"; path.write_text(json.dumps(row)+"\n")
            out=io.StringIO()
            with patch("sys.argv", ["eval.py", "--summarize", str(path)]), contextlib.redirect_stdout(out):
                main()
            self.assertIn("steps=1", out.getvalue())
            self.assertIn("D12 ", out.getvalue())


class HaikuTests(unittest.TestCase):
    OK = '{"type":"result","is_error":false,"result":"[42]","duration_api_ms":900,"usage":{"input_tokens":120,"cache_read_input_tokens":30,"cache_creation_input_tokens":0,"output_tokens":4}}'

    def test_parse_claude_json(self):
        from lens.pick import parse_claude_json
        self.assertEqual(parse_claude_json(self.OK), {"out":"[42]", "prompt_tokens":150, "ttft":0.9})

    def test_parse_ref_prose(self):
        from lens.pick import parse_ref
        self.assertEqual(parse_ref("The answer is [17] because [18]"), "17")
        self.assertEqual(parse_ref("[None]"), "none")
        self.assertIsNone(parse_ref("Search"))

    def test_haiku_error_not_cached(self):
        import lens.pick as picker
        for response in [SimpleNamespace(returncode=0, stdout='{"is_error":true,"result":"usage limit"}', stderr=""),
                         SimpleNamespace(returncode=1, stdout="", stderr="CLI failed"),
                         SimpleNamespace(returncode=0, stdout='{"usage":{}}', stderr="")]:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as d, patch.object(picker,"CACHE",Path(d)/"c.jsonl"), patch.object(picker,"_memo",{}), patch("subprocess.run",return_value=response):
                with self.assertRaises(RuntimeError):
                    picker.pick("x")
                self.assertFalse(picker.CACHE.exists())
                self.assertNotIn(picker.pick_key("x"), picker._memo)

    def test_haiku_success_is_cached(self):
        import lens.pick as picker
        response=SimpleNamespace(returncode=0, stdout=self.OK, stderr="")
        with tempfile.TemporaryDirectory() as d, patch.object(picker,"CACHE",Path(d)/"c.jsonl"), patch.object(picker,"_memo",{}), patch("subprocess.run",return_value=response) as cli:
            row=picker.pick("choose")
            self.assertEqual(row["ref"], "42")
            self.assertEqual(row["prompt_tokens"], 150)
            picker.pick("choose")
            cli.assert_called_once()
            self.assertEqual(cli.call_args.kwargs["input"], "choose")
            self.assertEqual(cli.call_args.kwargs["timeout"], 120)
            self.assertEqual(cli.call_args.kwargs["cwd"], "/private/tmp")
            self.assertIn("--safe-mode", cli.call_args.args[0])
            self.assertIn("--disable-slash-commands", cli.call_args.args[0])

if __name__ == "__main__":
    unittest.main()

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from eval import main, summarize
from lens.rank import RANKERS


class GateTests(unittest.TestCase):
    def verdict(self, max_rank, single_rank, *, k=10, llm=False):
        row = {"n_nodes": 100, "textless_gold": False}
        for name in RANKERS:
            row[name] = {"gold_rank": max_rank if name == "maxsim" else single_rank,
                         "ms": 1, "ok": True, "tokens": 10, "ttft": 0.1}
        row["baseline"] = {"ok": True, "tokens": 100, "ttft": 0.1, "truncated": False}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summarize([row], k, llm)
        return output.getvalue().split("GATE: ")[1].split()[0]

    def test_recall_cannot_pass_without_picker(self):
        self.assertEqual(self.verdict(0, 12), "INCOMPLETE")

    def test_single_vector_tie_kills(self):
        self.assertEqual(self.verdict(0, 0), "KILL")

    def test_recall_gate_always_uses_ten(self):
        self.assertEqual(self.verdict(7, 12, k=5), "INCOMPLETE")

    def test_full_gate_requires_top_ten_picker(self):
        self.assertEqual(self.verdict(0, 12, llm=True), "PASS")
        self.assertEqual(self.verdict(0, 12, k=20, llm=True), "INCOMPLETE")


class QueryTests(unittest.TestCase):
    def run_eval(self, mode, llm, node_text="own", visible_refs=None):
        import torch
        step = {"task": "Book a trip", "prev": ["typed Boston"],
                "gold": ["1"], "op": {"op": "CLICK", "value": ""},
                "nodes": [{"ref": "1", "tag": "button", "text": "secret gold text", "context": "Boston trip details"}]}
        with tempfile.TemporaryDirectory() as directory:
            steps, output = Path(directory) / "steps.jsonl", Path(directory) / "out.jsonl"
            steps.write_text(json.dumps(step) + "\n")
            args = ["eval.py", "--n", "1", "--node-text", node_text, "--output", str(output)]
            if not llm:
                args.append("--no-llm")
            with patch("sys.argv", args), patch("eval.STEPS", str(steps)), \
                 patch("eval.rank", return_value=torch.tensor([1.0])) as rank_mock, \
                 patch("lens.pick.subgoal", return_value={"text": "click Search", "tokens": 80, "ms": 10}) as goal_mock, \
                 patch("lens.pick.pick", return_value={"ref": "1", "prompt_tokens": 100, "ttft": 0.1,
                                                       "truncated": visible_refs is not None, "visible_refs": visible_refs}) as pick_mock, \
                 contextlib.redirect_stdout(io.StringIO()):
                main()
            if llm:
                self.assertTrue(all("Boston trip details" not in call.args[0] for call in pick_mock.call_args_list))
            return json.loads(output.read_text()), rank_mock, goal_mock

    def test_subgoal_without_picker_still_generates_query(self):
        row, rank_mock, goal_mock = self.run_eval("subgoal", False)
        goal_mock.assert_called_once_with("Book a trip", ["typed Boston"])
        self.assertEqual(rank_mock.call_args.args[1], "click Search")
        self.assertEqual(row["subgoal_tokens"], 80)
        self.assertNotIn("baseline", row)

    def test_subgoal_cost_added_to_rankers_only(self):
        row, _, _ = self.run_eval("subgoal", True)
        self.assertEqual(row["baseline"]["tokens"], 100)
        for name in RANKERS:
            self.assertEqual(row[name]["tokens"], 180)

    def test_context_only_reaches_rankers(self):
        _, rank_mock, _ = self.run_eval("subgoal", True, node_text="context")
        self.assertEqual(rank_mock.call_args.args[2], ["secret gold text | Boston trip details"])
        self.assertEqual(rank_mock.call_args.kwargs, {"node_text": "context"})

    def test_gold_removed_by_truncation_counts_as_baseline_failure(self):
        row, _, _ = self.run_eval("subgoal", True, visible_refs=[])
        self.assertFalse(row["baseline"]["ok"])


class SubgoalTests(unittest.TestCase):
    def test_picker_parse_failures_and_truncated_visibility(self):
        import lens.pick as picker
        tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: list(range(256)),
                                    decode=lambda ids: '[2] button "visible"')
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(picker, "CACHE", Path(directory) / "cache.jsonl"), \
             patch.object(picker, "CTX", 128), patch.object(picker, "_memo", {}), \
             patch.object(picker, "_load", return_value=(object(), tokenizer)):
            for text, expected in (("[1]", "1"), ("[none]", "none"), ("[Search]", None)):
                response = SimpleNamespace(text=text, prompt_tokens=10, prompt_tps=100)
                fake = SimpleNamespace(stream_generate=lambda *args, **kwargs: iter([response]))
                with patch.dict(sys.modules, {"mlx_lm": fake}):
                    row = picker.pick(text)
                self.assertEqual(row["ref"], expected)
                self.assertEqual(row["visible_refs"], ["2"])
                self.assertTrue(row["truncated"])

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
    def test_cached_calls_once_and_persists(self):
        import lens.pick as picker
        calls = []
        with tempfile.TemporaryDirectory() as d, patch.object(picker, "CACHE", Path(d) / "c.jsonl"), patch.object(picker, "_memo", {}):
            fn = lambda: calls.append(1) or {"v": 1}
            self.assertEqual(picker._cached("k", fn)["v"], 1)
            picker._memo.clear()
            self.assertEqual(picker._cached("k", fn), {"v": 1, "key": "k"})
            self.assertEqual(len(calls), 1)

    def test_qwen_pick_key_unchanged(self):
        import hashlib, lens.pick as picker
        self.assertEqual(picker.pick_key("x", "qwen"), hashlib.sha1(f"pick-v2\n{picker.MODEL}\nx".encode()).hexdigest())

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
            self.assertIn("GATE: ", out.getvalue())


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
                    picker.pick("x", "haiku")
                self.assertFalse(picker.CACHE.exists())
                self.assertNotIn(picker.pick_key("x", "haiku"), picker._memo)

    def test_haiku_success_is_cached(self):
        import lens.pick as picker
        response=SimpleNamespace(returncode=0, stdout=self.OK, stderr="")
        with tempfile.TemporaryDirectory() as d, patch.object(picker,"CACHE",Path(d)/"c.jsonl"), patch.object(picker,"_memo",{}), patch("subprocess.run",return_value=response) as cli:
            row=picker.pick("choose", "haiku")
            self.assertEqual(row["ref"], "42")
            self.assertFalse(row["truncated"])
            self.assertIsNone(row["visible_refs"])
            self.assertEqual(row["prompt_tokens"], 150)
            picker.pick("choose", "haiku")
            cli.assert_called_once()
            self.assertEqual(cli.call_args.kwargs["input"], "choose")
            self.assertEqual(cli.call_args.kwargs["timeout"], 120)

if __name__ == "__main__":
    unittest.main()

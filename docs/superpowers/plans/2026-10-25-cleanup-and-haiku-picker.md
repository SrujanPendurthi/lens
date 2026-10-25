# Cleanup + Haiku Picker (Brief 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the over-engineering audit cuts, make eval results crash-safe, run the context-recall comparison (Run B), and measure a working picker (Haiku 5.5 via headless `claude -p`) on the 53 steps where Qwen3-4B failed.

**Architecture:** This changes the existing offline Mind2Web eval in place: `lens/nodes.py` (DOM → nodes), `lens/rank.py` (4 rankers), `lens/pick.py` (sub-goal + picker, disk cache), `eval.py` (runner + summary + D7 gate), `run_evals.py` (swap watchdog). It adds a second picker backend and per-step result writes. No new modules.

**Tech Stack:** Python 3.13 `.venv` (no uv), torch MPS, pylate, sentence-transformers, rank-bm25, lxml, mlx-lm (sub-goals only), Claude Code CLI (`claude -p`, Pro plan, no API key).

**Spec:** `PLAN.md` (decisions D1-D10 + Results). Supersedes `CODEX_BRIEF_4.md`; read `BRIEF_3_REPORT.md` for current state.

## Global Constraints

- Run everything with `.venv/bin/python` from `/Users/srujanpendurthi/Projects/Current/lens`.
- **No git commits.** The repo root is `/Users/srujanpendurthi/Projects/Current`, which is shared with sibling projects and has no commits yet. Every "commit" step below is a test checkpoint instead.
- Never touch `../lagunaLab` or `../moeStream`. GPU/model runs go one at a time via `../.gpu/gpu-run lens ...` under `run_evals.py`'s swap watchdog.
- **Do not resume the Qwen3-4B picker run** (D10). Qwen remains only for sub-goal generation, which is fully cached for the 200-step sample.
- Fixed values: sample = seed-0 shuffled `data/steps.jsonl`; the Haiku sample is `--n 53`; the D7 gate uses recall@10 and top-10 picks; the Haiku model id is `claude-haiku-5-5`.
- Keep the MPS sequence-length caps in `lens/rank.py` (ColBERT query 32 / doc 64 own, 128 context; cross 128 own / 192 context; single 128). They are the out-of-memory fix.
- Existing Qwen cache keys (`pick-v2\n{MODEL}\n{text}`, `subgoal-v1\n{MODEL}\n{text}`) must stay byte-identical so `data/llm_cache.jsonl` keeps hitting.
- If a sandbox, Metal, network, CLI auth or usage limit blocks you, stop and report exactly what failed. Never bypass the sandbox.

## Review Focus

1. **Hidden context in `claude -p`:** hooks, plugins or CLAUDE.md files can be injected into every call and inflate input tokens. Expected: a 10-token prompt reports < 300 input tokens. Pinned in Task 3 Step 6.
2. **CLI error or usage limit mid-run:** non-zero exit, or JSON with `is_error: true`. Expected: raise, write nothing to the cache, and the run stops with completed rows already on disk. Pinned in Task 3 (`test_haiku_error_not_cached`) and Task 2 (incremental rows).
3. **Run killed partway (watchdog, Ctrl-C):** expected: rows for finished steps are on disk, and `--summarize` rebuilds the table from them. Pinned in Task 2.
4. **Picker answers with prose or a ref not shown:** expected: the first `[digits|none]` is parsed; a ref outside the shown list counts as wrong and is counted as hallucinated. Pinned in Task 4 (`test_hallucinated_ref`).
5. **Refactor silently changes ranker output:** expected: own-text recall on the first 20 steps matches `results_subgoal.jsonl` exactly. Pinned in Task 1 Step 5.

---

### Task 1: Audit cuts (code)

**Files:**
- Modify: `lens/rank.py` (delete the `rank()` try/finally `empty_cache` wrapper and rename `_rank` → `rank`; drop the `batch_size=64` args)
- Modify: `lens/pick.py` (extract the duplicated cache logic)
- Modify: `eval.py` (delete the `task` and `oracle` query modes; the query is always the sub-goal)
- Modify: `run_evals.py` (take args from argv instead of the hardcoded brief-3 runs)
- Test: `test_eval.py`

**Interfaces:**
- Produces: `rank(name: str, query: str, texts: list[str], node_text: str = "own") -> torch.Tensor`, the same signature as today.
- Produces: `lens.pick._cached(key: str, fn: Callable[[], dict]) -> dict`. It loads the jsonl into `_memo` once, returns the hit, or else calls `fn()`, sets `row["key"] = key`, appends it to `CACHE`, and returns it.
- Produces: `run_evals.py` CLI `run_evals.py LOG -- <eval.py args...>`, which runs `../.gpu/gpu-run lens .venv/bin/python eval.py <args>` with the existing watchdog.
- Produces: `eval.py` has no `--query` flag. The output default is `results_subgoal{_ctx}.jsonl`.

- [ ] **Step 1: Update tests.** In `test_eval.py`, delete `test_oracle_uses_operation_and_tag_without_gold_text` and `test_task_keeps_original_query`, and remove `--query` from `QueryTests.run_eval` args. Add:

```python
class CacheTests(unittest.TestCase):
    def test_cached_calls_once_and_persists(self):
        import lens.pick as picker
        calls = []
        with tempfile.TemporaryDirectory() as d, \
             patch.object(picker, "CACHE", Path(d) / "c.jsonl"), patch.object(picker, "_memo", {}):
            fn = lambda: calls.append(1) or {"v": 1}
            self.assertEqual(picker._cached("k", fn)["v"], 1)
            picker._memo.clear()
            self.assertEqual(picker._cached("k", fn), {"v": 1, "key": "k"})
            self.assertEqual(len(calls), 1)

    def test_qwen_pick_key_unchanged(self):
        import hashlib, lens.pick as picker
        self.assertEqual(picker.pick_key("x", "qwen"),
                         hashlib.sha1(f"pick-v2\n{picker.MODEL}\nx".encode()).hexdigest())
```

- [ ] **Step 2: Run them; the new ones fail.** Run `.venv/bin/python -m unittest test_eval -v`. Expected: `CacheTests` errors with `AttributeError: ... '_cached'` / `'pick_key'`.

- [ ] **Step 3: Implement.** Add `_cached` and `pick_key(text: str, picker: str = "qwen") -> str`. For qwen, `pick_key` returns the sha1 of `f"pick-v2\n{MODEL}\n{text}"`; for haiku, the sha1 of `f"pick-v2\n{HAIKU}\n{text}"`, with `HAIKU = "claude-haiku-5-5"`. Route `subgoal()` and `pick()` through `_cached`, keeping their cache keys unchanged. Apply the rank.py, eval.py and run_evals.py cuts listed under Files. `subgoal()` sets `row["miss"] = True` only on rows it generates (not stored in the cache); `eval.py` counts misses and prints `subgoal cache misses: N` before the summary.

- [ ] **Step 4: Run all tests.** `.venv/bin/python -m unittest test_eval -v && .venv/bin/python test_nodes.py && .venv/bin/python test_maxsim.py`. Expected: all pass.

- [ ] **Step 5: Regression check (real models, not mocked).** Run `.venv/bin/python eval.py --n 20 --no-llm --output /tmp/reg.jsonl` (sub-goals come from the cache), then:

```bash
.venv/bin/python -c "
import json
a=[json.loads(l) for l in open('/tmp/reg.jsonl')]; b=[json.loads(l) for l in open('results_subgoal.jsonl')][:20]
bad=[(i,n) for i,(x,y) in enumerate(zip(a,b)) for n in ('bm25','single','maxsim','cross') if x[n]['gold_rank']!=y[n]['gold_rank']]
print('mismatches', bad); assert not bad"
```

Expected: `mismatches []`. Also print a `subgoal cache misses: N` line; it must be 0.

- [ ] **Step 6: Checkpoint.** Re-run Step 4; no commit (see Global Constraints).

### Task 2: Crash-safe results + `--summarize`

**Files:**
- Modify: `eval.py`
- Test: `test_eval.py`

**Interfaces:**
- Consumes: Task 1's `eval.py` (no `--query`).
- Produces: the output jsonl gets one line per finished step, flushed immediately (the file is opened `"w"` at start).
- Produces: `eval.py --summarize PATH [--k 10]` loads rows from PATH and calls `summarize(rows, k, llm)`, with `llm = "baseline" in rows[0]`. Partial files are valid input.

- [ ] **Step 1: Write the failing tests.**

```python
class IncrementalTests(unittest.TestCase):
    def test_rows_on_disk_before_crash(self):
        # make rank raise on the 2nd step; the 1st row must already be written
        ...  # build 2 steps like QueryTests.run_eval; patch eval.rank side_effect=[t, t, t, t, RuntimeError("boom")]
        self.assertEqual(len(output.read_text().splitlines()), 1)

    def test_summarize_partial_file(self):
        # write 1 recall-only row; run `eval.py --summarize PATH`; stdout contains "steps=1" and "GATE:"
```

Assertions: after the induced crash the output has exactly 1 line, and `--summarize` stdout contains `steps=1` and `GATE: `.

- [ ] **Step 2: Run them.** `.venv/bin/python -m unittest test_eval.IncrementalTests -v`. Expected: FAIL (the file is empty because rows are written at the end; `--summarize` is unknown).

- [ ] **Step 3: Implement** the per-row write with `f.write(json.dumps(row) + "\n"); f.flush()`, and the `--summarize` branch.

- [ ] **Step 4: Run all tests** (Task 1 Step 4 command). Expected: pass.

- [ ] **Step 5: Checkpoint.**

### Task 3: Haiku picker backend via `claude -p`

**Files:**
- Modify: `lens/pick.py`
- Test: `test_eval.py`

**Interfaces:**
- Consumes: `_cached`, `pick_key` (Task 1).
- Produces: `pick(text: str, picker: str = "qwen") -> dict` with keys `ref, out, prompt_tokens, ttft, truncated, visible_refs`. For haiku, `truncated=False`, `visible_refs=None`, and `ttft` = `duration_api_ms / 1000` (total API time, not true TTFT; label it so in the summary).
- Produces: `parse_ref(out: str) -> str | None`: the first `\[(\d+|none)\]` match, case-insensitive, lowercased. Reuse it in the qwen path.
- Produces: `parse_claude_json(stdout: str) -> dict` returning `{"out": result, "prompt_tokens": input + cache_read + cache_creation input tokens, "ttft": duration_api_ms/1000}`. It raises `RuntimeError` if `is_error` is true or `result` is missing.
- Haiku command (stdin = prompt text):
  `claude -p --model claude-haiku-5-5 --tools "" --strict-mcp-config --setting-sources "" --no-session-persistence --output-format json --system-prompt "You pick one web page element. Reply with only its number in brackets, e.g. [123], or [none]."`
  Run it via `subprocess.run(..., input=text, capture_output=True, text=True, timeout=120)`. A non-zero exit raises `RuntimeError(stderr[-500:])`.

- [ ] **Step 1: Write the failing tests** (no network):

```python
OK = '{"type":"result","is_error":false,"result":"[42]","duration_api_ms":900,"usage":{"input_tokens":120,"cache_read_input_tokens":30,"cache_creation_input_tokens":0,"output_tokens":4}}'

def test_parse_claude_json(self):
    from lens.pick import parse_claude_json
    self.assertEqual(parse_claude_json(OK), {"out": "[42]", "prompt_tokens": 150, "ttft": 0.9})

def test_parse_ref_prose(self):
    from lens.pick import parse_ref
    self.assertEqual(parse_ref("The answer is [17] because"), "17")
    self.assertEqual(parse_ref("[None]"), "none"); self.assertIsNone(parse_ref("Search"))

def test_haiku_error_not_cached(self):
    # patch subprocess.run -> stdout='{"is_error":true,"result":"usage limit"}', returncode=0
    # pick("x", "haiku") raises RuntimeError and the CACHE file has no line for pick_key("x","haiku")
```

- [ ] **Step 2: Run them.** Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement** the haiku branch inside `pick()` through `_cached(pick_key(text, picker), ...)`. The exception propagates before anything is cached.

- [ ] **Step 4: Run all tests.** Expected: pass.

- [ ] **Step 5: Check the CLI version and auth.** Run `claude --version` and `echo 'Reply [7].' | <haiku command>`. Expected: JSON with `"result"` containing `[7]`. If auth fails or the sandbox blocks `~/.claude`/network, stop and report.

- [ ] **Step 6: Hidden-context check.** From that output, input tokens (`input + cache_read + cache_creation`) must be **< 300**. If higher, try adding `--bare` (if `claude --help` lists it) or `CLAUDE_CODE_DISABLE_*` env options shown in `claude --help`, and re-check. Do not continue to Task 6 until it is < 300. Report the final command.

- [ ] **Step 7: Checkpoint.**

### Task 4: Multi-k picks, rankers subset, hallucination + oracle-in-list metrics

**Files:**
- Modify: `eval.py`
- Test: `test_eval.py`

**Interfaces:**
- Consumes: `pick(text: str, picker: str) -> dict` and `parse_ref` from Task 3.
- Produces CLI: `--ks 10,50` (default `10`), `--rankers single,cross,maxsim` (default all of `RANKERS`; recall is computed for the listed rankers only), `--picker {qwen,haiku}` (default `qwen`).
- Produces row schema: `row[name] = {"ms", "gold_rank"}` for recall, and `row[f"{name}@{k}"] = pick_row | {"ok", "tokens", "shown": n_shown, "gold_shown": bool, "hallucinated": bool}` per pick arm. `row["baseline"]` gets the same added fields.
  - `ok` = ref ∈ gold and ref ∈ shown refs (and visible after truncation).
  - `hallucinated` = ref not None, ref ≠ "none", and ref ∉ shown refs.
  - `gold_shown` = any gold ref ∈ shown refs.
- Produces summary: one line per pick arm with acc, acc where gold was shown (`acc|shown`), hallucinated %, median tokens, median ttft, plus per-op acc (CLICK/TYPE/SELECT). The gate uses arm `maxsim@10` and prints `INCOMPLETE` if that arm is absent.

- [ ] **Step 1: Write the failing tests.**

```python
def test_hallucinated_ref(self):
    # picker returns ref "999" which is not among shown nodes
    row = ...  # run_eval with pick returning {"ref": "999", ...}, --ks 10
    self.assertFalse(row["single@10"]["ok"]); self.assertTrue(row["single@10"]["hallucinated"])

def test_multi_k_arms_and_subset(self):
    row = ...  # --ks 10,50 --rankers single
    self.assertIn("single@10", row); self.assertIn("single@50", row); self.assertNotIn("cross", row)

def test_gate_incomplete_without_maxsim10(self):
    # summarize rows that have no "maxsim@10" key, llm=True -> "GATE: INCOMPLETE"
```

- [ ] **Step 2: Run them.** Expected: FAIL (unknown flags / KeyError).

- [ ] **Step 3: Implement.** Update the existing `GateTests`/`QueryTests` that relied on `row[name]["ok"]`, so they read `row[f"{name}@10"]`.

- [ ] **Step 4: Run all tests.** Expected: pass.

- [ ] **Step 5: Checkpoint.**

### Task 5: Run B, context recall (no picker)

**Files:** none modified. Produces `data/recall_subgoal_ctx.log` and `results_subgoal_ctx.jsonl`.

- [ ] **Step 1: Run** `.venv/bin/python run_evals.py data/recall_subgoal_ctx.log -- --n 200 --no-llm --steps data/steps_ctx.jsonl --node-text context`. Expected: 200 rows, a `GATE:` line, and `subgoal cache misses: 0`.
- [ ] **Step 2: Compare.** Compute R@5/10/20/50/100 per ranker from `results_subgoal_ctx.jsonl` vs `results_subgoal.jsonl` (own text, brief 2). Put both tables side by side in the report.

### Task 6: Haiku picker run (53 steps)

**Files:** none modified. Produces `data/haiku_53.log` and `results_haiku_53.jsonl`.

- [ ] **Step 1: Run** (no GPU lock needed for the picker; rankers use MPS briefly, so keep the watchdog): `.venv/bin/python run_evals.py data/haiku_53.log -- --n 53 --picker haiku --ks 10,50 --rankers single,cross,maxsim --output results_haiku_53.jsonl`. That is 53 × 7 = 371 calls, sequential. On a usage-limit error, stop and report rows done; rerun the same command later to resume (cached picks are skipped).
- [ ] **Step 2: Summarize.** `.venv/bin/python eval.py --summarize results_haiku_53.jsonl`. Paste it verbatim.

### Task 7: Docs cleanup + report

**Files:**
- Delete: `EVAL_REPORT.md` (stale run-1 KILL), `CODEX_BRIEF_2.md`, `CODEX_BRIEF_3.md`, `CODEX_BRIEF_4.md` (executed or superseded; their reports and PLAN.md keep the outcomes)
- Modify: `claude_guidance_file.md`: replace §3 "Codebase Standards", §4 "Protocol Control-Flow Pattern" and §5 "Primary Contribution Areas" with a 5-line "Current status" section. It says v1 is the offline eval; the multi-vector index, quantization, CUDA and yield-and-inspect work is deferred per PLAN.md D5/D8; and the runner and picker details live in PLAN.md.
- Create: `BRIEF_5_REPORT.md`

- [ ] **Step 1: Apply** the deletions and the guidance edit.
- [ ] **Step 2: Write `BRIEF_5_REPORT.md`** with:
  - (a) the Task 1 regression result;
  - (b) the Task 5 side-by-side recall table;
  - (c) the Task 6 summary verbatim, labelled "53 steps: gate indicative only";
  - (d) the final `claude -p` command, CLI version and hidden-context token count;
  - (e) code changes (file:line + why);
  - (f) the test status;
  - (g) anything surprising.
- [ ] **Step 3: Conditional (do NOT do this, only report it):** if context-mode MaxSim R@10 ≤ single R@10 *and* `maxsim@10` acc ≤ `single@10` acc, write in the report: "maxsim + pylate removable (audit item 8)". Removing them is a planning decision.

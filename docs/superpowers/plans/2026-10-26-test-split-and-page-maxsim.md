# Test Split + Whole-Page Contextual MaxSim (Vision Piece 1) Implementation Plan

> **Executor: Codex (`gpt-6.1-sol`).** Execute the tasks yourself in order (there are no sub-skills or subagents). Each task runs TDD: failing test, verify fail, implement, verify pass, commit. Tick each `- [x]` → `- [x]` in this file as you finish it, and commit the ticks with that task's commit, so another agent can resume exactly where you stop. Read **Codex execution notes** below before starting.

**Goal:** Move the eval to the official Multimodal-Mind2Web `test_task` split (saving screenshots for vision piece 2), re-measure the four rankers there, then test whether whole-page contextual MaxSim beats single-vector recall@10 by ≥ 5 points (gate D12).

**Architecture:** Per-node rankers embed each element's text alone. `pagemaxsim` instead encodes the page's visible text in overlapping fixed-length windows with ColBERT, so every token vector carries its page neighbors. Each token keeps the ref of the candidate element it belongs to, and a node's score is MaxSim over that node's tokens across all windows. Everything else (sub-goal query, recall metrics, `--summarize`) is reused.

**Tech Stack:** Python 3.13 `.venv`, torch (rankers on `LENS_DEVICE=cpu`, per D11), pylate ColBERT `answerdotai/answerai-colbert-small-v1`, lxml, pyarrow (new), mlx-lm (Qwen sub-goals only).

**Spec:** `PLAN.md` D12 (plus D1-D11 for context). Later plans cover vision piece 2 (ColPali) and piece 3 (fine-tuning); this plan only saves screenshots for piece 2.

## Global Constraints

- Run from `/Users/srujanpendurthi/Projects/Current/lens` with `.venv/bin/python`. lens is its own repo (`origin` = github.com/SrujanPendurthi/lens, `main`).
- Commit at each task's last step: a conventional message ending in `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Stage only the files that task touched, never `data/`. Push after Task 4. No force-push.
- Never touch `../lagunaLab`, `../moeStream`, `../.gpu` contents. Model runs go through `run_evals.py LOG -- <eval args>` (gpu-run lock + swap watchdog).
- Rankers run with `LENS_DEVICE=cpu` (D11: MPS graph-cache out-of-memory). Encode ms on CPU is not a latency claim.
- Do not change existing ranker caps, the cache keys, or the D7 gate code. The Qwen picker stays retired (D10). No Haiku runs in this plan.
- Fixed values:
  - Split `test_task`; sample = seed-0 shuffle, first **200** steps with gold surviving extraction.
  - pagemaxsim: window **64 words**, stride **32 words**, ColBERT document length **128** tokens, query length **32**.
  - Gate: `pagemaxsim` R@10 ≥ `single` R@10 + **5.0** pts on the same 200 steps.
- If the download, sandbox, Metal, or memory blocks a run, stop and report the exact error. Don't bypass the sandbox.

## Codex execution notes

- **Sandbox needs:**
  - Task 2 Step 1 downloads from Hugging Face, so it needs network access.
  - `run_evals.py` / `../.gpu/gpu-run` need execution outside the workspace sandbox (the lock lives in `../.gpu/`); request approval as in brief 5.
  - The model caches in `~/.cache/huggingface` are already populated for ColBERT, bge, the cross-encoder and Qwen.
  - If any of these is denied, stop and report it. Don't work around it.
- **`ps` is blocked** in your sandbox. To wait on a background run, poll its log for the `GATE:` line (e.g. every 60 s) instead of checking PIDs.
- **Long runs (Task 4 Step 1, roughly 30-60 min on CPU):** launch with `nohup <command> > /dev/null 2>&1 &` (the log goes to the run_evals LOG path), then poll. Rows are written incrementally, so a stopped run keeps its finished steps; `eval.py --summarize` works on partial files.
- **Low on usage?** Stop at a task boundary, commit, push, and put one status line at the top of `BRIEF_7_REPORT.md` (create it if needed): `STATUS: stopped after Task N; next = Task N+1 Step M`. A Sonnet agent will resume from the ticked checkboxes.
- **Deviations:** if the real data or library contradicts the plan (e.g. pylate exposes the `[D]` marker differently), make the smallest change that satisfies the plan's tests and intent, and list it under "Deviations" in the report. Don't change the fixed values, the gate, or the D1-D12 decisions.
- **Final message:** under 25 lines. Give the D12 verdict line, the R@10 row for each ranker, dataset facts, the commit hashes, the push status, and any deviations or blockers.

## Review Focus

1. **The multimodal parquet stores `operation` / `pos_candidates` / `neg_candidates` as JSON strings,** not dicts. Prep must parse both forms. Pinned: Task 2 `test_prep_parses_json_string_fields`.
2. **A token can belong to nested candidates** (a `span` inside a candidate `a`). It is attributed to the *innermost* candidate ancestor. Text outside any candidate gives context but scores no node. Pinned: Task 1 `test_page_words_innermost_candidate`.
3. **Textless gold nodes** (icon buttons) have no tokens. Their pagemaxsim score is `-inf` (ranked last), not a crash, and they are counted in `textless gold`. Pinned: Task 3 `test_textless_node_ranks_last`.
4. **Token/word misalignment:** subword tokens, the `[D]` marker, and padding must not shift refs. Every non-special token maps to its source word's ref. Pinned: Task 3 `test_alignment_counts`.
5. **The same node appearing in two overlapping windows:** take the max per query token across all occurrences, without double-counting. Pinned: Task 3 `test_overlap_takes_max`.

---

### Task 1: `page_words`, page text with innermost-candidate refs

**Files:**
- Modify: `lens/nodes.py`
- Test: `test_nodes.py`

**Interfaces:**
- Produces: `page_words(cleaned_html: str, refs: list[str]) -> list[tuple[str, str | None]]`. It returns whitespace-split words of the page's visible text in document order, so each element's `.text` and `.tail` sit in their correct positions. Each word is tagged with the ref of its innermost ancestor-or-self element whose `backend_node_id` is in `refs`, otherwise `None`. Attribute text (`aria_label`, `placeholder`, `value`, `alt`, `title`) of a candidate is emitted as words at that element's position, tagged with its ref, so attribute-only nodes still get tokens.

- [x] **Step 1: Write the failing tests** (append to `test_nodes.py`, assert style):

```python
from lens.nodes import page_words
html = '<html><body>Flights <div backend_node_id="1">noon <a backend_node_id="2"><span>View details</span></a></div> after</body></html>'
w = page_words(html, ["1", "2"])
assert w == [("Flights", None), ("noon", "1"), ("View", "2"), ("details", "2"), ("after", None)], w  # test_page_words_innermost_candidate
w2 = page_words('<html><body><button backend_node_id="3" aria_label="Close"/></body></html>', ["3"])
assert w2 == [("Close", "3")], w2
```

- [x] **Step 2: Run it.** `.venv/bin/python test_nodes.py`. Expected: FAIL (`ImportError: page_words`).
- [x] **Step 3: Implement it** with lxml `iter()` plus `.text`/`.tail` handling and an ancestor walk (or a ref stack during a recursive walk).
- [x] **Step 4: Run all tests** (`.venv/bin/python -m unittest test_eval -v && .venv/bin/python test_nodes.py && .venv/bin/python test_maxsim.py`). Expected: PASS.
- [x] **Step 5: Commit** with `feat: page_words maps page tokens to innermost candidate`.

### Task 2: Prep the Multimodal-Mind2Web test split

**Files:**
- Modify: `eval.py` (add `prep_mm`, plus the `--prep-mm DIR` flag that writes `--steps`)
- Modify: `requirements.txt` (add `pyarrow`)
- Test: `test_eval.py`

**Interfaces:**
- Produces: `prep_mm(parquet_dir: str, steps_path: str, shots_dir: str = "data/shots") -> None`.
  - It writes one jsonl row per action with keys `task, prev, op, gold, nodes, uid, page`. `nodes` comes from `extract` and `page` from Task 1's `page_words`, and `page` is a list of `[word, ref_or_null]` pairs.
  - It saves `shots_dir/{action_uid}.png` from the `screenshot` column; skip saving if the file exists.
  - `prev` = `action_reprs[:target_action_index][-3:]`.
  - Rows whose gold isn't in `nodes` are kept (coverage is checked by the existing eval check).
- Produces: `.venv/bin/python eval.py --prep-mm data/mm_test_task --steps data/steps_test.jsonl`.

- [ ] **Step 1: Download** only the `test_task` parquet shards into `data/mm_test_task/`, about 2.3 GB, via `huggingface_hub.snapshot_download(repo_id="osunlp/Multimodal-Mind2Web", repo_type="dataset", allow_patterns="data/test_task-*", local_dir="data/mm_test_task")`. Print one row's field types so you can see which fields are JSON strings.

- [ ] **Step 2: Write the failing test.**

```python
def test_prep_parses_json_string_fields(self):
    # build a 1-row parquet with operation/pos_candidates/neg_candidates as JSON *strings*,
    # cleaned_html '<html><body><button backend_node_id="7">Go</button></body></html>',
    # action_reprs ["a","b","c","d"], target_action_index 2, screenshot {"bytes": <tiny PNG>}
    rows = ...  # prep_mm(tmpdir, out, shots) then read out
    self.assertEqual(rows[0]["gold"], ["7"])
    self.assertEqual(rows[0]["prev"], ["a", "b"])
    self.assertEqual(rows[0]["op"]["op"], "CLICK")
    self.assertTrue((Path(shots) / f'{rows[0]["uid"]}.png').exists())
    self.assertIn(["Go", "7"], rows[0]["page"])
```

- [ ] **Step 3: Run it.** `.venv/bin/python -m unittest test_eval -k prep_parses -v`. Expected: FAIL (`prep_mm` not defined).
- [ ] **Step 4: Implement** `prep_mm`, reading with `pyarrow.parquet`.
- [ ] **Step 5: Run all tests:** `.venv/bin/python -m unittest test_eval -v && .venv/bin/python test_nodes.py && .venv/bin/python test_maxsim.py`. Expected: PASS.
- [ ] **Step 6: Build the real file.** `.venv/bin/python eval.py --prep-mm data/mm_test_task --steps data/steps_test.jsonl`. Report the row count (expect 1339), the gold-coverage line from a `--no-llm --n 1` dry run, and the shots count.
- [ ] **Step 7: Commit** with `feat: prep Multimodal-Mind2Web test split with screenshots`.

### Task 3: `pagemaxsim` ranker

**Files:**
- Modify: `lens/rank.py` (add `rank_page`; add `"pagemaxsim"` to the allowed names, but **not** to the default `RANKERS` tuple)
- Modify: `eval.py` (`--rankers` accepts `pagemaxsim`; for it, call `rank_page(query, s["page"], [n.ref for n in nodes])`)
- Test: `test_eval.py`

**Interfaces:**
- Consumes: `page_words` output stored in `s["page"]` (Task 2).
- Produces: `rank_page(query: str, words: list[tuple[str, str | None]], refs: list[str], window: int = 64, stride: int = 32) -> torch.Tensor`, one score per entry of `refs`, in order, with `-inf` for refs that have no tokens.
- Produces: `encode_doc_tokens(texts_as_word_lists: list[list[str]]) -> list[tuple[torch.Tensor, list[int | None]]]`, giving the normalized per-token embeddings `[L, d]` plus each token's source word index (`None` for special/marker/pad tokens).
  - Use the pylate ColBERT model from `_model("maxsim", "context")` (doc length 128).
  - Get token-level embeddings **before** pylate's punctuation skiplist so indices stay aligned. Tokenize with `is_split_into_words=True` and use `BatchEncoding.word_ids()`.
  - Verify how pylate adds its `[D]` document marker, and reproduce it (marker token → `None`).
  - Pad every window to the fixed 128 tokens.
- Algorithm (not determined by the tests):

```text
for each window of `words` (start in range(0, len, stride), slice [start:start+window]):
    emb, word_idx = encode_doc_tokens([window words])[0]
    for each token t with word_idx not None and ref = words[start+word_idx][1] not None:
        best[ref] = elementwise max(best[ref], q @ emb[t])   # q: [Lq, d] query tokens (pylate is_query=True)
score[ref] = best[ref].sum() if ref in best else -inf
```

- [ ] **Step 1: Write the failing tests** (add a `PageMaxSimTests` class; the real model loads on CPU, 33M params, about seconds):

```python
def test_alignment_counts(self):
    emb, idx = rank.encode_doc_tokens([["View", "details", "noon"]])[0]
    self.assertEqual(emb.shape[0], len(idx))
    self.assertEqual(sorted({i for i in idx if i is not None}), [0, 1, 2])

def test_relevant_node_wins(self):
    words = [("Search", "1"), ("button", "1"), ("Privacy", "2"), ("policy", "2")]
    s = rank.rank_page("click the search button", words, ["1", "2"])
    self.assertGreater(s[0], s[1])

def test_textless_node_ranks_last(self):
    s = rank.rank_page("search", [("Search", "1")], ["1", "9"])
    self.assertEqual(s[1].item(), float("-inf"))

def test_overlap_takes_max(self):
    words = [("w%d" % i, None) for i in range(40)] + [("Search", "1")] + [("w", None)] * 40
    a = rank.rank_page("search", words, ["1"], window=64, stride=32)
    b = rank.rank_page("search", words, ["1"], window=64, stride=64)
    self.assertGreaterEqual(a[0].item(), b[0].item() - 1e-4)   # overlap never lowers a node's score
```

- [ ] **Step 2: Run them.** `LENS_DEVICE=cpu .venv/bin/python -m unittest test_eval.PageMaxSimTests -v`. Expected: FAIL (`AttributeError: encode_doc_tokens`).
- [ ] **Step 3: Implement** `encode_doc_tokens`, `rank_page`, and the eval wiring.
- [ ] **Step 4: Run all tests** (Task 2 Step 5 command, with `LENS_DEVICE=cpu`). Expected: PASS.
- [ ] **Step 5: Commit** with `feat: whole-page contextual MaxSim ranker (vision piece 1)`.

### Task 4: Runs + report

**Files:**
- Create: `BRIEF_7_REPORT.md`

- [ ] **Step 1: Sub-goals for the new steps.** They are uncached, so Qwen generates them once. Run `LENS_DEVICE=cpu .venv/bin/python run_evals.py data/recall_test.log -- --n 200 --no-llm --steps data/steps_test.jsonl --rankers bm25,single,maxsim,cross,pagemaxsim --output results_test_200.jsonl`. Expected: 200 rows and a `GATE:` line. `subgoal cache misses` should be ≤ 200 on this run and 0 on any rerun.
- [ ] **Step 2: Gate.** From `results_test_200.jsonl`, compute R@5/10/20/50/100 per ranker plus the D12 verdict line: `D12 piece 1: pagemaxsim R@10 = X, single R@10 = Y, margin = X-Y → PASS if ≥ 5.0 else FAIL`. Also report pagemaxsim median ms per step (CPU) and its R@10 on steps where gold is textless vs has text.
- [ ] **Step 3: Write `BRIEF_7_REPORT.md`** with:
  - (a) the dataset facts: rows, coverage, shots count, and disk use of `data/mm_test_task`;
  - (b) the verbatim `--summarize` output;
  - (c) the R@k table and the D12 verdict line;
  - (d) a comparison against the train-shard brief 2 numbers (own text);
  - (e) code changes (file:line + why) and test status;
  - (f) anything surprising.
  Numbers only; no interpretation beyond them.
- [ ] **Step 4: Commit** `results_test_200.jsonl` + `BRIEF_7_REPORT.md` with `docs: brief 7 results (test split, pagemaxsim gate)`, then `git push`.

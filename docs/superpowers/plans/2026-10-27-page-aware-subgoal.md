# Close Multi-Vector + Page-Aware Sub-Goal (D18) Implementation Plan

> **Executor: Codex (`gpt-6.1-sol`).** Execute the tasks yourself in order. Each code task runs TDD: failing test, verify fail, implement, verify pass, commit. Tick each `- [ ]` → `- [x]` in this file as you finish it and commit the ticks with that task's commit. Read **Codex execution notes** before starting.

**Goal:**
1. Cut the closed MaxSim path.
2. Test D18: does a sub-goal written **after seeing the page's top-50 candidates** raise zero-shot single-vector recall@10 over today's blind sub-goal?

**Architecture:** two-stage query, for each step:
1. **Stage 1:** today's blind `subgoal()` query ranks all nodes with zero-shot `single`.
2. **Stage 2:** new `subgoal_seen()` gives Qwen3-4B the task, previous actions and the stage-1 top-50 as `[ref] tag "text"` lines, best first. Qwen writes one line naming the element by its visible text.
3. That line re-ranks all nodes with `single`.

Selected by `eval.py --query seen`; the default stays `subgoal`.

**Tech Stack:** Python `.venv`, mlx-lm Qwen3-4B (existing, cached via `data/llm_cache.jsonl`), sentence-transformers bge-small (existing). No picker calls, no Haiku quota.

**Spec:** `PLAN.md` D17, D18. Sources: `BRIEF_6` results (via D17), `BRIEF_11_REPORT.md`, the 2026-10-27 caveman review and ponytail audit.

## Global Constraints

- **Repo rules:**
  - Commit per task; push `origin main` after each task (no force-push).
  - Never stage `data/` except `data/llm_cache.jsonl`, which gets its own `chore: llm cache` commit.
  - Never touch `../lagunaLab`, `../moeStream`, `../.gpu` contents.
  - Model jobs go through `run_evals.py`.
- **Sample:** the same 200 `test_task` steps as `results_test_200.jsonl`. The stage-1 sub-goals are cached, so 0 misses.
- **Fixed values:**
  - stage-1 shortlist = 50 nodes;
  - Qwen `max_tokens=24` (same as `subgoal`);
  - ranker for both stages = zero-shot `single`, `LENS_DEVICE=cpu`;
  - one prompt, worded exactly as in Task 2. No prompt tuning.
- **Gate (D18):**
  - Compare step by step against `single` in `results_test_200.jsonl` (53/200 at R@10). Gained = new hit and old miss; lost = the reverse.
  - **PASS** if `gained − lost ≥ 5` and the one-sided sign test `P(X ≥ gained | n = gained + lost, p = 0.5) < 0.05`, computed with `math.comb`. Else **FAIL**.
- **Cost accounting:** the seen query's cost = stage-1 sub-goal tokens + stage-2 tokens. Report medians.
- **Do not touch:** `models/single_ft`, `data/triplets_train.jsonl`, `triplets()`.

## Codex execution notes

- **Approvals:** `run_evals.py` (GPU lock, Qwen on MLX) needs execution outside the sandbox; request approval. If denied, stop and report.
- **`ps` is blocked:** poll logs every 60 s.
- **Run time:** about 200 new Qwen calls at roughly 700-token prompts, ~2 s each, under 10 min. A rerun resumes from the cache.
- **Low on usage?** Stop at a task boundary, commit, push, and put `STATUS: stopped after Task N; next = Task N+1 Step M` at the top of `BRIEF_12_REPORT.md`. A Sonnet agent resumes from the ticks.
- **Deviations:** smallest change keeping tests and intent; list them in the report. Never change the prompt, shortlist size, gate or sample.
- **Final message:** under 15 lines: gained/lost/net, the sign-test p-value, the D18 verdict line, R@10/R@50 old vs new, median query tokens, commits, push status, deviations.

## Review Focus

1. **The shortlist must be in stage-1 rank order and capped at 50.** A page with fewer than 50 nodes passes all of them. Pinned: Task 2 `test_seen_query_uses_ranked_top50`.
2. **The cache key must differ from `subgoal`'s**, or seen outputs would collide with blind ones in `data/llm_cache.jsonl`. Pinned: Task 2 `test_subgoal_seen_cached_separately`.
3. **The default `--query subgoal` must stay byte-identical** for existing runs: same query, same rows. Pinned: the existing `QueryTests` pass unchanged.
4. **Textless nodes in the shortlist** render as `[ref] tag ""`, never dropped, so refs stay aligned with ranks. Pinned: covered by `test_seen_query_uses_ranked_top50` (one textless node in the fixture).
5. **Old result files after the MaxSim cut:** `--summarize results_test_200.jsonl` still prints bm25/single/cross rows unchanged. Pinned: Task 1 Step 4.

---

### Task 1: Close the multi-vector path + audit cuts

**Files:**
- Modify: `lens/rank.py`, `train.py`, `eval.py`, `test_eval.py`, `requirements.txt`, `claude_guidance_file.md`, `PLAN.md`
- Delete: `test_maxsim.py`, `docs/superpowers/plans/2026-10-27-cleanup-and-website-check.md`

**Interfaces:**
- Produces:
  - `RANKERS = ("bm25", "single", "cross")`;
  - `ALLOWED_RANKERS = RANKERS + ("single_ft",)`;
  - `summarize(rows, llm)` with no D12 lines.

- [ ] **Step 1:** Cut the MaxSim path:
  - `rank.py`: `maxsim()`, the `maxsim` branches, and the `pylate` import.
  - `train.py`: the `maxsim` branch of `fit()`; `--fit` choices become `("single",)`.
  - `test_maxsim.py`.
  - `pylate` in `requirements.txt`.
- [ ] **Step 2:** Cut the D12 margin block in `summarize` and `MarginTests`, except `test_haiku_pick_key_unchanged`, which moves into `CacheTests`. In `test_ft_models_keep_zero_shot_caps`, drop the `maxsim_ft` half.
- [ ] **Step 3: Docs:**
  - `claude_guidance_file.md` §3 becomes one line: `Status and decisions: PLAN.md (latest D18).`
  - Append to the D16 outcome in `PLAN.md`: `Paired: single 14 gained vs 12 lost (p≈0.85), so the single fine-tune lift is indistinguishable from 0 on unseen sites; maxsim 23 vs 7 (p≈0.005) only catches up to zero-shot single (55 vs 52). test_website = 10 sites, so the effective sample is ~10, not 200.`
  - Delete the finished plan doc.
- [ ] **Step 4: Verify:**
  - Run `LENS_DEVICE=cpu .venv/bin/python -m unittest test_eval test_train -v && .venv/bin/python test_nodes.py`. Expected: PASS.
  - `grep -rn "maxsim\|pylate" lens eval.py train.py requirements.txt` prints nothing.
  - `.venv/bin/python eval.py --summarize results_test_200.jsonl` prints bm25/single/cross R@5/10/20 rows identical to `BRIEF_7_REPORT.md`.
- [ ] **Step 5: Commit** `refactor: close multi-vector path (D17)` and push.

### Task 2: `subgoal_seen` + `--query seen`

**Files:**
- Modify: `lens/pick.py`, `eval.py`, `test_eval.py`

**Interfaces:**
- Consumes: `subgoal(task, prev) -> dict` (`text`, `tokens`, `ms`, optional `miss`); `rank("single", query, texts) -> Tensor`; `Node.line()`.
- Produces:
  - `subgoal_seen(task: str, prev: list[str], shortlist: list[Node]) -> dict`, with the same fields as `subgoal`. Cache key `sha1(f"subgoal-seen-v1\n{MODEL}\n{prompt}")`.
  - `eval.py --query {subgoal,seen}` (default `subgoal`).
  - Rows gain `"first_query"` (the stage-1 text) when the mode is seen; `row["query"]` is the seen text.
  - `subgoal_tokens` / `subgoal_ms` hold stage-1 + stage-2 sums.
  - Seen-mode misses count toward `subgoal cache misses`.

- [ ] **Step 1: Write the failing tests** (`test_eval.py`):
  - `test_seen_query_uses_ranked_top50`:
    - one step with 60 nodes (refs `"0"`–`"59"`, node `"5"` textless, gold `"1"`);
    - patch `lens.pick.subgoal` → `{"text": "blind", "tokens": 80, "ms": 10}`;
    - patch `eval.rank` with a function returning `torch.arange(60.)` for query `"blind"` and `torch.zeros(60)` otherwise;
    - patch `lens.pick.subgoal_seen` → `{"text": "click Go", "tokens": 700, "ms": 50}`;
    - run with `--query seen --no-llm --rankers single`.
    - Assert:
      - `subgoal_seen` got a shortlist of exactly 50 nodes whose refs are the 50 highest-scoring under `"blind"`, in descending score order;
      - the textless node is kept if it ranks in;
      - `row["query"] == "click Go"`, `row["first_query"] == "blind"`, `row["subgoal_tokens"] == 780`;
      - the final `rank` call used `"click Go"`.
  - `test_subgoal_seen_cached_separately`:
    - Patch `CACHE`, `_memo` and `_load` like `test_subgoal_miss_is_transient`.
    - The `subgoal_seen` key differs from the `subgoal` key for the same task/prev.
    - A second call is a cache hit (no `miss`).
    - The prompt passed to the tokenizer contains `[7] button "Go"` for a shortlist `[Node("7", "button", "Go")]`.
- [ ] **Step 2: Run** `LENS_DEVICE=cpu .venv/bin/python -m unittest test_eval -v`. Expected: the 2 new tests FAIL.
- [ ] **Step 3: Implement:**
  - `lens/pick.py`:
    - Move the body of `subgoal`'s `generate` into `_qwen_line(key: str, text: str) -> dict`. It returns the cached row with `"miss": True` on generation; both `subgoal` and `subgoal_seen` call it.
    - `subgoal`'s key and prompt stay byte-identical.
    - The `subgoal_seen` prompt is exactly:
      ```
      Task: {task}
      Previous actions: {'; '.join(prev) or 'none'}
      Candidate elements on the page (most likely first):
      {one Node.line() per shortlist node}

      In one short line, what is the next thing to do on the page? Name the element by its visible text and the action, e.g. "click the Search button" or "type Boston into the location box".
      ```
  - `eval.py`: with `--query seen`, after the blind sub-goal:
    - `order = rank("single", goal["text"], texts).argsort(descending=True)`;
    - `shortlist = [nodes[i] for i in order[:50]]`;
    - call `subgoal_seen`, then continue with its text as `query`.
- [ ] **Step 4: Run all tests** (`test_eval test_train test_nodes`). Expected: PASS, including the unchanged `QueryTests` and `test_haiku_pick_key_unchanged`.
- [ ] **Step 5: Commit** `feat: page-aware sub-goal query (D18)` and push.

### Task 3: Run, gate, record

**Files:**
- Create: `results_seen.jsonl`, `BRIEF_12_REPORT.md`
- Modify: `PLAN.md` (D18 outcome, a Brief 12 results line)

- [ ] **Step 1: Run** `LENS_DEVICE=cpu .venv/bin/python run_evals.py data/seen.log -- eval.py --n 200 --no-llm --rankers single --query seen --output results_seen.jsonl`.
  - Expected: 200 rows.
  - Every `first_query` equals the `query` of the same row in `results_test_200.jsonl` (blind cache hits). If not, STOP.
- [ ] **Step 2:** Compute, step by step against `results_test_200.jsonl` `single`:
  - gained, lost and net at R@10;
  - the sign-test p-value;
  - R@5/10/20/50/100 old vs new;
  - the share of steps where the seen query contains the gold node's text verbatim (case-insensitive), as a "Qwen named it" diagnostic;
  - median query tokens and ms, old vs new;
  - R@10 by op.
- [ ] **Step 3: Write `BRIEF_12_REPORT.md`:**
  - the verbatim `eval.py --summarize results_seen.jsonl`;
  - the Step 2 tables;
  - 5 example steps (task, blind query, seen query, gold line);
  - the verdict line `D18: net <+n> hits (gained <g>, lost <l>, p=<p>) → PASS|FAIL`;
  - deviations.
- [ ] **Step 4: `PLAN.md`:**
  - Under D18, add `Outcome: PASS|FAIL; R@10 <a>/200 vs 53/200; net <n>, p=<p>`.
  - Add a `- Brief 12 (...)` line to Results.
- [ ] **Step 5: Commit** `docs: brief 12 page-aware sub-goal (D18)` + `chore: llm cache` and push. **Stop.** On PASS the planner writes the D17 Haiku rerun; on FAIL, the next step is another re-grill.

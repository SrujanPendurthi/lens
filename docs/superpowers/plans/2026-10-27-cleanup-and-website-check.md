# Brief 10 Cleanup + Unseen-Website Check Implementation Plan

> **Executor: Codex (`gpt-6.1-sol`).** Execute the tasks yourself in order. Each task runs TDD where code changes: failing test, verify fail, implement, verify pass, commit. Tick each `- [ ]` → `- [x]` in this file as you finish it and commit the ticks with that task's commit. Read **Codex execution notes** before starting.

**Goal:**
1. Apply the brief 10 review and audit cuts.
2. Measure whether fine-tuning's recall gain holds on **unseen websites** (`test_website` split). This is the one open fact the thesis re-grill (D12: all three pieces failed) needs.

**Architecture:**
- **Cleanup:** removes spent tests and redundant code. Untracked data is kept (user decision, 2026-10-27).
- **Unseen-website check:** reuses the existing prep, the cached Qwen sub-goals and the four existing rankers on a new split. It adds no new code paths.

**Tech Stack:** Python `.venv`, existing `eval.py` / `lens/` / `train.py`; `huggingface_hub` for the download.

**Spec:** `PLAN.md` D12, D15, D16. Sources: `BRIEF_10_REPORT.md`, the 2026-10-27 caveman review, ponytail review and ponytail audit.

## Global Constraints

- **Repo rules:**
  - Commit per task; push `origin main` after each task (no force-push).
  - Never stage `data/` except `data/llm_cache.jsonl`.
  - Never touch `../lagunaLab`, `../moeStream`, `../.gpu` contents.
  - Model jobs go through `run_evals.py` one at a time.
- **D15 is frozen:**
  - Do not retrain.
  - Do not touch `models/`, `data/triplets_train.jsonl`, or `triplets()`.
- **Sample:** `test_website`, seed-0 shuffle + gold-survival filter, first 200 (same rule as `test_task`).
- **Rankers:** `single,maxsim,single_ft,maxsim_ft` with `LENS_DEVICE=cpu`. No picker (`--no-llm`).
- **Decision rule (D16), pre-registered:**
  - If `single_ft` R@10 − `single` R@10 on `test_website` is ≤ 0 hits, fine-tuning's gain is site memorization.
  - Otherwise, it generalizes.
  - Report the same lift for maxsim. There is no other gate.

## Codex execution notes

- **Approvals:** the download needs network; `run_evals.py` needs execution outside the sandbox. If denied, stop and report.
- **`ps` is blocked:** poll logs every 60 s.
- **Sub-goals:** the 200 sampled website steps need new Qwen sub-goals, about 200 misses (~3 min). They are cached in `data/llm_cache.jsonl`; commit it as `chore: llm cache`.
- **Low on usage?** Stop at a task boundary, commit, push, and put `STATUS: stopped after Task N; next = Task N+1 Step M` at the top of `BRIEF_11_REPORT.md`. A Sonnet agent resumes from the ticks.
- **Deviations:** smallest change keeping tests and intent; list them in the report. Never change the sample, rankers or decision rule.
- **Final message:** under 15 lines: website overlap facts, the R@10 table, the D16 verdict line, commits, push status, deviations.

## Review Focus

1. **Zero-shot rankers getting D12 verdicts against `single_ft`:** no comparison like that was pre-registered. When the base is `single_ft`, only `_ft` rankers print D12 lines. Pinned: Task 1 `test_d12_lines_only_for_ft_when_base_is_ft`.
2. **Website leakage:** a `test_website` site that also appears in train would turn "unseen" into "seen". Expected: 0 shared websites. If not 0, STOP. Pinned: Task 2 Step 2.
3. **Gold coverage on a new split under 98%:** `eval.py` already refuses to run; report the number instead of lowering the bar. Pinned: the existing guard, checked in Task 2 Step 3.
4. **The `_ft` checkpoint-path shrink must load the same dirs:** pinned by the existing `test_ft_models_keep_zero_shot_caps`, which must still pass unchanged.

---

### Task 1: Review and audit cuts

**Files:**
- Modify: `eval.py`, `lens/rank.py`, `train.py`, `test_train.py`, `test_eval.py`, `PLAN.md`
- Delete: `docs/superpowers/plans/2026-10-27-cleanup-and-finetune.md`, `results_test_colpali.jsonl`

**Interfaces:**
- Produces: `summarize(rows, llm)`.
  - With `single_ft` present, it prints `D12` lines only for rankers whose names end in `_ft`.
  - Otherwise, it prints them for every non-`single` ranker (unchanged).

- [x] **Step 1: Write the failing test** in `MarginTests`:
  ```python
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
  ```
- [x] **Step 2: Run** `LENS_DEVICE=cpu .venv/bin/python -m unittest test_eval -v`. Expected: the new test FAILS.
- [x] **Step 3: Cuts:**
  - `eval.py`:
    - add the D12 filter;
    - delete the `ks = a.ks` alias (use `a.ks`);
    - move `from pathlib import Path` back to the top-level imports.
  - `lens/rank.py`:
    - `checkpoint = f"models/{name}" if name.endswith("_ft") else None`;
    - in `rank()`, compute `kind = name.removesuffix("_ft")` once and branch on `kind`.
  - `train.py`:
    - delete `set_seed(0)` and its import (`TrainingArguments(seed=0, data_seed=0)` already seeds);
    - delete the `else: raise ValueError` (argparse `choices` already rejects other names).
  - `test_train.py`: delete `FitTests`. `TripletTests` stays.
  - `PLAN.md`, append to the D15 outcome: `Tie is real: 51 shared hits, 9 unique each; median gold rank 43.5 vs 45.5. Final train losses 3.09/3.25 vs ln 64 ≈ 4.16 chance, so both models learned only modestly in 224 steps; the FAIL holds for this budget only.`
  - Delete the two files listed under Files.
- [x] **Step 4: Verify:**
  - Run `LENS_DEVICE=cpu .venv/bin/python -m unittest test_eval test_train -v && .venv/bin/python test_nodes.py && .venv/bin/python test_maxsim.py`. Expected: PASS, including the unchanged `test_ft_models_keep_zero_shot_caps`.
  - Run `.venv/bin/python eval.py --summarize results_test_ft.jsonl`. The R@ rows must match the verbatim summary in `BRIEF_10_REPORT.md`, with only the `maxsim_ft` D12 line.
- [x] **Step 5: Commit** `refactor: brief 10 review and audit cuts` and push.

### Task 2: Unseen-website recall check

**Files:**
- Create: `results_website_ft.jsonl`, `BRIEF_11_REPORT.md`
- Modify: `PLAN.md` (D16 outcome, a Brief 11 results line)

- [ ] **Step 1:**
  1. Download `osunlp/Multimodal-Mind2Web` `data/test_website-*` (4 shards, about 1.0 GiB) to `data/mm_test_website/`.
  2. Run `.venv/bin/python eval.py --prep-mm data/mm_test_website --steps data/steps_website.jsonl`.
- [ ] **Step 2: Leakage check.**
  - Read only the `website` and `confirmed_task` columns, with no full download:
    `pq.read_table(path, columns=["website", "confirmed_task"], filesystem=huggingface_hub.HfFileSystem())`
    for each `datasets/osunlp/Multimodal-Mind2Web/data/train-*.parquet` and `test_website-*.parquet`.
  - Print the shared-website count and the shared-task count. Expected: both 0. **If either is not 0, STOP and report.**
- [ ] **Step 3: Run** `LENS_DEVICE=cpu .venv/bin/python run_evals.py data/website_ft.log -- eval.py --steps data/steps_website.jsonl --n 200 --no-llm --rankers single,maxsim,single_ft,maxsim_ft --output results_website_ft.jsonl`.
  - Expected: gold coverage ≥ 98%, 200 rows.
  - If coverage is below 98%, STOP and report the number.
- [ ] **Step 4: Write `BRIEF_11_REPORT.md`:**
  - website/task overlap counts and gold coverage;
  - the verbatim `eval.py --summarize results_website_ft.jsonl`;
  - R@5/10/20/50/100 for all four rankers;
  - a lift table (ft − zero-shot, in hits and pts) for **both** splits, with `test_task` taken from `results_test_ft.jsonl`;
  - the D16 verdict line: `D16: single_ft − single on test_website = <h> hits → GENERALIZES | MEMORIZATION`;
  - deviations.
- [ ] **Step 5: `PLAN.md`:**
  - Add `Outcome: GENERALIZES|MEMORIZATION (single lift <h> hits; maxsim lift <h> hits)` under D16.
  - Add a `- Brief 11 (...)` line to Results.
- [ ] **Step 6:**
  - Delete `data/mm_test_website/`.
  - Commit `docs: brief 11 unseen-website check` + `chore: llm cache` and push.
  - **Stop.** The thesis re-grill is the planner's and user's job.

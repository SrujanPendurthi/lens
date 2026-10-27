# Cleanup + Fine-Tuned ColBERT vs Single-Vector (Vision Piece 3) Implementation Plan

> **Executor: Codex (`gpt-6.1-sol`).** Execute the tasks yourself in order. Each task runs TDD: failing test, verify fail, implement, verify pass, commit. Tick each `- [ ]` → `- [x]` in this file as you finish it and commit the ticks with that task's commit. Read **Codex execution notes** before starting.

**Goal:** Apply the brief 9 review and audit cuts, then test D12 piece 3, the last one. When both models are fine-tuned on the same Mind2Web train triplets, does ColBERT MaxSim beat single-vector recall@10 by ≥ 5 pts on the same 200 test steps?

**Architecture:**
- **Train data:** Multimodal-Mind2Web `train` (27 parquet shards, 7.2 GiB; cross-task with `test_task`) is prepped with the existing `prep_mm`. Each step becomes one `(sub-goal, gold text, hard negative text)` triplet. The sub-goal is the same cached Qwen query the eval uses; the hard negative is a seeded random same-page candidate with text.
- **Training:** new `train.py` fine-tunes `bge-small` (sentence-transformers, `MultipleNegativesRankingLoss`) and `answerai-colbert-small` (pylate, `losses.Contrastive`) on identical triplets with identical pre-registered hyperparameters.
- **Eval:** `rank.py` gains `single_ft`/`maxsim_ft`, loaded from `models/` with the zero-shot caps. The D12 line compares against `single_ft` when it is present.

**Tech Stack:** Python `.venv`, torch, sentence-transformers 5.3, pylate 1.6, pyarrow, mlx-lm (sub-goal misses only), `huggingface_hub`.

**Spec:** `PLAN.md` D12, D14, D15. Sources: `BRIEF_9_REPORT.md`, the 2026-10-27 caveman review, ponytail audit and ponytail review (summarized under Task 1).

## Global Constraints

- **Repo rules:**
  - Commit per task; push `origin main` after Tasks 1 and 4 (no force-push).
  - Never stage `data/` paths except `data/llm_cache.jsonl`.
  - Never touch `../lagunaLab`, `../moeStream`, or `../.gpu` contents.
  - Model jobs (Qwen sub-goals, training, eval) go through `run_evals.py` one at a time.
- **Pre-registered hyperparameters (D15), the same for both models:**
  - seed 0, 1 epoch, lr 2e-5, batch 32, warmup ratio 0.1;
  - device `cpu` (D11);
  - caps as zero-shot: single `max_seq_length=128`; ColBERT `query_length=32`, `document_length=64`.
- **No tuning:** no hyperparameter search, no dev-set early stopping, no second training run. One run each; the numbers are final.
- **Gate:** `maxsim_ft` R@10 ≥ `single_ft` R@10 + 5.0 pts on the same 200 `test_task` steps, compared in integer counts.
- **Disk:** about 25 GiB free. `data/mm_train/` (7.2 GiB) is deleted after Task 2 prep verifies.

## Codex execution notes

- **Approvals:**
  - The train shard download needs network.
  - `run_evals.py` needs execution outside the sandbox.
  - If denied, stop and report. Don't work around it.
- **`ps` is blocked:** poll logs every 60 s.
- **Long runs:**
  - Sub-goal generation is about 7.7k Qwen calls, roughly 2 h.
  - Launch with `nohup <command> > /dev/null 2>&1 &`.
  - Generation resumes from `data/llm_cache.jsonl` if stopped. Rerun the same command; it is not a new run.
- **Low on usage?** Stop at a task boundary, commit, push, and put `STATUS: stopped after Task N; next = Task N+1 Step M` at the top of `BRIEF_10_REPORT.md`. A Sonnet agent resumes from the ticks.
- **Deviations:** smallest change that keeps the tests and intent; list it under "Deviations" in the report. Never change the gate, sample, hyperparameters or triplet rule.
- **Final message:** under 20 lines: triplet count, train/test task overlap, the D12 piece 3 verdict line, R@10 counts, commits, push status, deviations.

## Review Focus

1. **Train/test leakage:** any `confirmed_task` shared between train triplets and `test_task` makes the gate meaningless. Expected 0 shared tasks. Pinned: Task 2 Step 5.
2. **Textless gold, or a page with no textful negative:** skip that step, never emit an empty positive or negative, and count the skips. Pinned: Task 2 `test_triplets_skip_and_pick`.
3. **Fairness of caps:** a fine-tuned model loaded with different length caps than its zero-shot twin confounds the result. Pinned: Task 3 `test_ft_models_keep_zero_shot_caps`.
4. **D12 baseline switch:** with `single_ft` present, margins compare against `single_ft`, not zero-shot `single`. Pinned: Task 3 `test_d12_uses_single_ft_when_present`.
5. **Haiku cache keys survive the test reshuffle.** Pinned: the moved `test_haiku_pick_key_unchanged` must still pass in Task 1.

---

### Task 1: Review and audit cuts

**Files:**
- Modify: `eval.py`, `test_eval.py`, `PLAN.md`
- Delete: `docs/superpowers/plans/2026-10-27-colpali-fix-and-audit.md` (done)
- Commit separately: `data/llm_cache.jsonl`

**Interfaces:**
- Produces:
  - `prep_mm(parquet_dir: str, steps_path: str) -> None` (no screenshots, no `uid`);
  - `summarize(rows, llm)` (no `query` param);
  - `has_gold(step) -> bool`.
- `ALLOWED_RANKERS` **stays**: Task 3 extends it.

- [ ] **Step 1: Cuts in `eval.py`:**
  - Delete screenshot saving in `prep_mm`: the `shots_dir` param, mkdir, bytes write, the `missing screenshot` print, and the `uid` row field.
  - Delete the `query_mode` row field, the `query` param of `summarize`, and the `query=` print.
  - `--ks`: `type=lambda s: sorted({int(k) for k in s.split(",")}), default=[10]`. Delete the try/except, the `min(ks) < 1` guard and the `a.n < 1` guard.
  - Replace the `survival = {id(s): ...}` dict with direct `has_gold(s)` calls.
  - Delete the `raise ValueError` empty check inside `summarize`; the `--summarize` `ap.error` stays.
- [ ] **Step 2: Cuts in `test_eval.py`:**
  - Delete `CleanupTests` and the png assertion in `test_prep_parses_json_string_fields`.
  - Delete `test_summarize_preserves_saved_query_mode`.
  - Drop the `mode` param of `run_eval` and its `"subgoal"` args.
  - Delete `assertNotIn("GATE:", ...)`.
  - Move `test_haiku_pick_key_unchanged` into `CacheTests`, with `import hashlib` at the top of the file.
  - Collapse blank-line runs to 2 between classes and 1 between methods.
- [ ] **Step 3: `PLAN.md`:**
  - Rewrite the D14 headline to `**D12 piece 2 FAILS** (mean-pooled ColPali 39/200 vs single 53/200; the first run was size-biased).`
  - Add to D14: `Alignment 4/5; even if all 39 hits came from the aligned 80% and the rest scored equally, the ceiling is ~49/200 < 63, so FAIL stands.`
  - Collapse the blank lines at the top.
- [ ] **Step 4: Verify:**
  - Run `LENS_DEVICE=cpu .venv/bin/python -m unittest test_eval -v && .venv/bin/python test_nodes.py && .venv/bin/python test_maxsim.py`. Expected: PASS.
  - Run `.venv/bin/python eval.py --summarize results_test_200.jsonl`. The R@5/10/20 rows for bm25/single/maxsim/cross must match `BRIEF_7_REPORT.md:24-29`.
  - Run `wc -l eval.py test_eval.py` and record it; expected about -90 lines combined.
- [ ] **Step 5: Commit**
  - `refactor: brief 9 review and audit cuts` (code, tests, PLAN.md, the deleted plan doc).
  - Then `chore: llm cache` (`data/llm_cache.jsonl` only).
  - Push.

### Task 2: Train-split prep + triplets

**Files:**
- Create: `train.py`
- Modify: `run_evals.py` (it runs any script: `run_evals.py LOG -- <script> <args>`), `.gitignore` (add `models/`)
- Test: `test_train.py`

**Interfaces:**
- Consumes: `prep_mm`, `has_gold` (Task 1); `lens.pick.subgoal(task, prev) -> dict` (`"text"`).
- Produces:
  - `triplets(steps: list[dict], seed: int = 0) -> tuple[list[dict], dict]`. It returns rows `{"query", "positive", "negative"}` and skip counts `{"textless_gold": int, "no_negative": int, "no_gold": int}`.
  - The file `data/triplets_train.jsonl`.

- [ ] **Step 1: Write the failing test** `test_train.py`:
  ```python
  def test_triplets_skip_and_pick(self):
      steps = [
          {"task": "t", "prev": [], "gold": ["1"], "nodes": [{"ref": "1", "tag": "a", "text": "Go"},
           {"ref": "2", "tag": "a", "text": "Stop"}, {"ref": "3", "tag": "a", "text": ""}]},
          {"task": "t", "prev": [], "gold": ["1"], "nodes": [{"ref": "1", "tag": "a", "text": ""},
           {"ref": "2", "tag": "a", "text": "Stop"}]},
          {"task": "t", "prev": [], "gold": ["1"], "nodes": [{"ref": "1", "tag": "a", "text": "Go"}]},
          {"task": "t", "prev": [], "gold": ["9"], "nodes": [{"ref": "1", "tag": "a", "text": "Go"}]}]
      with patch("lens.pick.subgoal", return_value={"text": "click go"}):
          rows, skips = train.triplets(steps)
      self.assertEqual(rows, [{"query": "click go", "positive": "Go", "negative": "Stop"}])
      self.assertEqual(skips, {"textless_gold": 1, "no_negative": 1, "no_gold": 1})
  ```
- [ ] **Step 2: Run** `.venv/bin/python -m unittest test_train -v`. Expected: FAIL (`ModuleNotFoundError: train`).
- [ ] **Step 3: Implement `triplets`.**
  - Positive = text of the first gold node with text, in node order.
  - Negative = `random.Random(f"{seed}\n{task}\n{len(prev)}").choice` over non-gold nodes with text.
  - Check order: no surviving gold, then textless gold, then no negative. Call the sub-goal as `lens.pick.subgoal(...)` (module attribute; the test patches it).
  - Add a `train.py --triplets STEPS OUT` CLI that writes the rows and prints the skip counts.
  - `run_evals.py`: run `["../.gpu/gpu-run", "lens", ".venv/bin/python", *args]`; usage becomes `run_evals.py LOG -- <script.py> <args...>`.
- [ ] **Step 4: Run the test.** Expected: PASS.
- [ ] **Step 5: Prep and generate (long):**
  1. Download the 27 `train-*.parquet` files of `osunlp/Multimodal-Mind2Web` to `data/mm_train/` (`huggingface_hub.snapshot_download(..., allow_patterns="train-*")`).
  2. Run `.venv/bin/python eval.py --prep-mm data/mm_train --steps data/steps_train.jsonl`.
  3. Print the train/test overlap: `len({s["task"] for s in train} & {s["task"] for s in test})`. **Expected: 0. If it is not 0, STOP and report.**
  4. Run `run_evals.py data/triplets.log -- train.py --triplets data/steps_train.jsonl data/triplets_train.jsonl`. Expected: about 7k rows, with skip counts printed.
  5. Delete `data/mm_train/`.
- [ ] **Step 6: Commit** `feat: train-split triplets for piece 3` (train.py, test_train.py, run_evals.py, .gitignore) plus `chore: llm cache`.

### Task 3: Fine-tune + rankers

**Files:**
- Modify: `train.py`, `lens/rank.py`, `eval.py` (`summarize` D12 base), `test_train.py`, `test_eval.py`

**Interfaces:**
- Consumes: `data/triplets_train.jsonl` (Task 2).
- Produces:
  - `train.py --fit {single,maxsim}` saves to `models/{single,maxsim}_ft`;
  - `ALLOWED_RANKERS = RANKERS + ("single_ft", "maxsim_ft")`;
  - `rank(name, query, texts)` accepts the `_ft` names.

- [ ] **Step 1: Write the failing tests:**
  - `test_ft_models_keep_zero_shot_caps` (`test_eval.py`):
    - Patch `sentence_transformers.SentenceTransformer` and `pylate.models.ColBERT` with mocks, then clear the `rank._model` cache.
    - `_model("single_ft")` → SentenceTransformer called with `"models/single_ft"`, and the result's `max_seq_length == 128`.
    - `_model("maxsim_ft")` → ColBERT called with `"models/maxsim_ft"`, `query_length=32`, `document_length=64`.
  - `test_d12_uses_single_ft_when_present` (`MarginTests`): rows with `single` 53/200, `single_ft` 60/200 and `maxsim_ft` 65/200 → output contains `D12 maxsim_ft: R@10 = 65/200, single_ft = 60/200, margin = +2.5 pts → FAIL`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement:**
  - In `rank.py`, map each name to a checkpoint (`single_ft` → `models/single_ft`, `maxsim_ft` → `models/maxsim_ft`) and branch on `name.removesuffix("_ft")`. Caps are identical.
  - In `summarize`, `base = "single_ft" if "single_ft" in rankers else "single"`. Print a D12 line for every ranker except `base`, formatted with the base's name.
  - `train.py --fit`:
    - Load `data/triplets_train.jsonl` into `datasets.Dataset` with columns `query`, `positive`, `negative`.
    - Use the hyperparameters in Global Constraints through `SentenceTransformerTrainingArguments` (`seed=0`, `num_train_epochs=1`, `learning_rate=2e-5`, `per_device_train_batch_size=32`, `warmup_ratio=0.1`, `use_cpu=True`, `save_strategy="no"`).
    - single: `MultipleNegativesRankingLoss`.
    - maxsim: `pylate.losses.Contrastive` + `pylate.utils.ColBERTCollator(model.tokenize)`.
    - Save to `models/<name>_ft`.
- [ ] **Step 4: Run all tests.** Expected: PASS.
- [ ] **Step 5: Train both:**
  - `run_evals.py data/fit_single.log -- train.py --fit single`, then `run_evals.py data/fit_maxsim.log -- train.py --fit maxsim`.
  - Each must exit 0 and leave a loadable model dir. Record the final train loss for each.
- [ ] **Step 6: Commit** `feat: fine-tuned single/maxsim rankers (piece 3)`.

### Task 4: Run, gate, record

**Files:**
- Create: `results_test_ft.jsonl`, `BRIEF_10_REPORT.md`
- Modify: `PLAN.md` (D15 outcome and a Brief 10 results line)

- [ ] **Step 1: Run** `LENS_DEVICE=cpu .venv/bin/python run_evals.py data/ft_test.log -- eval.py --n 200 --no-llm --rankers single,maxsim,single_ft,maxsim_ft --output results_test_ft.jsonl`.
  - Expected: 200 rows, `subgoal cache misses: 0`, and zero-shot single/maxsim R@10 = 53/200 and 36/200 (matching `results_test_200.jsonl`).
  - If any of these is off, STOP.
- [ ] **Step 2: Write `BRIEF_10_REPORT.md`:**
  - triplet count and skip counts; train/test overlap;
  - final train losses;
  - the verbatim `eval.py --summarize results_test_ft.jsonl`;
  - R@5/10/20/50/100 for all four rankers;
  - the fine-tuning lift (ft − zero-shot R@10) for each model;
  - `maxsim_ft` R@10 by op;
  - deviations.
- [ ] **Step 3: `PLAN.md`:**
  - Add `Outcome: PASS|FAIL, maxsim_ft R@10 = a/200 vs single_ft b/200 (margin m pts)` under D15.
  - Add a `- Brief 10 (...)` line to Results.
- [ ] **Step 4: Commit** `docs: brief 10 results (piece 3 gate)` + `chore: llm cache`, then push. **Stop there**: whether it's PASS (infra) or FAIL (re-grill the thesis), the next decision belongs to the planner.

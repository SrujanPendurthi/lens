# Retrieve+Pick Product Test (D17) Implementation Plan

> **Executor: Codex (`gpt-6.1-sol`).** Execute the tasks yourself in order. Each code task runs TDD: failing test, verify fail, implement, verify pass, commit. Tick each `- [ ]` → `- [x]` in this file as you finish it and commit the ticks with that task's commit. Read **Codex execution notes** before starting.

**Goal:** Answer the question the project exists for (D7, revived as D17). Does Haiku picking from the top-k of zero-shot single-vector retrieval keep accuracy within 2 pts of picking from the full element list, at ≤ 20% of its prompt tokens?

**Architecture:**
- The multi-vector thesis is closed: D12 pieces 1–3 failed and the re-grill chose D17. So the MaxSim path, `pylate`, and the D12 margin code are cut.
- `summarize` gains one D17 verdict line per picker arm.
- The run reuses the existing picker path (`pick_nodes`, cached Haiku via `claude -p`) on the same 200 `test_task` steps.

**Tech Stack:** Python `.venv`, existing `eval.py` / `lens/`; Haiku 5.5 via headless `claude -p` (Pro plan, no API key).

**Spec:** `PLAN.md` D7, D10, D11, D16, D17. Sources: `BRIEF_11_REPORT.md`, the 2026-10-27 caveman review and ponytail audit.

## Global Constraints

- **Repo rules:**
  - Commit per task; push `origin main` after each task (no force-push).
  - Never stage `data/` except `data/llm_cache.jsonl`, which gets its own `chore: llm cache` commit.
  - Never touch `../lagunaLab`, `../moeStream`, `../.gpu` contents.
- **Sample:** the same 200 `test_task` steps as `results_test_200.jsonl` (seed-0 shuffle + gold-survival filter). Sub-goal cache misses must be 0.
- **Arms (pre-registered, only these):**
  - `baseline`: all candidate nodes, in the existing neutral shuffle;
  - `single@10` and `single@50`: zero-shot `single` (bge-small), `LENS_DEVICE=cpu`.
- **Gate (D17), per arm, in integer counts.** With `a` = arm correct, `b` = baseline correct, and token medians `ta` and `tb`:
  - **PASS** if `(a - b) * 100 >= -2 * n` and `ta * 5 <= tb`;
  - **KILL** if `(a - b) * 100 < -5 * n`;
  - else **NEITHER**.
  - D17 PASS if either arm passes.
- **Tokens:**
  - The existing accounting stays: arm tokens = Haiku prompt tokens + sub-goal tokens; baseline = Haiku prompt tokens.
  - Report raw numbers and numbers minus the ~203-token CLI overhead (D11). The gate uses raw.
- **Do not touch:** `models/single_ft`, `data/triplets_train.jsonl`, `triplets()`.

## Codex execution notes

- **Approvals:**
  - `claude -p` and `run_evals.py` run outside the sandbox; request approval.
  - If denied, stop and report. Don't work around it.
- **Haiku usage:**
  - About 600 calls, about 3M input tokens (baseline prompts are ~15k tokens each).
  - On a Pro usage-limit or rate error, `pick()` raises and nothing bad is cached. Wait for the reset, then rerun the **same** command; cached picks replay instantly.
  - Never change arms, k or sample to get under a limit.
- **`ps` is blocked:** poll logs every 60 s.
- **Long runs:** `nohup <command> > /dev/null 2>&1 &`.
- **Low on usage?** Stop at a task boundary, commit, push, and put `STATUS: stopped after Task N; next = Task N+1 Step M` at the top of `BRIEF_12_REPORT.md`. A Sonnet agent resumes from the ticks.
- **Deviations:** smallest change keeping tests and intent; list them in the report. Never change the gate, arms or sample.
- **Final message:** under 15 lines: the per-arm D17 lines, the overall verdict, Haiku calls made, commits, push status, deviations.

## Review Focus

1. **Gate boundaries:** exactly −2.0 pts with exactly 20% tokens must PASS, −5.5 pts must KILL, and ≤ −2 pts with 26% tokens must be NEITHER. Pinned: Task 2 `test_d17_verdicts`.
2. **Gold never shown** (gold outside the top-k): a pick there can't be correct, and the line must still count it in `n`. Covered by the existing `ok` definition; the Task 2 fixture includes arm misses.
3. **Usage-limit interruption mid-run:** a rerun must replay cached picks and give byte-identical rows for the finished steps. Pinned: Task 3 Step 2 (compare the first-N rows of the interrupted and final files when an interruption happens; record "no interruption" otherwise).
4. **Old result files after the MaxSim cut:** `--summarize results_test_200.jsonl` must still work and print the bm25/single/cross rows unchanged. Pinned: Task 1 Step 4.
5. **Haiku cache keys unchanged:** pinned by the existing `test_haiku_pick_key_unchanged`.

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
  - `claude_guidance_file.md` §3 becomes one line: `Status and decisions: PLAN.md (latest D17).`
  - Append to the D16 outcome in `PLAN.md`: `Paired: single 14 gained vs 12 lost (p≈0.85), so the single fine-tune lift is indistinguishable from 0 on unseen sites; maxsim 23 vs 7 (p≈0.005) only catches up to zero-shot single (55 vs 52). test_website = 10 sites, so the effective sample is ~10, not 200.`
  - Delete the finished plan doc.
- [ ] **Step 4: Verify:**
  - Run `LENS_DEVICE=cpu .venv/bin/python -m unittest test_eval test_train -v && .venv/bin/python test_nodes.py`. Expected: PASS.
  - `grep -rn "maxsim\|pylate" lens eval.py train.py requirements.txt` prints nothing.
  - `.venv/bin/python eval.py --summarize results_test_200.jsonl` prints bm25/single/cross R@5/10/20 rows identical to `BRIEF_7_REPORT.md`.
- [ ] **Step 5: Commit** `refactor: close multi-vector path (D16/D17)` and push.

### Task 2: D17 verdict line

**Files:**
- Modify: `eval.py` (`summarize`), `test_eval.py`

**Interfaces:**
- Produces: when `llm` is true and every row has `baseline`, `summarize` prints, for each arm containing `@` (in row order):
  `D17 <arm>: acc <a>/<n> vs baseline <b>/<n> (Δ <±x.x> pts; arm-only <c>, baseline-only <d>), tokens <ta> vs <tb> (<p.p>%) → PASS|KILL|NEITHER`
  - `c` = steps the arm gets right and baseline gets wrong; `d` = the reverse.
  - `ta` and `tb` are medians formatted `:.0f`.
  - `p` = `100 * ta / tb`.

- [ ] **Step 1: Write the failing test** in `test_eval.py`:
  ```python
  class D17Tests(unittest.TestCase):
      def test_d17_verdicts(self):
          pick = lambda ok, tok: {"ok": ok, "tokens": tok, "ttft": 1, "gold_shown": True, "hallucinated": False}
          rows = [{"n_nodes": 400, "textless_gold": False, "op": "CLICK", "single": {"gold_rank": 0, "ms": 1},
                   "baseline": pick(i < 72, 15000), "single@50": pick(i < 68, 3000),
                   "single@10": pick(i < 61, 1000), "single@20": pick(i < 72, 3900)} for i in range(200)]
          out = io.StringIO()
          with contextlib.redirect_stdout(out):
              summarize(rows, True)
          text = out.getvalue()
          self.assertIn("D17 single@50: acc 68/200 vs baseline 72/200 (Δ -2.0 pts; arm-only 0, baseline-only 4), "
                        "tokens 3000 vs 15000 (20.0%) → PASS", text)
          self.assertIn("D17 single@10: acc 61/200 vs baseline 72/200 (Δ -5.5 pts; arm-only 0, baseline-only 11), "
                        "tokens 1000 vs 15000 (6.7%) → KILL", text)
          self.assertIn("D17 single@20: acc 72/200 vs baseline 72/200 (Δ +0.0 pts; arm-only 0, baseline-only 0), "
                        "tokens 3900 vs 15000 (26.0%) → NEITHER", text)
  ```
- [ ] **Step 2: Run** `LENS_DEVICE=cpu .venv/bin/python -m unittest test_eval.D17Tests -v`. Expected: FAIL (no D17 line).
- [ ] **Step 3: Implement** in `summarize` using the counts rule from Global Constraints. Print nothing when `llm` is false or any row lacks `baseline`.
- [ ] **Step 4: Run all tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: D17 retrieve+pick verdict line` and push.

### Task 3: Run, gate, record

**Files:**
- Create: `results_d17.jsonl`, `BRIEF_12_REPORT.md`
- Modify: `PLAN.md` (D17 outcome, a Brief 12 results line)

- [ ] **Step 1: Smoke.** Run `LENS_DEVICE=cpu .venv/bin/python run_evals.py data/d17_smoke.log -- eval.py --n 3 --rankers single --ks 10,50 --output /private/tmp/d17_smoke.jsonl`. Expected: 3 rows, each with `baseline`, `single@10` and `single@50` picks, and a D17 line per arm.
- [ ] **Step 2: Full run.** Run `LENS_DEVICE=cpu .venv/bin/python run_evals.py data/d17.log -- eval.py --n 200 --rankers single --ks 10,50 --output results_d17.jsonl`.
  - Expected: 200 rows, `subgoal cache misses: 0`, and single R@10 = 53/200 (same as `results_test_200.jsonl`).
  - If interrupted by a usage limit, rerun the identical command after the reset and record each interruption (time, step count). Copy the partial file first, then confirm its rows are identical to the same rows in the final file.
- [ ] **Step 3: Write `BRIEF_12_REPORT.md`:**
  - the verbatim `eval.py --summarize results_d17.jsonl` (pick table and D17 lines);
  - the overall verdict: `D17: PASS|KILL|NEITHER` (PASS if either arm passes; KILL if both kill);
  - accuracy given gold shown, for each arm;
  - tokens raw and minus 203;
  - accuracy by op;
  - Haiku calls made vs cache hits;
  - interruptions;
  - deviations.
- [ ] **Step 4: `PLAN.md`:**
  - Under D17, add `Outcome: <verdict>; single@10 <a>/200, single@50 <a>/200 vs baseline <b>/200; token ratio <p>% / <p>%`.
  - Add a `- Brief 12 (...)` line to Results.
- [ ] **Step 5: Commit** `docs: brief 12 retrieve+pick product test (D17)` + `chore: llm cache` and push. **Stop.** The next direction belongs to the planner and user.

# ColPali Size-Bias Fix + Audit Cuts Implementation Plan

> **Executor: Codex (`gpt-6.1-sol`).** Execute the tasks yourself in order. Each task runs TDD: failing test, verify fail, implement, verify pass, commit. Tick each `- [x]` → `- [x]` in this file as you finish it and commit the ticks with that task's commit. Read **Codex execution notes** before starting.

**Goal:** Decide D12 piece 2 for real. Brief 8's ColPali FAIL (R@10 1.5%) is inconclusive: the scorer favors large boxes. Apply the repo-audit cuts, check that the box↔screenshot alignment is right, fix the scoring, rerun ColPali on cached embeddings, and record PASS or FAIL.

**Architecture:**
- **Scoring change:** box score changes from `Σ_q max_{p∈box} sim(q,p)` to `Σ_q mean_{p∈box} sim(q,p)`. A container's max always equals or beats its child's, since the container covers every patch the child does. A mean doesn't have that property.
- **Cuts:** the D7 verdict is replaced by a D12 margin line, and dead picker, truncation and debug paths are cut.
- **ColPali tile embeddings** are already cached (`data/colpali/*.pt`), so the rerun costs query encoding only.

**Tech Stack:** same as brief 8: Python `.venv`, torch, transformers `ColPaliForRetrieval` (`vidore/colpali-v1.3-hf`), pillow.

**Spec:** `PLAN.md` D12, D13, D14. Sources: `BRIEF_8_REPORT.md`, the 2026-10-27 caveman review, the 2026-10-27 ponytail audit (summarized per task).

## Global Constraints

- **Repo rules:**
  - Commit per task; never stage `data/`; push to `origin main` after Task 4 (no force-push).
  - Never touch `../lagunaLab`, `../moeStream`, or `../.gpu` contents.
  - GPU/model jobs go through `run_evals.py` (`../.gpu/gpu-run`) one at a time.
- **Sample:** the same 200 `test_task` steps as `results_test_200.jsonl`. Text rankers use `LENS_DEVICE=cpu`; `LENS_VISION_DEVICE` defaults to `mps`.
- **Gate (D12):** colpali R@10 ≥ single R@10 + 5.0 pts. Single = 26.5% (53/200), so ≥ 63/200 hits. Compare in integer counts, not floats.
- **Pre-registered:** one pooling variant only (mean). If it fails, piece 2 is FAIL; don't try other poolings.
- **Signal check:** the size-only floor is the smallest-box-first ranker at R@10 6.0% (12/200; computed 2026-10-27). ColPali below 12/200 = "no usable signal" in the report.
- **Model caches:**
  - Existing `data/colpali/*.pt` files were built with `MODEL=vidore/colpali-v1.3-hf`, `TILE=1280`, and must stay cache hits.
  - Never delete `data/colpali/`.

## Codex execution notes

- **Approvals:**
  - Network is needed only if the HF cache is missing.
  - `run_evals.py` needs execution outside the sandbox; request approval.
  - If denied, stop and report. Don't work around it.
- **`ps` is blocked:** to wait on a run, poll the run log for the `subgoal cache misses:` line every 60 s.
- **Long runs:** `nohup <command> > /dev/null 2>&1 &`, then poll.
- **Low on usage?** Stop at a task boundary, commit, push, and put `STATUS: stopped after Task N; next = Task N+1 Step M` at the top of `BRIEF_9_REPORT.md`. A Sonnet agent resumes from the ticks.
- **Deviations:** make the smallest change that satisfies the tests and intent, and list it under "Deviations" in `BRIEF_9_REPORT.md`. Never change the gate, sample, pooling choice or fixed values.
- **Final message:** under 20 lines: the alignment result, the D12 piece 2 verdict line, R@10 counts, commits, push status, deviations.

## Review Focus

1. **Nested boxes** (a page-wide div vs the button inside it, same matching patch): the child must outrank the container. Pinned: Task 3 `test_nested_child_beats_container`.
2. **Legacy cache files with no metadata** must load as hits, or the rerun re-encodes 859 tiles. Pinned: Task 3 `test_legacy_cache_hits`.
3. **A cache built with a different model or tile size** must re-encode, never silently reuse. Pinned: Task 3 `test_stale_cache_reencodes`.
4. **The gate boundary at exactly +5.0 pts** must PASS, and +4.5 must FAIL, with no float drift. Pinned: Task 1 `test_d12_margin_boundary`.
5. **Haiku pick cache keys must stay identical after the `picker` param is cut**, or 53 cached Haiku picks miss. Pinned: Task 1 `test_haiku_pick_key_unchanged`.

---

### Task 1: Audit cuts

**Files:**
- Modify: `eval.py`, `lens/pick.py`, `test_eval.py`, `.gitignore`, `PLAN.md`
- Delete: `docs/superpowers/plans/2026-10-26-cleanup-and-colpali.md` (done; history keeps it)

**Interfaces:**
- Produces:
  - `pick(text: str) -> dict`;
  - `pick_key(text: str) -> str`;
  - `summarize(rows, llm, query="subgoal")`, which prints `D12 <name>: R@10 = <a>/<n>, single = <b>/<n>, margin = <m:+.1f> pts → PASS|FAIL` for every ranker present except `single`, and no longer prints `GATE:`.

- [x] **Step 1: Write the failing tests** in `test_eval.py`.
  - Replace `GateTests` and `test_gate_incomplete_without_maxsim10` with:
    ```python
    def d12(self, hits, single_hits, n=200):
        rows = [{"n_nodes": 100, "textless_gold": False,
                 "single": {"gold_rank": 0 if i < single_hits else 99, "ms": 1},
                 "colpali": {"gold_rank": 0 if i < hits else 99, "ms": 1}} for i in range(n)]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            summarize(rows, False)
        return out.getvalue()

    def test_d12_margin_boundary(self):
        self.assertIn("D12 colpali: R@10 = 63/200, single = 53/200, margin = +5.0 pts → PASS", self.d12(63, 53))
        self.assertIn("margin = +4.5 pts → FAIL", self.d12(62, 53))
        self.assertNotIn("GATE:", self.d12(63, 53))

    def test_haiku_pick_key_unchanged(self):
        self.assertEqual(pick_key("x"), hashlib.sha1(b"pick-v2\nclaude-haiku-5-5\nx").hexdigest())
    ```
  - Delete `test_qwen_picker_rejected` and `test_qwen_pick_key_unchanged`.
  - Drop `picker`, `truncated` and `example` from the remaining fixtures and assertions.
- [x] **Step 2: Run** `LENS_DEVICE=cpu .venv/bin/python -m unittest test_eval -v`. Expected: the two new tests FAIL.
- [x] **Step 3: Apply the cuts:**
  - `eval.py`:
    - Delete the D7 verdict block (`:204-217`) and add the D12 line. The PASS test is `(hits - single_hits) * 100 >= 5 * n`.
    - Delete `--picker` and the `"picker"` row field.
    - Delete the `example` field and `EXAMPLE:` print (`:130-134`).
    - Delete the `baseline truncated` line and the haiku/ttft branch; the timing column header becomes `API s`.
    - Delete the screenshot `path` fallback in `prep_mm` (`:55,58-59`); the `missing screenshot` print stays.
    - Compute gold survival once with `has_gold(s)`, used by both the coverage check and the filter.
  - `lens/pick.py`:
    - Remove the `picker` params; `pick_key` hashes `f"pick-v2\n{HAIKU}\n{text}"`.
    - Drop `"truncated": False`.
  - `.gitignore`: replace the `data/train_*.json` and `data/steps*.jsonl` lines with `data/*` and `!data/llm_cache.jsonl`.
  - `PLAN.md`: delete the stale original-plan sections (`## Files (fewest possible)`, `## Steps`, `## Verification`, `## Risks to watch`) and both `## Handover status` blocks.
  - Delete the finished plan doc.
- [x] **Step 4: Verify:**
  - Run `LENS_DEVICE=cpu .venv/bin/python -m unittest test_eval test_vision -v && .venv/bin/python test_nodes.py && .venv/bin/python test_maxsim.py`. Expected: PASS.
  - Run `.venv/bin/python eval.py --summarize results_test_200.jsonl`. Its R@5/R@10/R@20 table must match `BRIEF_7_REPORT.md:24-29` exactly; the `pagemaxsim` row is absent (no longer allowed) and there is no `GATE:` line.
  - `git status --short` must not list `data/` paths other than `data/llm_cache.jsonl`.
- [x] **Step 5: Commit** `refactor: audit cuts; D12 margin replaces D7 verdict`.

### Task 2: Box alignment check (no code committed)

**Interfaces:** none. Its output is a pass/STOP decision that goes in the report.

- [x] **Step 1:** Take the first 5 of the 200 sampled steps that have a positive-area gold box.
  - For each, crop the screenshot to the gold box plus a 40 px margin and draw the box in red.
  - Save to `data/box_check/<uid>.png`, using a throwaway script in `/private/tmp`, not committed.
- [x] **Step 2:** Open each crop. A crop passes if the red box frames the element the gold node text names (`[ref] tag "text"` from the step's nodes). Record pass/fail and the node line for all 5 in `BRIEF_9_REPORT.md`.
- [x] **Step 3:** If fewer than 4/5 pass, **STOP**. Commit the report with `STATUS: alignment failed` and report: the boxes are off, so no ColPali rerun is valid. Otherwise continue.

### Task 3: Mean-pooled box score + cache key

**Files:**
- Modify: `lens/vision.py`, `eval.py:148-155`, `test_vision.py`, `test_eval.py`

**Interfaces:**
- Consumes: `tile_tops`, `box_patches`, `encode_tiles`, `encode_query` (unchanged signatures).
- Produces:
  - `rank_colpali(query, uid, refs, boxes) -> Tensor`, same signature, new score = `Σ_q mean over the box's patches across all tiles`;
  - the cache payload gains `"model": MODEL, "tile": TILE`.

- [x] **Step 1: Write the failing tests** in `test_vision.py`. Use the same patching style as the existing `test_rank_clips_bottom_and_takes_max_across_tiles`:
  - `test_nested_child_beats_container`:
    - one tile of zeros except patch 0 = `[1., 0.]`, with `encode_query` returning `torch.eye(2)[:1]`;
    - an image of 1280×1280;
    - boxes `{'child': [0, 0, 40, 40], 'container': [0, 0, 1280, 1280]}`.
    - Assert `scores[0] > scores[1]` and `scores[0].item() == 1.0`.
  - Rename `test_rank_clips_bottom_and_takes_max_across_tiles` to `..._and_pools_across_tiles`. Expected becomes `[1., -inf, 1., -inf, -inf]`: span = mean of (1,0) and (0,1) per query token, so 0.5 + 0.5.
  - `test_legacy_cache_hits`: write `{"tiles": [t], "stats": {...}}` with no model/tile keys to `CACHE_DIR/u.pt`; with `load_tiles` patched to raise, `encode_tiles('u')` returns `[t]`.
  - `test_stale_cache_reencodes`: the payload has `"model": "other"`; `encode_tiles` calls `load_tiles` (patch `load_tiles` and `_load` with mocks; assert `load_tiles` was called).
  - In `test_eval.py`'s `test_colpali_receives_uid_refs_boxes_and_saves_stats`: with mocked stats `encode_s=2.0`, assert `row["colpali"]["ms"] < 2000`. `ms` excludes tile encoding.
- [x] **Step 2: Run** `.venv/bin/python -m unittest test_vision test_eval -v`. Expected: the new tests FAIL; `test_nested_child_beats_container` fails because the scores tie at 1.0.
- [x] **Step 3: Implement:**
  - In `rank_colpali`, gather each box's patch similarity columns from every tile it overlaps, concatenate them (`[Lq, P]`), and score `.mean(dim=1).sum()`.
  - In `encode_tiles`:
    - save `model`/`tile` in the payload;
    - on load, re-encode if `payload.get("model", MODEL) != MODEL or payload.get("tile", TILE) != TILE`. A missing key means a legacy file built with the current constants.
  - `DEVICE` default: `"mps" if torch.backends.mps.is_available() else "cpu"`, as in `rank.py:7`.
  - In `eval.py`, colpali `ms` = elapsed ms − `encode_s * 1000`.
- [x] **Step 4: Verify:**
  - Run plain: `.venv/bin/python -m unittest test_vision test_eval -v`.
  - Then run heavy via `../.gpu/gpu-run lens env LENS_VISION_TESTS=1 .venv/bin/python -m unittest test_vision -v`. Expected: all PASS, including `test_text_region_wins` (both boxes are the same size, so mean pooling keeps Search > Privacy). If it fails, STOP and report. Don't swap in synthetic vectors.
- [x] **Step 5: Commit** `fix: mean-pool ColPali box scores (size bias); versioned tile cache`.

### Task 4: Rerun, gate, record

**Files:**
- Create: `results_test_colpali_mean.jsonl`, `BRIEF_9_REPORT.md`
- Modify: `PLAN.md` (D14 outcome and a Brief 9 results line)

- [ ] **Step 1: Run** `LENS_DEVICE=cpu .venv/bin/python run_evals.py data/colpali_mean.log -- --n 200 --no-llm --rankers single,colpali --output results_test_colpali_mean.jsonl`.
  - Expected: 200 rows, `subgoal cache misses: 0`, every colpali row has `cache_hit: true`, and single R@10 = 53/200.
  - If any of these is off, STOP and report.
- [ ] **Step 2:** Run `.venv/bin/python eval.py --summarize results_test_colpali_mean.jsonl` and paste the output verbatim into the report.
- [ ] **Step 3: Write `BRIEF_9_REPORT.md`:**
  - Task 2 alignment table.
  - The verbatim summary.
  - R@5/10/20/50/100 for single, colpali-max (from `results_test_colpali.jsonl`) and colpali-mean.
  - Colpali-mean R@10 vs the 12/200 size-only floor.
  - Colpali-mean R@10 split by op (CLICK/TYPE/SELECT) and by `textless_gold`.
  - Median gold-rank percentile (`gold_rank / n_nodes`) for colpali-max (0.324) vs colpali-mean.
  - Deviations.
- [ ] **Step 4: `PLAN.md`:**
  - Under D14, add `Outcome: PASS|FAIL, colpali-mean R@10 = a/200 vs single 53/200 (margin m pts)`.
  - Add `- Brief 9 (...)` to Results.
- [ ] **Step 5: Commit** `docs: brief 9 results (ColPali mean-pooled gate)` and push `origin main`.

### Task 5: Remove ColPali, only if Task 4 = FAIL

Skip this task (tick it with "skipped: PASS") if D14 = PASS.

**Files:**
- Delete: `lens/vision.py`, `test_vision.py`
- Modify: `eval.py` (remove the `rank_colpali`/`encoding_stats` import and dispatch, plus `boxes` in `prep_mm`), `lens/rank.py:9` (`ALLOWED_RANKERS = RANKERS`), `test_eval.py` (colpali and box tests), `requirements.txt` (drop `pillow`)

- [ ] **Step 1:** Delete as listed. Keep `results_test_colpali*.jsonl` and `data/colpali/` as evidence.
- [ ] **Step 2: Verify:**
  - `LENS_DEVICE=cpu .venv/bin/python -m unittest test_eval -v && .venv/bin/python test_nodes.py && .venv/bin/python test_maxsim.py` passes.
  - `grep -rn "colpali\|vision\|boxes" lens eval.py` prints nothing.
  - `eval.py --summarize results_test_200.jsonl` still matches `BRIEF_7_REPORT.md:24-29`.
- [ ] **Step 3: Commit** `refactor: remove ColPali ranker (D14 FAIL)` and push.

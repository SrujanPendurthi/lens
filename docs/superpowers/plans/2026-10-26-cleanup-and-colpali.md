# Cleanup + ColPali Screenshot Retrieval (Vision Piece 2) Implementation Plan

> **Executor: Codex (`gpt-6.1-sol`).** Execute the tasks yourself in order. Each task runs TDD: failing test, verify fail, implement, verify pass, commit. Tick each `- [ ]` → `- [x]` in this file as you finish it and commit the ticks with that task's commit. Read **Codex execution notes** below before starting.

**Goal:** Apply the brief 7 review and audit cuts, then test D12 piece 2: does ColPali MaxSim over screenshot patches, mapped to elements by bounding box, beat single-vector recall@10 by ≥ 5 pts?

**Architecture:**
- **Cleanup:** removes settled or failed paths (pagemaxsim, context mode, Qwen picker, train prep, `--k`, stale files) and adds element boxes to the test-split prep.
- **Vision ranker (new `lens/vision.py`):** tiles each 1280-px-wide full-page screenshot into 1280×1280 squares (the bottom tile is padded white), encodes each tile with native-transformers ColPali (32×32 = 1024 patches, 40 px each) and caches the tile embeddings per screenshot. An element's score is MaxSim of the sub-goal query tokens over the patches its box overlaps, across all tiles.

**Tech Stack:** Python 3.13 `.venv`, torch, `transformers` 5.3 `ColPaliForRetrieval` / `ColPaliProcessor` with `vidore/colpali-v1.3-hf` (no colpali-engine), `pillow` (new), and the existing rankers on `LENS_DEVICE=cpu`.

**Spec:** `PLAN.md` D12 + D13. Review and audit sources: the brief 7 review and the repo audit (summarized under each task).

## Global Constraints

- Same repo, commit, push and sandbox rules as the brief 7 plan's Global Constraints. Commit per task, never stage `data/`, push after Task 4.
- Fixed values:
  - Sample = the same 200 `test_task` steps as `results_test_200.jsonl` (seed-0 shuffle, gold-surviving filter). Verify by query equality and 0 sub-goal cache misses.
  - Tile 1280×1280, patch grid 32×32 (40 px), white bottom padding.
  - Model `vidore/colpali-v1.3-hf`, bf16.
  - Vision device from env `LENS_VISION_DEVICE` (default `mps`); text rankers use `LENS_DEVICE=cpu`.
  - Gate: `colpali` R@10 ≥ `single` R@10 + 5.0 pts.
- Fixed tile shapes are the MPS out-of-memory mitigation (D11: memory grows per distinct input shape). If MPS still trips the swap watchdog, rerun with `LENS_VISION_DEVICE=cpu` (slow, possibly hours; launch with `nohup`) and report it. Never disable the watchdog.
- Don't change the D7 gate code, existing ranker caps, or cache keys.

## Codex execution notes

- **Sandbox needs:**
  - the ColPali model download (Task 3) needs network, so it needs network access.
  - `run_evals.py` / `../.gpu/gpu-run` need execution outside the workspace sandbox (the lock lives in `../.gpu/`); request approval as in brief 5.
  - The model caches in `~/.cache/huggingface` are already populated for ColBERT, bge, the cross-encoder and Qwen.
  - If any of these is denied, stop and report it. Don't work around it.
- **`ps` is blocked** in your sandbox. To wait on a background run, poll its log for the `GATE:` line (e.g. every 60 s) instead of checking PIDs.
- **Long runs (Task 4 Step 1):** launch with `nohup <command> > /dev/null 2>&1 &` (the log goes to the run_evals LOG path), then poll. Rows are written incrementally, so a stopped run keeps its finished steps; `eval.py --summarize` works on partial files.
- **Low on usage?** Stop at a task boundary, commit, push, and put one status line at the top of `BRIEF_8_REPORT.md` (create it if needed): `STATUS: stopped after Task N; next = Task N+1 Step M`. A Sonnet agent will resume from the ticked checkboxes.
- **Deviations:** if the real data or library contradicts the plan (e.g. pylate exposes the `[D]` marker differently), make the smallest change that satisfies the plan's tests and intent, and list it under "Deviations" in the report. Don't change the fixed values, the gate, or the D1-D12 decisions.
- **Final message:** under 25 lines. Give the D12 verdict line, the R@10 row for each ranker, dataset facts, the commit hashes, the push status, and any deviations or blockers.

## Review Focus

1. **Screenshots are JPEG bytes in `.png` files.** Load by content (`PIL.Image.open`), never by extension. Pinned: Task 3 `test_loads_jpeg_named_png`.
2. **Boxes partly off-page, zero-size, or missing (`bounding_box_rect` absent or `"0,0,0,0"`):** score `-inf` (ranked last), never crash. A box partly below the image keeps its on-image patches. Pinned: Task 3 `test_box_edge_cases`.
3. **Patch-embedding order:** use only positions where `input_ids == processor.image_token_id`, never "the first 1024 rows". Pinned: Task 3 `test_image_token_selection`.
4. **An element spanning two tiles:** take the max per query token across all its patches in both tiles. Pinned: Task 3 `test_box_spanning_tiles`.
5. **A missing screenshot** (1 known uid) or a cache miss for a deleted tile file: every node gets `-inf` for that step, and the step is counted in the report as `no_screenshot`. Pinned: Task 3 `test_missing_screenshot`.

---

### Task 1: Audit and review cuts

**Files:**
- Modify: `lens/rank.py`
  - Delete `encode_doc_tokens`, `rank_page` and the context caps; `_model(name)` loses its `node_text` param.
  - `ALLOWED_RANKERS = RANKERS + ("colpali",)`. It is declared here; Task 3 wires it.
- Modify: `lens/nodes.py`
  - Delete `page_words`, `Node.context` and the context extraction.
  - Keep `_page_root` (the iframe fix) for `extract`.
- Modify: `lens/pick.py`
  - Delete the Qwen branch of `pick()` (`CTX`, truncation, `visible_refs`).
  - `pick(text, picker="haiku")`; any other picker raises `ValueError`.
  - Keep `_load` for `subgoal()`.
- Modify: `eval.py`
  - Delete `prep()`, `--prep`, `--node-text`, `--k`, the `_ctx` suffix, `page_textless_gold`, and the `page` row field in `prep_mm`.
  - `STEPS = "data/steps_test.jsonl"`; `--picker` choices `("haiku",)`, default `haiku`.
  - `summarize(rows, llm, query="subgoal")` uses the constant 10 for the gate.
  - The screenshot write skips (logs `missing screenshot <uid>`) when bytes are absent and `path` doesn't exist, instead of crashing.
- Delete:
  - `results.jsonl`, `results_oracle.jsonl`, `results_subgoal_ctx.jsonl`;
  - `CODEX_BRIEF_6.md`;
  - `docs/superpowers/plans/2026-10-25-cleanup-and-haiku-picker.md` and its `.execution.md`;
  - `docs/superpowers/plans/2026-10-26-test-split-and-page-maxsim.md`.
- Test: `test_eval.py`, `test_nodes.py` (drop the tests of deleted code; update calls to the new signatures).

**Interfaces:**
- Produces: `rank(name, query, texts) -> Tensor`; `pick(text, picker="haiku") -> dict`; `summarize(rows, llm, query="subgoal")`; `ALLOWED_RANKERS`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_qwen_picker_rejected(self):
    import lens.pick as picker
    with self.assertRaises(ValueError):
        picker.pick("x", "qwen")

def test_missing_screenshot_does_not_crash_prep(self):
    # 1-row parquet as in test_prep_parses_json_string_fields, but screenshot {"bytes": None, "path": "nope.png"}
    rows = ...  # prep_mm(tmpdir, out, shots)
    self.assertEqual(len(rows), 1)
    self.assertNotIn("page", rows[0])
```

- [ ] **Step 2: Run them.** `.venv/bin/python -m unittest test_eval -v`. Expected: both FAIL (Qwen is still accepted; the bad path raises `FileNotFoundError`).
- [ ] **Step 3: Apply the cuts** listed under Files.
- [ ] **Step 4: Run all tests:** `LENS_DEVICE=cpu .venv/bin/python -m unittest test_eval -v && .venv/bin/python test_nodes.py && .venv/bin/python test_maxsim.py`. Expected: PASS. `grep -rn "pagemaxsim\|page_words\|node_text\|visible_refs" lens eval.py` must print nothing.
- [ ] **Step 5: Commit** with `refactor: remove failed pagemaxsim and settled modes (D13)`.

### Task 2: Element boxes in prep, re-prep, regression

**Files:**
- Modify: `eval.py` (`prep_mm` adds `"boxes": {ref: [x, y, w, h]}`, floats parsed from each candidate's `attributes.bounding_box_rect`; omit refs without a parseable rect)
- Test: `test_eval.py`

**Interfaces:**
- Produces: each steps row has a `boxes: dict[str, list[float]]` key. `data/steps_test.jsonl` is regenerated.

- [ ] **Step 1: Write the failing test.** Extend the 1-row prep fixture: the gold candidate's `attributes` has `"bounding_box_rect": "534,45,151,22"`, and a neg candidate has no rect.

```python
self.assertEqual(rows[0]["boxes"], {"7": [534.0, 45.0, 151.0, 22.0]})
```

- [ ] **Step 2: Run it.** Expected: FAIL (`KeyError: 'boxes'`).
- [ ] **Step 3: Implement it.** The attributes may be a JSON string (as candidates are, per brief 7's deviations).
- [ ] **Step 4: Re-prep and check for regressions:**
  - `.venv/bin/python eval.py --prep-mm data/mm_test_task --steps data/steps_test.jsonl`
  - then `LENS_DEVICE=cpu .venv/bin/python eval.py --n 20 --no-llm --rankers bm25,single,maxsim,cross --output /tmp/reg.jsonl`
  - Expected:
    - 0 sub-goal cache misses;
    - all 80 `gold_rank`s and all 20 `query`s identical to the first 20 rows of `results_test_200.jsonl`;
    - the share of the 200 sampled steps whose gold has a box is reported (it must be ≥ 95%, else stop and report).
- [ ] **Step 5: Run all tests** (Task 1 Step 4 command), then commit with `feat: store candidate bounding boxes in test-split prep`.

### Task 3: ColPali ranker

**Files:**
- Create: `lens/vision.py`
- Modify: `eval.py` (dispatch `colpali` → `rank_colpali(query, s["uid"], [n.ref for n in nodes], s["boxes"])`)
- Modify: `requirements.txt` (add `pillow`)
- Test: `test_vision.py` (unittest; heavy-model tests are guarded by `@unittest.skipUnless(os.environ.get("LENS_VISION_TESTS"), ...)`)

**Interfaces:**
- Produces, all in `lens/vision.py`:
  - `TILE = 1280`, `GRID = 32`, `PATCH = TILE // GRID`, `SHOTS = Path("data/shots")`, `CACHE_DIR = Path("data/colpali")`.
  - `tile_tops(height: int) -> list[int]` returns `[0, 1280, 2560, ...]` covering the height.
  - `box_patches(box: list[float], top: int) -> list[int]` returns row-major patch indices `r*GRID + c` in the tile starting at `top` that overlap the box with positive area. `[]` if none.
  - `load_tiles(uid: str) -> list[PIL.Image.Image] | None` gives RGB 1280×1280 tiles, with the bottom one padded white. `None` if the screenshot is missing.
  - `encode_tiles(uid: str) -> list[torch.Tensor] | None` gives one `[1024, d]` float32 CPU tensor per tile, of L2-normalized patch embeddings selected by `input_ids == processor.image_token_id`. It caches to `CACHE_DIR/{uid}.pt` and loads that on a hit.
  - `rank_colpali(query: str, uid: str, refs: list[str], boxes: dict[str, list[float]]) -> torch.Tensor` gives one score per ref: `Σ_i max_{p ∈ patches(ref)} q_i·p`, or `-inf` when there are no patches or no screenshot. Query tokens come from `processor.process_queries`.
- The model and processor load lazily and once (`functools.cache`) on `LENS_VISION_DEVICE` in bf16. Batch one tile at a time, so the shape is fixed.

- [ ] **Step 1: Write the failing tests.**

```python
class GeometryTests(unittest.TestCase):
    def test_tile_tops(self):
        self.assertEqual(vision.tile_tops(1889), [0, 1280]); self.assertEqual(vision.tile_tops(1280), [0])
    def test_box_patches_basic(self):
        self.assertEqual(vision.box_patches([0, 0, 40, 40], 0), [0])
        self.assertEqual(vision.box_patches([534, 45, 151, 22], 0), [45, 46, 47, 48, 49])  # x 534-685 -> cols 13-17; y 45-67 -> row 1
    def test_box_spanning_tiles(self):
        b = [0, 1270, 40, 20]                     # crosses y=1280
        self.assertEqual(vision.box_patches(b, 0), [31 * 32]); self.assertEqual(vision.box_patches(b, 1280), [0])
    def test_box_edge_cases(self):
        self.assertEqual(vision.box_patches([0, 0, 0, 0], 0), [])
        self.assertEqual(vision.box_patches([0, 5000, 40, 40], 0), [])
    def test_loads_jpeg_named_png(self):
        # write a 1280x1500 JPEG to tmp/abc.png, patch SHOTS; load_tiles("abc") -> 2 tiles, each (1280, 1280), tile 2 bottom pixel white
    def test_missing_screenshot(self):
        # patch SHOTS to an empty dir: load_tiles("x") is None; rank_colpali("q", "x", ["1"], {"1": [0,0,10,10]})[0] == -inf

@unittest.skipUnless(os.environ.get("LENS_VISION_TESTS"), "loads 3B model")
class ColPaliModelTests(unittest.TestCase):
    def test_image_token_selection(self):
        # one white 1280x1280 tile: encoded patches tensor has shape [1024, d]
    def test_text_region_wins(self):
        # draw "SEARCH" large in patch rows 0-3 and "PRIVACY POLICY" in rows 20-23 of a white tile;
        # box A over SEARCH, box B over PRIVACY; rank_colpali("click the search button", ...) scores A > B.
        # If this fails with the real model, STOP and report (don't replace it with synthetic vectors; see the brief 7 review).
```


- [ ] **Step 2: Run them.** `.venv/bin/python -m unittest test_vision -v`. Expected: FAIL (`ModuleNotFoundError: lens.vision`).
- [ ] **Step 3: Implement** `lens/vision.py` and the eval dispatch; `pip install pillow` into `.venv`.
- [ ] **Step 4: Run the tests.** Run them plain (geometry/IO), then `LENS_VISION_TESTS=1 .venv/bin/python -m unittest test_vision -v` (downloads about 6 GB once; needs network approval). Expected: all PASS. If `test_text_region_wins` fails, stop and report (Global Constraints).
- [ ] **Step 5: Run all tests** (Task 1 Step 4 command + `test_vision`), then commit with `feat: ColPali screenshot-patch ranker (vision piece 2)`.

### Task 4: Run, gate, report

**Files:** Create `BRIEF_8_REPORT.md`; output `results_test_colpali.jsonl`.

- [ ] **Step 1: Run.** `LENS_DEVICE=cpu .venv/bin/python run_evals.py data/colpali_test.log -- --n 200 --no-llm --rankers bm25,single,maxsim,cross,colpali --output results_test_colpali.jsonl` (`nohup`; tile embeddings cache, so reruns resume cheaply). Expected: 200 rows and 0 sub-goal cache misses.
- [ ] **Step 2: Gate line.** `D12 piece 2: colpali R@10 = X, single R@10 = Y, margin = X-Y → PASS if ≥ 5.0 else FAIL`. Also give:
  - R@5/10/20/50/100 for all 5 rankers;
  - colpali R@10 split by `textless_gold` and by op (CLICK/TYPE/SELECT);
  - the `no_screenshot` count;
  - median tiles per step;
  - the median encode s per tile and per step, with device;
  - the embedding cache size on disk.
- [ ] **Step 3: Write `BRIEF_8_REPORT.md`** with:
  - (a) the cleanup summary (lines removed, the regression result);
  - (b) the verbatim `--summarize` output;
  - (c) the gate line and tables;
  - (d) code changes (file:line + why) and test status;
  - (e) deviations;
  - (f) anything surprising.
  Numbers only.
- [ ] **Step 4: Commit** `BRIEF_8_REPORT.md` + `results_test_colpali.jsonl` with `docs: brief 8 results (ColPali gate)`, then `git push`.

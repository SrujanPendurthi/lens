# FRAMES Groundedness Test (D19) Implementation Plan

> **Executor: Codex (`gpt-6.1-sol`).** Execute the tasks yourself in order. Each code task runs TDD: failing test, verify fail, implement, verify pass, commit. Tick each `- [ ]` → `- [x]` in this file as you finish it and commit the ticks with that task's commit. Read **Codex execution notes** before starting. **Task 4 ends in a hard STOP**: the planner labels the judge sample before Task 5 may start.

**Goal:** Test D19. On 100 FRAMES questions, does Lens evidence make Haiku produce fewer unsupported claims than chunked RAG at the same 2,000-token budget and similar speed? Also, does it beat reranked RAG (the product edge)?

**Architecture:**
- **New `lens/evidence.py`** is the Lens core. It turns `question + pages` into an evidence string under a token budget for arms `raw`, `rag`, `lens` and `rerank`, reusing `lens.rank.rank()`, which gains a `length` cap.
- **New `frames.py`** is the harness: fetch and cache pages, answer with Haiku, judge with Sonnet, summarize and gate. Both model calls go through one `_claude()` helper factored out of `lens/pick.py`, with the existing cache.
- **Nothing from Mind2Web is rerun.** Its rankers, cache, `claude -p` flags (203-token overhead), CPU device (D11) and run watchdog are reused as-is.

**Tech Stack:** Python `.venv`, torch, sentence-transformers (bge-small, ms-marco-MiniLM cross-encoder), pylate (answerai-colbert-small), transformers tokenizer, stdlib `urllib`/`csv`/`ast`, `huggingface_hub`, Haiku 5.5 + Sonnet 5.5 via `claude -p` (Pro plan).

**Spec:** `PLAN.md` D19 (all values below are copied from it).

## Global Constraints

- **Repo rules:**
  - Commit per task; push `origin main` after each task (no force-push).
  - Never stage `data/` except `data/llm_cache.jsonl`, which gets its own `chore: llm cache` commit.
  - Never touch `../lagunaLab`, `../moeStream`, `../.gpu` contents.
  - Model jobs go through `run_evals.py`.
- **Sample:** `google/frames-benchmark` `test.tsv`, `random.Random(0).sample` of 100 of the 824 rows. Pages are the row's `wiki_links` (a Python-literal list).
- **Pages:**
  - Fetched once from the Wikipedia API as plain-text extracts, with revision id and timestamp recorded.
  - Stdlib `urllib`, `User-Agent: lens-research/0.1 (github.com/SrujanPendurthi/lens)`, at most 1 request/s.
  - A question with any page that fails to fetch is replaced by the next row in the seeded order. Report replacements.
- **Token counting:** the `BAAI/bge-small-en-v1.5` tokenizer, without special tokens, everywhere a budget is checked. Haiku's real prompt tokens come from the `claude -p` usage and are reported.
- **Budgets:**
  - `rag`/`lens`/`rerank` evidence ≤ 2000 tokens.
  - `raw` ≤ 150000 tokens (truncated at a page boundary first, then mid-page); report the truncation rate.
  - The judge sees the same 150k-capped pages.
- **Arms:**
  - **`rag`:**
    - 256-token chunks with 32-token overlap, sliced from the original text by tokenizer offsets (keeps case).
    - Scored by `rank("single", q, chunks, length=256)`.
  - **`lens`:**
    - Sentences from `re.split(r"(?<=[.!?])\s+|\n+", text)` (non-empty, stripped).
    - Each sentence is scored by `rank("maxsim", q, windows, length=128)`, where its window = previous + itself + next sentence.
    - Packed greedily by score; selected sentences that are adjacent within a page are merged into one passage in document order.
  - **`rerank`:**
    - `rag` chunks, top 50 by `single`.
    - Rescored by `rank("cross", q, top50, length=512)`, then packed greedily.
  - **Packing:**
    - Units are added in descending score while the rendered evidence stays within budget; a unit that would overflow is skipped and later ones are still tried.
    - Rendered as `"[<page title>] <text>"` blocks joined by `"\n\n"`.
- **Device:** `LENS_DEVICE=cpu` for all rankers. Retrieval ms = page encoding + scoring + packing for that question (no cross-question cache).
- **Answer prompt (Haiku, `claude-haiku-5-5`), exact system prompt:** `Answer the question using only the evidence provided. Be concise. If the evidence is not enough, reply exactly: I could not find this.` User message: `Evidence:\n{evidence}\n\nQuestion: {question}`.
  - **Abstained** = the answer, stripped, starts with `I could not find this` (case-insensitive).
- **Judge (Sonnet, `claude-sonnet-5-5`):**
  - One call per question covering all 4 answers, anonymized as `a1`–`a4` in `random.Random(question_id)`-shuffled order. The exact prompt is in Task 3.
  - Timeout 600 s.
- **Gate (D19), C=`lens` vs B=`rag`:**
  - Per question, compare unsupported-claim counts: win if C < B, loss if C > B, ties dropped.
  - **PASS** needs all of:
    - `wins − losses ≥ 5`;
    - `p = Σ_{k≥wins} C(n,k) / 2^n < 0.05`, with `n = wins + losses`;
    - pooled unsupported rate C < B (unsupported ÷ claims over all answers);
    - abstention rate C ≤ B + 10 pts;
    - median retrieval ms C ≤ 1.25 × B.
  - The same rule with D=`rerank` in place of B is the **product edge** line. Reported, not gated: `rerank` vs `rag`, accuracy, `raw`.
- **Judge check:** 30 claims, seeded sample from the dry-run questions. Agreement between Opus and Sonnet labels ≥ 85% (after the user resolves disagreements via `final_label`), or Task 5 does not start.

## Codex execution notes

- **Approvals:**
  - Wikipedia and HF downloads need network.
  - `claude -p` and `run_evals.py` run outside the sandbox.
  - If denied, stop and report.
- **Pro quota:**
  - About 400 Haiku answers (~4M tokens, mostly the raw arm) and 100 Sonnet judge calls (~4M tokens).
  - On a usage-limit or rate error, `_claude` raises and nothing is cached. Wait for the reset and rerun the **same** command; cached calls replay.
  - Never shrink the sample or arms to fit a limit.
- **`ps` is blocked:** poll logs every 60 s.
- **Long runs:** `nohup <command> > /dev/null 2>&1 &`.
- **Low on usage?** Stop at a task boundary, commit, push, and put `STATUS: stopped after Task N; next = Task N+1 Step M` at the top of `BRIEF_12_REPORT.md`. A Sonnet agent resumes from the ticks.
- **Deviations:** smallest change keeping tests and intent; list them in the report. Never change prompts, budgets, arms, sample or gate.
- **Final message:** under 20 lines: replacements, the judge-check agreement, the arm table, both D19 lines, commits, push status, deviations.

## Review Focus

1. **Evidence over budget, or a single unit bigger than the budget:** the evidence must still be ≤ 2000 tokens, oversized units skipped. Pinned: Task 2 `test_pack_respects_budget`.
2. **Judge output mapped to the wrong arm after the shuffle:** labels must land on the right arm regardless of order. Pinned: Task 3 `test_judge_maps_shuffled_ids`.
3. **Abstaining answers inflating "groundedness":** zero claims, counted in abstention, and the guard can fail the gate. Pinned: Task 3 `test_d19_gate`.
4. **Mind2Web defaults changing through the new `length` param, or Haiku pick keys changing through the `_claude` refactor:** pinned by `test_rank_length_default_unchanged` and the existing `test_haiku_pick_key_unchanged` / `HaikuTests`.
5. **Chunk text losing case or characters** (bge is uncased, so decoding would lowercase it): chunks are slices of the original text. Pinned: Task 2 `test_chunks_are_original_slices`.

---

### Task 1: Plumbing: `length` caps + `_claude` helper

**Files:**
- Modify: `lens/rank.py`, `lens/pick.py`, `test_eval.py`

**Interfaces:**
- Produces:
  - `rank(name, query, texts, length=None)`. `length` overrides the cap (single `max_seq_length`, ColBERT `document_length`, cross `max_length`). `None` keeps 128/64/128 exactly. `_model` caches per `(name, length)`.
  - `_claude(model: str, system: str, text: str, timeout: int = 120) -> dict`, returning `parse_claude_json` output. `pick()` uses it with unchanged flags, system prompt and cache key.

- [x] **Step 1: Write the failing test** `test_rank_length_default_unchanged`.
  - Patch `sentence_transformers.SentenceTransformer` and `sentence_transformers.CrossEncoder`; clear `_model`'s cache.
  - `_model("single").max_seq_length == 128` and `_model("single", 256).max_seq_length == 256`.
  - `CrossEncoder` is called with `max_length=512` for `_model("cross", 512)`.
- [x] **Step 2: Run** `LENS_DEVICE=cpu .venv/bin/python -m unittest test_eval -v`. Expected: the new test FAILS.
- [x] **Step 3: Implement both interfaces.**
- [x] **Step 4: Run** `LENS_DEVICE=cpu .venv/bin/python -m unittest test_eval test_train -v && .venv/bin/python test_nodes.py && .venv/bin/python test_maxsim.py`. Expected: PASS, with `HaikuTests` and `test_haiku_pick_key_unchanged` unchanged.
- [x] **Step 5: Commit** `refactor: ranker length caps; shared claude -p helper` and push.

### Task 2: `lens/evidence.py`

**Files:**
- Create: `lens/evidence.py`, `test_evidence.py`

**Interfaces:**
- Consumes: `rank(name, query, texts, length)` (Task 1).
- Produces:
  - `count(text: str) -> int`;
  - `chunks(text: str, size=256, overlap=32) -> list[str]`;
  - `sentences(text: str) -> list[str]`;
  - `pack(units: list[tuple[str, str, float]], budget: int) -> str`, where units are `(page_title, text, score)`;
  - `evidence(arm: str, question: str, pages: dict[str, str], budget: int = 2000) -> tuple[str, dict]`. `arm` is one of `raw|rag|lens|rerank`. The stats dict is `{"ms": float, "tokens": int, "units": int, "truncated": bool}`.

- [x] **Step 1: Write the failing tests** (`test_evidence.py`, unittest; mock `lens.evidence.rank` everywhere):
  - `test_chunks_are_original_slices`:
    - `text = "Alpha Beta. " * 400`;
    - every chunk is a substring of `text`, `count(chunk) <= 256`, and the first 32 tokens of chunk *i+1* equal the last 32 of chunk *i*;
    - `"Alpha"` keeps its capital.
  - `test_pack_respects_budget`:
    - units: a 300-token one with score 9, and ten 150-token ones with scores 8..; budget 1000;
    - `count(result) <= 1000`;
    - the highest-scoring unit appears first, and a unit too big to fit is skipped while later smaller ones are still added.
  - `test_lens_merges_adjacent`:
    - page `"S0. S1. S2. S3. S4."`; mocked `rank` scores windows `[0, 0, 5, 4, 0]`;
    - output contains `"[P] S2. S3."` as one block and no separate `"[P] S3."`.
  - `test_rerank_uses_top50_then_cross`:
    - 60 chunks; the mocked `single` returns ascending scores and the mocked `cross` returns descending;
    - `cross` is called with exactly the 50 best-by-`single` chunks, and the final order follows `cross`.
  - `test_raw_truncates_at_cap`: pages totalling over the cap give `stats["truncated"] is True` and `count(text) <= cap`.
- [x] **Step 2: Run** `.venv/bin/python -m unittest test_evidence -v`. Expected: FAIL (`ModuleNotFoundError`).
- [x] **Step 3: Implement** per the Global Constraints arms and packing. Load the tokenizer once with `functools.cache`.
- [x] **Step 4: Run all tests.** Expected: PASS.
- [x] **Step 5: Commit** `feat: Lens evidence arms (raw/rag/lens/rerank)` and push.

### Task 3: `frames.py` harness + page snapshot

**Files:**
- Create: `frames.py`, `test_frames.py` (`data/frames/` is already ignored by `data/*`)

**Interfaces:**
- Consumes: `evidence()` (Task 2); `_claude()` and `_cached()` (`lens/pick.py`).
- Produces:
  - **CLI:** `frames.py --prep | --run --n N --output PATH | --summarize PATH | --judge-sample PATH`.
  - **`data/frames/questions.jsonl`:** `{"id": int, "question", "answer", "pages": [url, ...]}`.
  - **`data/frames/pages/<sha1(url)>.json`:** `{"url", "title", "revid", "timestamp", "text"}`.
  - **Result rows:** `{"id", "arms": {arm: {"answer", "abstained", "evidence_tokens", "prompt_tokens", "ms", "claims": [{"claim", "label"}], "verdict"}}}`.
  - `answer(question, evidence) -> dict`, cached under key `sha1("frames-answer-v1\n" + HAIKU + "\n" + prompt)`.
  - `judge(q: dict, pages: dict[str, str], answers: dict[str, str]) -> dict[str, dict]`, cached under `sha1("frames-judge-v1\n" + SONNET + "\n" + prompt)`.
  - `gate(rows, c, b) -> str`.

- [ ] **Step 1: Write the failing tests** (`test_frames.py`, with `_claude` mocked):
  - `test_judge_maps_shuffled_ids`:
    - answers for 4 arms; the mocked judge JSON labels the `a`-id holding the `lens` answer with 2 UNSUPPORTED claims and the others with 0;
    - the result has `lens` with 2 unsupported, the other arms 0, for 3 different question ids (different shuffles).
  - `test_d19_gate`:
    - 16 decided questions, C wins 12 / loses 4, plus 4 ties; abstention C 10% vs B 5%; ms C 100 vs B 90;
    - output contains `wins 12, losses 4, net +8, p=0.038` and `→ PASS`.
    - With C ms 120 vs 90 (1.33×), the line ends `→ FAIL (speed)`. With abstention C 20% vs B 5%, it ends `→ FAIL (abstention)`.
  - `test_abstained_has_zero_claims`: answer `"I could not find this."` gives `abstained True` and `claims == []`, even if the judge returns claims for it.
- [ ] **Step 2: Run** `.venv/bin/python -m unittest test_frames -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - The exact judge system prompt:
    ```
    You grade answers for groundedness. SOURCE PAGES are the complete pages the answers could use. For each ANSWER: split it into atomic factual claims; label each claim SUPPORTED if the source pages state or directly entail it, else UNSUPPORTED; set verdict CORRECT if it matches the GOLD ANSWER in substance, else INCORRECT. An answer saying it could not find the information has no claims and verdict ABSTAINED. Reply with JSON only: {"answers": [{"id": "a1", "claims": [{"claim": "...", "label": "SUPPORTED"}], "verdict": "CORRECT"}]}
    ```
    The user message is `SOURCE PAGES:\n{pages as "[title]\n{text}" joined by "\n\n"}\n\nQUESTION: {question}\nGOLD ANSWER: {answer}\n\n` followed by `ANSWER a1:\n{text}` blocks.
  - `summarize` prints one row per arm:
    - claims, unsupported, pooled unsupported %;
    - abstain %, accuracy % (CORRECT ÷ 100);
    - median evidence tokens, median Haiku prompt tokens raw and −203;
    - median retrieval ms.
  - Then it prints three lines:
    - `D19 lens vs rag: …`;
    - `D19 product edge lens vs rerank: …`;
    - `reported rerank vs rag: …`.
  - `--judge-sample` writes 30 claims, seeded from all claims of the first 10 questions, as `{"id", "question", "page_titles", "claim", "sonnet_label", "opus_label": null, "final_label": null}`.
- [ ] **Step 4: Run all tests.** Expected: PASS.
- [ ] **Step 5: Prep.**
  - Run `.venv/bin/python frames.py --prep`. Expected: 100 questions and their pages saved.
  - Print the page count, the median and max tokens per question, the replacements, and the snapshot date range.
- [ ] **Step 6: Commit** `feat: FRAMES groundedness harness` and push.

### Task 4: Dry run + judge sample, then STOP

**Files:**
- Create: `frames_judge_check.jsonl`, `BRIEF_12_REPORT.md` (status + dry-run notes)

- [ ] **Step 1: Run** `LENS_DEVICE=cpu .venv/bin/python run_evals.py data/frames_dry.log -- frames.py --run --n 10 --output results_frames.jsonl`. Expected: 10 rows with all 4 arms judged, and evidence tokens ≤ 2000 for rag/lens/rerank.
- [ ] **Step 2: Run** `.venv/bin/python frames.py --judge-sample results_frames.jsonl`. Writes `frames_judge_check.jsonl` with 30 claims.
- [ ] **Step 3:** Write `BRIEF_12_REPORT.md` with `STATUS: waiting for judge labels (Task 4)`, the dry-run `--summarize` output, and the retrieval ms per arm. Commit `docs: FRAMES dry run + judge sample` + `chore: llm cache`, and push.
- [ ] **Step 4: STOP.** Final message: "Judge sample ready: frames_judge_check.jsonl". Do not start Task 5.

### Task 5: Gated run (only after labels)

**Start condition:** in `frames_judge_check.jsonl`, every row has `opus_label` set, and every row where `opus_label != sonnet_label` has `final_label` set. Agreement, defined as rows where `sonnet_label` equals (`final_label` or `opus_label`) divided by 30, must be ≥ 0.85. Otherwise STOP and report the agreement.

**Files:**
- Modify: `results_frames.jsonl` (extended to 100), `BRIEF_12_REPORT.md`, `PLAN.md`

- [ ] **Step 1: Run** `LENS_DEVICE=cpu .venv/bin/python run_evals.py data/frames.log -- frames.py --run --n 100 --output results_frames.jsonl`. The first 10 questions replay from the cache. Expected: 100 rows.
- [ ] **Step 2: Write `BRIEF_12_REPORT.md`:**
  - the judge-check agreement;
  - the verbatim `--summarize` output;
  - the per-arm table;
  - both D19 lines;
  - the raw-arm truncation rate;
  - accuracy per arm;
  - 3 example questions showing rag vs lens evidence snippets and unsupported claims;
  - deviations.
- [ ] **Step 3: `PLAN.md`:**
  - Under D19, add `Outcome: PASS|FAIL (<reason>); product edge PASS|FAIL`.
  - Add a `- Brief 12 (...)` line to Results.
- [ ] **Step 4: Commit** `docs: brief 12 FRAMES groundedness (D19)` + `chore: llm cache` and push. **Stop.** Next direction (knowledge graph or re-grill) is the planner's.

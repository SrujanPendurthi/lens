# TokenLens v1: offline MaxSim-perception eval on Mind2Web

## Context
lens/ is an empty git repo with three docs (the PRD is truncated after FR-2.2). The docs describe four projects at once: an encoder plus compression, a CUDA/PLAID index, a yield-and-inspect protocol, and RL. The grilling cut this down to one falsifiable question:
**Does token-level late interaction (ColBERT MaxSim) over DOM nodes let a small LLM pick the right element with far fewer prompt tokens than the full DOM list, without losing accuracy?**
If not, the multi-vector, quantization and PLAID stack is unjustified and doesn't get built.

## Decisions (from grill)
- D1 Compute: dev on the local M4 (24 GB) using torch MPS. Deploy later on cloud CUDA, so the encoder takes a device string and contains no MLX-only code. lagunaLab is idle, but real-model runs still go through `../.gpu/gpu-run` with a 20 GB cap.
- D2 Grounding: one DOM node per element, with a stable ref (Mind2Web `backend_node_id`). Keep a `token_idx -> node` map, and score each node as the max over its tokens. The LLM sees the top-k as `[ref] tag "text"` lines and answers with a ref or `none`. Text only.
- D3 Data: Mind2Web offline, about 200 steps (cleaned_html, gold element per step).
- D4 Rankers, all given the same candidates and k: BM25, single-vector `BAAI/bge-small-en-v1.5`, ColBERT MaxSim `answerdotai/answerai-colbert-small-v1`, and cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` as the ceiling.
- D5 Deferred: PLAID/1-bit quantization, yield-and-inspect, RL/GAR, ColPali.
- D6 (superseded by D6′ for the ranker query) Picker: `mlx-community/Qwen3-4B-Instruct-2507-4bit` through mlx_lm, temperature 0. Q is the task plus the last 3 actions. The baseline prompt (all nodes) is truncated to the context window, and a cut gold element counts as a baseline failure. LLM outputs are cached as jsonl.
- D7 Gate (k=10). Pass requires all three: MaxSim recall@10 ≥ 85%, pick accuracy ≥ baseline − 2pt, and median prompt tokens ≤ 20% of baseline. Kill if MaxSim recall@10 ≤ single-vector, or if accuracy < baseline − 5pt. Also report encode ms per page.

- D6′ (2026-10-23, after run 1) Ranker query = next **sub-goal**, not the whole task (per PRD: Intent Vector = the active sub-goal).
  - `subgoal` arm: Qwen writes one line, "the next thing to do on this page", from the task + last 3 actions. The ranker scores against that line. Sub-goal prompt + output tokens count toward the TokenLens token cost.
  - `oracle` arm (upper bound): Q = f"{op} {value} {gold tag}" from Mind2Web `operation`, with no element text. It shows what a perfect sub-goal would buy.
  - `task` arm (run 1 query) kept for comparison.
  - ColBERT `query_length` back to 32 (the default). 64 doubled [MASK] augmentation; the memory bound comes from `document_length`.
  - Gate D7 unchanged; it is evaluated on the `subgoal` arm.

- D8 (2026-10-24, after brief 2) **The MaxSim kill stands for zero-shot.** It lost to single-vector under all 3 query modes (task/oracle/sub-goal R@10: 4.5/8.5/15.5 vs 14.0/15.5/24.0). Likely cause: DOM nodes are already about 3-4 words, so one vector per node is already near token granularity. Do not build the PLAID index, 1-bit quantization or CUDA kernels. Only a trained-ranker comparison could reopen this.
- D9 Next: (a) the baseline picker on the full list, the number every direction depends on; (b) a recall rerun with context-enriched node text (ancestor text, MindAct-style). Gold is often generic ("Buy Now", "View details"). If MaxSim still trails under (b), the verdict is final.

- D10 (2026-10-25, after brief 3 was stopped at 53/200 by the swap watchdog) **Qwen3-4B is ruled out as the picker.** Baseline 3/53 = 5.7% (95% CI ~2-15%, far below the 30% "picker is the bottleneck" line), median 11,264 prompt tokens, TTFT 55 s. Run A is NOT resumed: 147 more steps can't change the branch. The local picker is replaced by **Haiku 5.5 through headless `claude -p`** (Pro plan, no API key), on the same first 53 steps. TTFT through the CLI is not comparable to API TTFT, so prompt tokens are the primary efficiency metric.

- D11 (2026-10-25, after brief 5 stopped) Context-mode MPS out-of-memory: the same per-shape graph-cache growth, made worse by the context caps (128/192), plus system swap already high. Remaining runs use **`LENS_DEVICE=cpu`** for rankers. Encode ms is not meaningful on CPU; latency is not the question here (deploy is CUDA). The Haiku run was not affected by the blocker and goes first. Haiku token counts include ~200 tokens of fixed CLI overhead per call (203-token probe); report the raw value and the value minus the overhead.

- D12 (2026-10-26) **Back to the original vision, gated by evidence.** The untested pieces of the PRD vision get tested in order of cost: (1) whole-page contextual MaxSim (token vectors encoded with their page neighbors, mapped back to nodes), (2) ColPali screenshot patches mapped to elements by bounding box, (3) fine-tuned ColBERT vs fine-tuned single-vector. Data moves to **Multimodal-Mind2Web `test_task`** (open; screenshots + official test split; fixes the train-shard contamination). **Gate per piece:** recall@10 ≥ single-vector own-text recall@10 + 5 pts on the same 200 test steps. The first piece to pass gets the deferred infrastructure (PLAID, 1-bit quantization, streaming daemon, yield-and-inspect) built around it. If all three fail, re-grill the thesis.

- D13 (2026-10-26, after brief 7) **D12 piece 1 FAILS:** pagemaxsim R@10 17.5% vs single 26.5% on 200 test_task steps. Cause, verified: contextual encoding blends neighboring nodes inside a window. In a toy case, per-node MaxSim ranks Search over Privacy 31.2 to 30.3, a shared window flips it 21.9 to 24.2, and filler separation restores it. Consequences:
  - pagemaxsim code is deleted.
  - Settled modes go: the context node-text mode (D9), the Qwen picker branch (D10) and train-split prep (D12 moved eval to test).
  - Default picker = haiku; default steps = `data/steps_test.jsonl`.
  - Train-split results (briefs 1-6) predate the iframe-parsing fix and the gold-survival filter, so they are not bit-reproducible. That's accepted; they stay recorded here.
  - Next: piece 2, ColPali screenshot patches, via native `transformers` ColPali (`vidore/colpali-v1.3-hf`), no new dependency. Same gate: R@10 ≥ single + 5 pts (≥ 31.5%) on the same 200 steps.

- D14 (2026-10-27, after brief 8) **D12 piece 2 FAILS** (mean-pooled ColPali 39/200 vs single 53/200; the first run was size-biased). ColPali R@10 1.5% (3/200) vs single 26.5%, but a ranker that just picks the smallest box gets 6.0%. Median gold-rank percentile is 32% (random is 50%), so the scorer has signal but its top ranks are taken by large boxes. Cause: the box score Σ_q max over the box's patches never decreases with box size, because a container covers every patch its child does. Fix: mean-pool over the box's patches; that is pre-registered as the only variant. Rerun on cached tile embeddings, after a visual box-alignment check. Same gate (≥ 63/200). Plan: docs/superpowers/plans/2026-10-27-colpali-fix-and-audit.md.

  - Alignment 4/5; even if all 39 hits came from the aligned 80% and the rest scored equally, the ceiling is ~49/200 < 63, so FAIL stands.

  - Outcome: FAIL, colpali-mean R@10 = 39/200 vs single 53/200 (margin -7.0 pts).

- D15 (2026-10-27, after brief 9) **Piece 3 setup (pre-registered).** Fine-tune both `bge-small` (single) and `answerai-colbert-small` (MaxSim) on identical Multimodal-Mind2Web `train` triplets: (cached Qwen sub-goal, gold own-text, seeded same-page hard negative). The `train` split is cross-task with `test_task`; the shared-task count must be 0. Same hyperparameters for both: seed 0, 1 epoch, lr 2e-5, batch 32, warmup 0.1, CPU, zero-shot length caps. No tuning and no reruns. Gate: maxsim_ft R@10 ≥ single_ft R@10 + 5 pts on the same 200 test steps. PASS → build the deferred infrastructure around fine-tuned ColBERT. FAIL → all three D12 pieces failed; re-grill the thesis. Plan: docs/superpowers/plans/2026-10-27-cleanup-and-finetune.md.

  - Outcome: FAIL, maxsim_ft R@10 = 60/200 vs single_ft 60/200 (margin +0.0 pts). Tie is real: 51 shared hits, 9 unique each; median gold rank 43.5 vs 45.5. Final train losses 3.09/3.25 vs ln 64 ≈ 4.16 chance, so both models learned only modestly in 224 steps; the FAIL holds for this budget only.

- D16 (2026-10-27, after brief 10) **All three D12 pieces failed, so the thesis gets re-grilled.** Multi-vector never beat single-vector by 5 pts: 17.5 (pagemaxsim), 19.5 (ColPali mean), 30.0 vs 30.0 (fine-tuned). Fine-tuning lifted both models (+7 single, +24 maxsim hits at R@10), but test_task reuses the training websites. One open fact goes into the re-grill: does the lift hold on unseen sites (`test_website`)? Pre-registered: single_ft − single R@10 ≤ 0 hits means MEMORIZATION, otherwise GENERALIZES. No other gate, and no new direction before the re-grill. Plan: docs/superpowers/plans/2026-10-27-cleanup-and-website-check.md.

  - Outcome: GENERALIZES (single lift +2 hits; maxsim lift +16 hits).

## Results
- Run 1 (recall-only, Q = task + prev, n=200, median 457 nodes): R@10 bm25 13.0, single 14.0, maxsim 4.5, cross 16.5; R@100 51/50/40/55. Gate formally KILL, but **invalid**: even the cross-encoder ceiling fails, because gold is the next step and the query is the whole task (e.g. gold `button "submit Search"` after typing "Ballet"). Superseded by D6′.
- Brief 2 (recall-only, n=200): oracle R@10 bm25 11.5, single 15.5, maxsim 8.5, cross 10.5. Sub-goal R@10 17.5/24.0/15.5/21.0, R@100 41.5/57.0/47.5/55.0. Sub-goal cost: median 112 tokens, 713 ms. Caveats: the oracle query is weak for CLICK (it carries no target name); the sub-goal can't see the page, so it hallucinates details or skips steps; the query_length change from 64 to 32 is confounded with the query-mode change. Full report: BRIEF_2_REPORT.md.
- Brief 3 (partial, 53/200; stopped by the swap watchdog): Qwen3-4B baseline 5.7% accuracy (CLICK 3/42, TYPE 0/9, SELECT 0/2), 0% parse failures, truncation 5.7%. Own-text rankers reproduce brief 2 exactly. Run B (context recall) not run. Report: BRIEF_3_REPORT.md.
- Brief 5 (partial): Tasks 1-4 done (audit cuts, crash-safe rows, Haiku backend, multi-k metrics). Run B stopped at 67/200 by MPS out-of-memory. On those 67 steps context R@10 was bm25 11.9, single 17.9, maxsim 16.4, cross 17.9; context lowered R@10 for bm25/single/cross and raised MaxSim (10.4 to 16.4) and R@100. The Haiku CLI probe passed at 203 input tokens (cwd=/private/tmp, --safe-mode, --disable-slash-commands). Report: BRIEF_5_REPORT.md.
- Brief 6: Haiku (53 steps) baseline 35.8% at 15,402 tokens; cross@10 13.2% overall but 87.5% when gold was shown, at 968 tokens; 0-2% hallucinated refs. 200-step context R@10: bm25 12.5, single 18.0, maxsim 19.0, cross 19.5 (own text: 17.5/24.0/15.5/21.0). CPU vs MPS ranks agree on 99-100%. Retrieval is the bottleneck, not the picker.
- Brief 7 (test_task, 200 steps, CPU): R@10 bm25 16.0, single 26.5, maxsim 18.0, cross 22.0, pagemaxsim 17.5; R@100 42.5/59.5/50.0/54.0/54.5. Gold coverage 100% of the 1,257 labeled actions; 1,338 screenshots saved (JPEG bytes in .png files). Report: BRIEF_7_REPORT.md.
- Brief 8 (test_task, 200 steps): colpali (max-pooled boxes) R@5 0/200, R@10 3/200, R@20 14/200; single R@10 53/200. Invalid: the size bias in D14. Report: BRIEF_8_REPORT.md.
- Brief 9 (test_task, 200 steps): box alignment 4/5; colpali-mean R@10 39/200 (19.5%) vs single 53/200 (26.5%), margin -7.0 pts → D14 FAIL. ColPali-max R@10 3/200; size-only floor 12/200. Mean median gold-rank percentile 0.141351 vs max 0.324259. All 200 tile caches hit; zero subgoal misses. Report: BRIEF_9_REPORT.md.
- Brief 10 (test_task, 200 steps): 7,163 shared train triplets, train/test task overlap 0. R@10 single 53/200, maxsim 36/200, single_ft 60/200, maxsim_ft 60/200. Fine-tuning lifts +3.5/+12.0 pts; maxsim_ft vs single_ft margin +0.0 pts → D15 FAIL. One CPU epoch each; final train losses single 3.0870015281, maxsim 3.2462501526. Report: BRIEF_10_REPORT.md.
- Brief 11 (test_website, 200 steps): zero shared websites and tasks with train; labeled gold coverage 100%. R@10 single 52/200, maxsim 39/200, single_ft 54/200, maxsim_ft 55/200. Website lifts single +2 hits (+1.0 pt), maxsim +16 (+8.0 pts), versus test_task +7/+24. D16 GENERALIZES. Report: BRIEF_11_REPORT.md.

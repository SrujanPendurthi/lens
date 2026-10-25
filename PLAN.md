# TokenLens v1: offline MaxSim-perception eval on Mind2Web

## Handover status (2026-10-25)

Brief 3 implementation and unit checks are complete, but **D9 is incomplete**.
The full picker run stopped after 53/200 steps when the required watchdog
detected more than 2 GiB of swap growth. The context recall run did not start.
All four own-mode rankers match Brief 2 on the 53 completed steps. Cached
picker outputs and the interrupted log are saved; see `BRIEF_3_REPORT.md`.
D8's zero-shot KILL remains the latest completed decision. Do not describe
the Opus plan as finished until both Brief 3 runs and its final report exist.

## Handover status (2026-10-23)

The 200-step recall evaluation completed: **KILL under D7**. MaxSim recall@10
is 4.5%, versus 14.0% for single-vector. Gold extraction and the trained
ColBERT projection were checked. Stop before picker evaluation and later-phase
infrastructure. See `EVAL_REPORT.md` for evidence, limits, and harness changes;
`results.jsonl` contains the saved per-step results. The original plan follows.

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

## Results
- Run 1 (recall-only, Q = task + prev, n=200, median 457 nodes): R@10 bm25 13.0, single 14.0, maxsim 4.5, cross 16.5; R@100 51/50/40/55. Gate formally KILL, but **invalid**: even the cross-encoder ceiling fails, because gold is the next step and the query is the whole task (e.g. gold `button "submit Search"` after typing "Ballet"). Superseded by D6′.
- Brief 2 (recall-only, n=200): oracle R@10 bm25 11.5, single 15.5, maxsim 8.5, cross 10.5. Sub-goal R@10 17.5/24.0/15.5/21.0, R@100 41.5/57.0/47.5/55.0. Sub-goal cost: median 112 tokens, 713 ms. Caveats: the oracle query is weak for CLICK (it carries no target name); the sub-goal can't see the page, so it hallucinates details or skips steps; the query_length change from 64 to 32 is confounded with the query-mode change. Full report: BRIEF_2_REPORT.md.
- Brief 3 (partial, 53/200; stopped by the swap watchdog): Qwen3-4B baseline 5.7% accuracy (CLICK 3/42, TYPE 0/9, SELECT 0/2), 0% parse failures, truncation 5.7%. Own-text rankers reproduce brief 2 exactly. Run B (context recall) not run. Report: BRIEF_3_REPORT.md.

## Files (fewest possible)
- `pyproject.toml` (uv): torch, pylate (ColBERT load + MaxSim), sentence-transformers, rank-bm25, lxml, datasets, mlx-lm.
- `lens/nodes.py`: cleaned_html → list of `Node(ref, tag, text)` using lxml (text, aria-label, placeholder, value, alt), skipping empty nodes.
- `lens/rank.py`: four rankers behind one function `rank(name, query, nodes) -> scores`. MaxSim is a numpy/torch einsum plus max, and node score is the max over the node's token rows.
- `lens/pick.py`: builds the multiple-choice prompt, calls mlx_lm, parses the ref, and caches by hash(prompt).
- `eval.py`: loads N steps, runs baseline plus the 4 rankers, writes `results.jsonl` plus a summary table (recall@k for k in {5,10,20}, pick accuracy, prompt tokens, encode ms, TTFT), and prints PASS/KILL against D7.
- `test_maxsim.py`: one assert-based check that MaxSim on a toy Q/D matches a hand-computed value, and that node scoring picks the planted gold node.
- Edit `claude_guidance_file.md` §3: change the PLAID/1-bit/CUDA "MUST"s to "later phase (post v1 gate)".

## Steps
1. Verify Mind2Web access. HF `osunlp/Mind2Web`: the train split is open, but the test splits may be a password-gated zip. Fall back to a held-out slice of train if needed, and note that in results.
2. Build nodes.py. Sanity-check that the gold `backend_node_id` survives extraction in ≥ 98% of steps (otherwise extraction is the bug, not ranking).
3. Build rank.py and test_maxsim.py. Run recall@k for all 4 rankers. This needs no LLM and is the cheap early kill signal.
4. Build pick.py. Measure the prefill tok/s curve once, then run baseline vs top-10 per ranker.
5. Run eval.py and get the gate verdict.

## Verification
- `uv run python test_maxsim.py` passes.
- `uv run python eval.py --n 20` runs smoke end to end. Then run `../.gpu/gpu-run lens uv run python eval.py --n 200` for the full run (any tag other than laguna or moe just takes the lock, with no priority hijack).
- The summary table prints PASS or KILL with the numbers from D7.

## Risks to watch
- Mind2Web cleaned_html nodes with empty text (icon buttons) look identical to rankers. Track the share of misses on textless gold nodes.
- Baseline truncation could make the baseline look artificially bad. Report the truncation rate separately.
- Qwen3-4B could be too weak for both arms. If baseline accuracy is < 30%, the picker is the bottleneck, not ranking.
- The PRD is truncated (FR-2.3 onward is empty). Restore it from its source before relying on it.

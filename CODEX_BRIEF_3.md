# Codex brief 3: baseline picker + context-enriched nodes (D9)

Repo: /Users/srujanpendurthi/Projects/Current/lens. Read PLAN.md (D8, D9, Results) and BRIEF_2_REPORT.md. Planning is done elsewhere; you implement, run, and report into `BRIEF_3_REPORT.md`. Commit nothing. Don't touch ../lagunaLab or ../moeStream. Use `.venv/bin/python`. GPU runs go one at a time through `../.gpu/gpu-run lens ...` with the same caps and swap watchdog as brief 2.

## Run A: full picker run (no code changes expected)
`../.gpu/gpu-run lens .venv/bin/python eval.py --n 200 --query subgoal > data/llm_subgoal.log 2>&1`
- The sub-goals are cached, so they will match brief 2 exactly.
- This gives the baseline (all candidate nodes) plus the top-10 picker accuracy for each ranker.
- If `lens/pick.py` / `eval.py` have bugs on the picker path, fix them minimally. Parse failures (no `[ref]`/`[none]` in the output) must be < 5%; report the rate and show 5 raw `out` strings.
- Report the baseline's median prompt tokens, its truncation rate, and median TTFT. Also give accuracy split by op (CLICK / TYPE / SELECT) for the baseline and each ranker.

## Run B: context-enriched node text, recall-only
Code:
1. `lens/nodes.py`: give `Node` a `context: str = ""` field. In `extract`, walk up from the element to the smallest ancestor whose `itertext()` has ≥ 12 words. Set `context` = that ancestor's text with the element's own text removed once, whitespace-collapsed, truncated to 40 words. `Node.line()` stays unchanged, so picker prompts and token counts are not affected.
2. `eval.py --prep` stores `context`. Regenerate it into a new `data/steps_ctx.jsonl` and leave `data/steps.jsonl` alone. Add `--steps PATH` (default `data/steps.jsonl`) and `--node-text {own,context}` (default `own`). In `context` mode the rankers score `f"{text} | {context}"`; everything else stays the same.
3. `lens/rank.py`: for context mode, raise the caps to ColBERT `document_length=128` and cross `max_length=192`; keep single at 128. Keep the caps fixed values (they prevent the MPS OOM; see the comment). Make sure own-mode results are unchanged, or report it if they aren't.
4. Check: the same 200 steps get sampled in both modes (same seed and order), and gold extraction coverage stays ≥ 98%.

Run: `../.gpu/gpu-run lens .venv/bin/python eval.py --n 200 --no-llm --query subgoal --steps data/steps_ctx.jsonl --node-text context > data/recall_subgoal_ctx.log 2>&1`

## Report (BRIEF_3_REPORT.md + final message)
- All summary tables + GATE lines, verbatim.
- R@50/R@100 per ranker for run B, next to brief 2's sub-goal numbers.
- 5 examples of gold node text + context (pick generic ones like "Buy Now").
- Code changes (file:line + why); test status (`test_maxsim.py`, `test_eval.py`, plus one new assert for context extraction).
- Anything surprising. If the sandbox, Metal, the network or the lock blocks you, stop and say exactly what failed.

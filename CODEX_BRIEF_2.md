# Codex brief 2: sub-goal query (D6′)

Repo: /Users/srujanpendurthi/Projects/Current/lens. Read PLAN.md (D6′ and Results). Planning is done elsewhere; you implement, run, and report. Commit nothing. Don't touch ../lagunaLab or ../moeStream.

## Why
In run 1 every ranker failed (cross-encoder R@10 16.5%) because the ranker query was the whole task, while gold is the next step. The fix is to rank against the next sub-goal.

## Changes (smallest diff, match existing style)
1. `lens/rank.py`: ColBERT `query_length=64` → `32`. Keep `document_length=64` and the other caps; they prevent the MPS OOM.
2. `lens/pick.py`: add `subgoal(task, prev) -> {text, tokens}`. It reuses `_load`/the cache pattern and sends one prompt like:
   `Task: ...\nPrevious actions: ...\nIn one short line, what is the next thing to do on the page? Name the element and the action, e.g. "click the Search button" or "type Boston into the location box".`
   Use `max_tokens=24` and strip whitespace/quotes. `tokens` = prompt + generated tokens. Cache it like `pick`.
3. `eval.py`: add `--query {task,subgoal,oracle}`, default `subgoal`.
   - `task` = the current query (run 1).
   - `subgoal` = `subgoal(s["task"], s["prev"])["text"]`; add its tokens to every ranker arm's `tokens` when the LLM is on, and record them in the row either way.
   - `oracle` = `f'{s["op"]["op"]} {s["op"]["value"]} {gold_tag}'`, where gold_tag is the tag of the first gold node. Never include gold node text.
   - Write results to `results_{query}.jsonl` and print the query mode at the top of the summary.
   - Print 5 example (task, prev[-1], subgoal, gold node line) rows so the sub-goal quality is visible.
   - `--no-llm` still means "no picker". The subgoal mode needs mlx even with `--no-llm`.
4. `uv` is not installed; use `.venv/bin/python`. `test_maxsim.py` must still pass.

## Runs (in order; GPU work via `../.gpu/gpu-run lens ...`, one at a time)
1. `.venv/bin/python eval.py --n 200 --no-llm --query oracle > data/recall_oracle.log 2>&1`
2. `../.gpu/gpu-run lens .venv/bin/python eval.py --n 200 --no-llm --query subgoal > data/recall_subgoal.log 2>&1`
3. Only if subgoal maxsim R@10 > single R@10: `../.gpu/gpu-run lens .venv/bin/python eval.py --n 200 --query subgoal > data/llm_subgoal.log 2>&1` (the full picker run). Otherwise stop after run 2 and report.

## Report (final message)
- Summary tables + GATE lines from each run, verbatim.
- R@50 and R@100 per ranker per run (from the results_*.jsonl gold_rank).
- The 5 sub-goal examples from the subgoal run.
- Sub-goal median tokens and ms per step.
- Code changes (file:line + why). Anything surprising.
- If the sandbox blocks Metal, the network, or the lock: stop and say exactly what failed; don't bypass it.

# Codex brief 4: context recall (Run B) + Haiku picker via `claude -p` (D10)

Repo: /Users/srujanpendurthi/Projects/Current/lens. Read PLAN.md (D8-D10, Results) and BRIEF_3_REPORT.md. Planning is done elsewhere; you implement, run, and report into `BRIEF_4_REPORT.md`. Commit nothing. Don't touch ../lagunaLab or ../moeStream. Use `.venv/bin/python`.

**Do not resume the Qwen3-4B Run A.** D10 ruled it out.

## 0. Fix first: incremental results
`eval.py` writes rows only at the end, so brief 3 lost the per-ranker picker results when it was stopped. Append each row to the output jsonl as soon as its step finishes (flush). A summary-only mode (`--summarize PATH`) must rebuild the table + GATE from a partial file. Keep `run_evals.py`'s swap watchdog for GPU runs.

## 1. Run B: context recall (no picker)
Same as brief 3 Run B. Sub-goals are cached; nothing loads Qwen unless there's a cache miss (report any misses).
`.venv/bin/python run_evals.py` adapted to run only:
`eval.py --n 200 --no-llm --query subgoal --steps data/steps_ctx.jsonl --node-text context` → `data/recall_subgoal_ctx.log`, `results_subgoal_ctx.jsonl`.
Report R@5/10/20/50/100 per ranker next to brief 2's own-text sub-goal numbers.

## 2. Haiku picker through headless Claude Code
The user is on a Claude Pro plan with **no API key**; use the CLI, which bills against the subscription.
- `lens/pick.py`: add a picker backend selected by `--picker {qwen,haiku}` (default `qwen` so old caches stay valid). The haiku backend runs a subprocess:
  `claude -p --model claude-haiku-5-5 --tools "" --strict-mcp-config --setting-sources "" --no-session-persistence --system-prompt "<short picker instruction>" --output-format json`, with the prompt on stdin.
  Parse `result` for `[ref]`/`[none]`. Record `usage.input_tokens` (+ cache-read/creation input tokens if present), `output_tokens`, and `duration_api_ms` (as `ttft`; note it is total API time, not TTFT).
  The cache key includes the picker model.
- **Verify no hidden context injection:** run one call with a ~10-token prompt. Reported input tokens must be small (< ~300). If they're thousands (hooks/plugins/CLAUDE.md being injected), find the flag that strips them (e.g. `--bare`, env, settings) and report; don't proceed until clean.
- Sequential calls; on rate-limit/usage-limit errors, stop and report how many finished (the cache makes it resumable). No retries in a tight loop.
- Arms on the **same first 53 steps** (`--n 53`, seed-0 order, sub-goal query, own node text):
  - baseline (all candidates)
  - `single` and `cross` at k=10 and k=50
  - `maxsim` at k=10 (for completeness of D8)
  Add `--ks 10,50` and `--arms` if needed; keep it minimal.
- For each arm report accuracy (overall and by CLICK/TYPE/SELECT), median input tokens, median `duration_api_ms`, parse-failure rate, and **hallucinated-ref rate** (picked ref not in the shown list).
- Also report "oracle-in-list accuracy": accuracy restricted to steps where gold was in the shown list. It separates picker skill from ranker recall.

## Report (BRIEF_4_REPORT.md + final message)
- Tables + GATE lines verbatim (the gate on the 53 steps is indicative only; say so).
- Claude CLI version, the exact command line, and the input-token check result.
- Code changes (file:line + why); test status (existing tests + one test for the haiku parser using a canned JSON string, no network).
- Anything surprising. If the sandbox, network, CLI auth or usage limits block you, stop and say exactly what failed.

# Brief 5 execution report — stopped on Metal/memory blocker

Implementation and tests are complete for Tasks 1–4. Task 5 stopped after 67/200 context-recall steps; Task 6 was not launched. The Haiku hidden-context prerequisite passed at 203 input tokens. Docs cleanup is applied, but the requested full context comparison and Haiku accuracy report remain incomplete. No Qwen picker run was resumed.

## Exact blocker

```text
RuntimeError: MPS backend out of memory (MPS allocated: 2.11 GiB, other allocations: 16.01 GiB, max allowed: 17.76 GiB). Tried to allocate 11.34 MiB on private pool.
RuntimeError: swap grew by more than 2 GiB; stopped PID 24113
```

The first error is in `data/recall_subgoal_ctx.log`; the second came from `run_evals.py`. The runner exited with status 1 and the shared GPU lock was released. Per the plan’s stop-on-Metal-blocker constraint, no GPU retry or Haiku evaluation followed. The 20 GiB model caps and all fixed sequence lengths remain in place; no watermark limit was disabled.

## Task 1 regression

First attempt: 20 steps, zero sub-goal cache misses, one cross-encoder mismatch at step 8 (saved gold rank 12, default-batch rank 10). Restoring cross-encoder `batch_size=64` yielded **20 rows / all 80 gold ranks identical** to `results_subgoal.jsonl`, and `subgoal cache misses: 0`. The redundant rank wrapper was removed and ColBERT uses its default batch size. Retaining cross batch size is a deliberate deviation to satisfy the stronger exact-parity requirement.

## Task 5 partial summary — verbatim

This summary was rebuilt with `.venv/bin/python eval.py --summarize results_subgoal_ctx.jsonl`. It demonstrates that completed rows survived the failure. It is not the planned 200-step result.

```text
query=subgoal
node_text=context
steps=67  median nodes=456  textless gold=9.0%
subgoal median tokens=107  ms=699.26
arm             R@5   R@10   R@20  enc ms
bm25          11.9%  11.9%  20.9%       3
single        10.4%  17.9%  29.9%    1197
maxsim        11.9%  16.4%  22.4%    1818
cross         14.9%  17.9%  23.9%    1380
GATE: KILL (recall-only)
```

## Recall comparison

The matched-prefix table compares the same first 67 steps, same ordered tasks, candidate counts and cached queries. Saved sub-goal token/time fields match Brief 2 exactly on all 67 rows. The full-run cache-miss line was not emitted because the run aborted before its final summary.

| Ranker | Representation | n | R@5 | R@10 | R@20 | R@50 | R@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | own, matched prefix | 67 | 14.9% | 20.9% | 25.4% | 35.8% | 40.3% |
| bm25 | context, partial | 67 | 11.9% | 11.9% | 20.9% | 31.3% | 46.3% |
| bm25 | own, Brief 2 full | 200 | 14.0% | 17.5% | 22.0% | 32.5% | 41.5% |
| single | own, matched prefix | 67 | 14.9% | 23.9% | 37.3% | 47.8% | 58.2% |
| single | context, partial | 67 | 10.4% | 17.9% | 29.9% | 46.3% | 58.2% |
| single | own, Brief 2 full | 200 | 18.5% | 24.0% | 31.5% | 41.5% | 57.0% |
| maxsim | own, matched prefix | 67 | 6.0% | 10.4% | 19.4% | 38.8% | 49.3% |
| maxsim | context, partial | 67 | 11.9% | 16.4% | 22.4% | 37.3% | 59.7% |
| maxsim | own, Brief 2 full | 200 | 9.0% | 15.5% | 24.0% | 37.0% | 47.5% |
| cross | own, matched prefix | 67 | 13.4% | 14.9% | 25.4% | 43.3% | 56.7% |
| cross | context, partial | 67 | 14.9% | 17.9% | 23.9% | 35.8% | 52.2% |
| cross | own, Brief 2 full | 200 | 16.0% | 21.0% | 26.5% | 42.5% | 55.0% |

Context MaxSim R@10 is 16.4% (11/67), versus single-vector 17.9% (12/67). This partial recall-only KILL does not complete D9 or the planned 200-step comparison.

## Haiku CLI configuration

Claude Code version: `2.1.310`. Every probe returned `[7]` successfully using the subscription. Input totals include ordinary, cache-read and cache-creation input tokens.

| Probe | Input tokens | <300 gate |
| --- | ---: | --- |
| Original plan command from lens | 954 | fail |
| Disable slash commands from lens | 954 | fail |
| Safe mode + disable slash commands from lens | 583 | fail |
| Safe mode + disable slash commands from /private/tmp | 203 | pass |

Final command (prompt supplied on stdin; the evaluator stays in lens and sets only the CLI child’s working directory to `/private/tmp`):

```sh
claude -p --model claude-haiku-5-5 --tools "" --strict-mcp-config --setting-sources "" --no-session-persistence --safe-mode --disable-slash-commands --output-format json --system-prompt "You pick one web page element. Reply with only its number in brackets, e.g. [123], or [none]."
```

`lens/pick.py` uses exactly this invocation with `input=text`, capture_output, text mode, timeout=120 and cwd=/private/tmp. No API credentials were introduced. `--bare` was not used: installed CLI help says it disables subscription OAuth/keychain authentication. Probe outputs are saved under `data/haiku_probe*.json` and are deliberately not staged, per the plan.

## Task 6 — 53 steps: gate indicative only

**Not run.** No `results_haiku_53.jsonl` or Haiku evaluation summary exists. The Metal blocker stopped execution before Task 6; the 203-token probe verifies setup only, not picker accuracy. Therefore no comparison of `maxsim@10` and `single@10` picker accuracy is available, and the conditional “maxsim + pylate removable” conclusion cannot be made.

## Changes and tests

- `lens/rank.py:36`: removed the redundant empty-cache wrapper; removed explicit ColBERT batching; kept cross batching at 64 for measured parity. Existing MPS caps stay unchanged.
- `lens/pick.py:30`, `:73`, `:78`, `:91`: shared cache helper, byte-identical Qwen keys, transient sub-goal misses, ref/CLI JSON parsing, Haiku subprocess and minimal-context invocation. Errors propagate before any cache write.
- `run_evals.py:41`: generic `LOG -- <eval args>` interface, retaining the shared GPU lock and swap watchdog.
- `eval.py:31`, `:125`: sub-goal-only execution; incremental flushed rows; saved-file summaries with original query metadata; multi-k arms; ranker subsets; candidate membership, hallucinations, acc|shown and per-operation accuracy. D7 uses `maxsim@10` and is INCOMPLETE when absent.
- `claude_guidance_file.md:22`: replaced obsolete sections 3–5 with current offline-eval status; removed obsolete EVAL_REPORT and briefs 2–4 as authorized. Historical outcomes remain in PLAN.md and the reports.
- `test_eval.py`: 24 tests pass under unittest discovery. `test_nodes.py` and `test_maxsim.py` pass. New tests were observed failing before implementation, including interrupted writes, CLI errors, multi-k/schema metrics, minimal CLI arguments and historical-summary labels. `git diff --check` passes.
- A fresh read-only reviewer checked the code and regression evidence. The CLI configuration mismatch and historical-summary mislabel were fixed in one RED→GREEN pass; no critical issues were found.

## Surprises and remaining work

The plan’s removal of empty_cache passed the short own-text regression but the longer context run still exhausted MPS memory; this is evidence that the planned memory cleanup needs reassessment before another run. It does not prove which change caused the failure. Laptop activity can also affect memory pressure and timings.

Reassess the memory behavior before rerunning Task 5, then run the 53-step Haiku evaluation and add its full summary. Do not resume the Qwen picker, remove MaxSim/PyLate, or build later-phase infrastructure on this partial evidence. Partial files are explicitly labeled here and in PLAN.md.

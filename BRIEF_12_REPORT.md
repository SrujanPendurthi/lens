STATUS: waiting for judge labels (Task 4)

# Brief 12: FRAMES groundedness dry run

Judge sample ready: `frames_judge_check.jsonl`. It has 30 seeded claims from the first ten questions; all opus_label and final_label fields are null. Judge-check agreement is **not yet measured**. Task 5 has not started and must not start until the planner labels all rows, disagreements have final_label values, and agreement is at least 85%.

## Snapshot

100 questions in the exact `random.Random(0).sample(range(824), 100)` order, 339 unique referenced pages, **0 replacements**. Median full source tokens per question: 12,870.5; maximum: 81,381. Recorded revision timestamp range: 2021-06-18T06:17:18Z to 2026-10-27T21:47:09Z. Page JSON stores URL, title, revision ID, revision timestamp and plaintext. Partial/extra pages from earlier attempts remain locally but are not referenced by the final manifest.

## Dry run

10/10 questions completed, all four arms answered and judged. All rag/lens/rerank evidence is at most 2,000 tokens; raw at most 150,000. All abstentions have zero claims. Actual sample order matches the prepared manifest. Raw truncation: 0/10. All model work ran through run_evals.py/shared lock/swap watchdog, with LENS_DEVICE=cpu. Watchdog stderr is empty.

Verbatim `frames.py --summarize results_frames.jsonl` output:

```text
questions=10
arm claims unsupported unsupported% abstain% accuracy% evidence_tokens prompt_tokens prompt_minus203 retrieval_ms
raw 25 0 0.00 30.0 60.0 10455 17490 17288 35.03
rag 14 1 7.14 60.0 30.0 1953 3910 3708 809.52
lens 20 0 0.00 40.0 50.0 1999 3892 3689 2655.54
rerank 20 0 0.00 40.0 60.0 1952 3912 3710 1201.95
D19 lens vs rag: wins 1, losses 0, net +1, p=0.500 → FAIL (net, significance, speed)
D19 product edge lens vs rerank: wins 0, losses 0, net +0, p=1.000 → FAIL (net, significance, unsupported rate, speed)
reported rerank vs rag: wins 1, losses 0, net +1, p=0.500 → FAIL (net, significance, speed)
raw truncated: 0/10
```

These are ten-question, uncalibrated judge results, not the final 100-question D19 outcome.

| Arm | Claims | Unsupported | Abstain | Accuracy | Median retrieval ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 25 | 0 | 30.0% | 60.0% | 35.03 |
| rag | 14 | 1 | 60.0% | 30.0% | 809.52 |
| lens | 20 | 0 | 40.0% | 50.0% | 2655.54 |
| rerank | 20 | 0 | 40.0% | 60.0% | 1201.95 |

Prompt counts are the Claude CLI's real input usage. The summary separately reports median prompt tokens and median prompt tokens minus the registered 203-token overhead; each column is rounded independently.

## Implementation and validation

Task 1: ranker length overrides cache per (name, length); defaults remain 128/64/128, with ColBERT query length 32. Shared _claude preserves picker flags, exact system prompt and cache key. The new cap test failed before implementation; unchanged Haiku/cache tests passed. Commit: `2579051`.

Task 2: raw, rag, lens and rerank evidence arms; tokenizer counting without special tokens, original-text offset chunks, greedy rendered-budget checks, adjacency merging and top-50 reranking. Missing-module tests failed before implementation; all five evidence tests passed afterward. Commit: `1176598`.

Task 3: cached API pages, deterministic FRAMES prep, exact answer/judge prompts, answer anonymization/mapping, validated judge JSON before caching, abstention normalization, binomial gate, summaries and judge-sample CLI. The three initial harness tests failed with missing frames; all passed after implementation. Initial commit: `a7ff631`; completed snapshot fixes/checklist: `95199a4`.

Latest verification: all 32 eval/train/evidence/frames tests plus node extraction and MaxSim checks pass. Snapshot and dry-run assertions passed: 100 original IDs, required page metadata, 10 judged result rows, every budget, and 30 unlabeled claims. No Mind2Web inference or training was rerun.

## Deviations and corrections

- Wikipedia returned HTTP 429 after the first ten cached pages and again on retry. Added a failing-then-passing test and bounded retries for the same page, honoring numeric Retry-After (observed 49–50 s; fallback backoff when absent). The one-request-per-second ceiling, User-Agent and Wikipedia APIs stayed as prescribed.
- The initial English-only URL check incorrectly rejected the dataset's mobile-English and Simple English links. Added a regression test and support for their appropriate Wikipedia APIs, then rebuilt the manifest from the original seeded 100 using cached pages. Both provisional replacements disappeared; the final replacement count is zero. No answer calls occurred before that correction.
- Successful prep overwrites replacements.json to remove obsolete failed-attempt entries. All final page selections are defined by questions.jsonl/snapshot.json.
- Launcher stderr is retained in /tmp/ for exact error reporting. Cached pages, TSV and logs remain under ignored data/ and are not staged.

## Handoff

Task 4 ends here with a hard STOP. The planner labels the sample; this assistant did not fill Opus or final labels. No Task 5 run or final PLAN.md D19 outcome was produced. Dry-run results, report, judge sample and checklist are committed together; new LLM cache entries receive a separate chore commit. Push target: origin/main.

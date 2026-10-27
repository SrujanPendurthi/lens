STATUS: Task 3 complete; next = Task 4 dry run, then hard STOP for labels.

# Brief 12: FRAMES groundedness test

Task 1 complete. Added optional ranker length overrides, with model caching per (name, length). Defaults remain single 128, ColBERT query/document 32/64, cross 128. Added shared _claude(model, system, text, timeout=120); pick retains its exact CLI flags, system prompt and cache key. Failed CLI responses still raise and are not cached.

TDD: the new length-cap test initially failed with TypeError for _model(single, 256), then passed. Validation: 22 unittest tests passed, including unchanged Haiku and cache-key tests, plus test_nodes.py and test_maxsim.py. git diff --check passed.

Task 2 complete: lens/evidence.py implements raw, rag, lens and rerank evidence with tokenizer-based budgets. Tokenizer is loaded once from the already-populated bge cache; token counting excludes special tokens. Chunks slice original text using offsets, preserving case. Packing skips oversized units and continues to smaller ones; Lens merges selected adjacent sentences, counting the rendered result against the budget. Raw defaults to its registered 150,000-token cap and prefers whole-page truncation before cutting a single oversized page.

TDD: evidence tests failed on the missing module, then passed. All 27 eval/train/evidence tests, node extraction and MaxSim checks passed. Rankers are mocked in evidence-arm tests; no FRAMES answer/judge calls or inference runs have occurred. The tokenizer warning for text longer than its model maximum is expected: counting/chunking intentionally tokenizes full pages without running them through the encoder.

Stopped at the next task boundary under the plan's low-usage handoff instructions. No FRAMES data downloaded or harness implemented. Replacements, judge agreement, arm metrics and D19 outcomes are not yet measured.

Next: Task 3 Step 1, write the FRAMES harness tests. Preserve Task 4's hard STOP for planner labels; do not start Task 5 without those labels and the required agreement.

Deviations: none in Task 1.


## Task 3: harness and snapshot attempt

Implemented frames.py and test_frames.py. TDD initially failed with ModuleNotFoundError: frames; all 30 tests plus node extraction and MaxSim checks now pass. Tests cover shuffled judge IDs over three seeds, forced zero-claim abstentions, and every prescribed gate fixture (12 wins/4 losses, speed failure, abstention failure). Prompts, model IDs, budgets and the D19 rule are unchanged. Answer/judge output validation happens before caching.

HF test.tsv downloaded: 824 rows with Prompt, Answer and Python-literal wiki_links. Snapshot prep used random.Random(0).sample for the initial 100, the prescribed Wikipedia API/plaintext extract parameters, exact User-Agent and no more than one request per second. Ten page JSON files were saved with revision IDs and timestamps; three questions completed before the API returned HTTP Error 429: Too Many Requests for question 41. Prep exited nonzero with RuntimeError: Wikipedia API access blocked (429); snapshot prep stopped. No completed 100-question manifest or snapshot statistics exist yet.

No Haiku answers or Sonnet judge calls have been made. No judge sample is ready. Tasks 4 and 5 have not started. Resume the same prep command after API access recovers; successful page snapshots will be reused. Do not alter the sample, endpoint, User-Agent, budgets or gate to work around the error.

Deviation: global API access errors (401/403/429) stop snapshot preparation instead of consuming replacement questions against an unavailable API. The failed attempt is recorded in data/frames/replacements.json; it does not constitute a completed replacement sample. Task 3 Step 5 remains unchecked. Partial snapshots and logs are retained locally and not staged.


## Snapshot completed after retry fixes

Snapshot: 100 questions in the exact original random.Random(0).sample order; 339 unique referenced pages; replacements 0. Median/full maximum source tokens: 12,870.5 / 81,381. Revision timestamp range: 2021-06-18T06:17:18Z to 2026-10-27T21:47:09Z. Every referenced page has url/title/text/revid/timestamp. Extra cached pages from the earlier provisional replacement attempt are retained but are not referenced by the final manifest.

HTTP 429 handling now retries the same page up to three times, respecting numeric Retry-After (fallback backoff if absent). The one-request-per-second ceiling remains. A new test failed before implementation and passed after. A second regression test caught the too-narrow English-only URL validator: valid en.m.wikipedia.org URLs now use the canonical English API, and Simple English URLs use their own Wikipedia API. The final snapshot restored both originally sampled questions that had been rejected by that validator. No question replacement was needed. Successful prep overwrites the replacement log, clearing obsolete failed-attempt entries.

All 32 tests plus node extraction/MaxSim checks pass. Task 3 is complete; earlier blocked-status notes above are historical. No answer/judge calls have yet run. Next is the ten-question dry run and judge sample; Task 5 remains prohibited until planner labels and agreement check are supplied.

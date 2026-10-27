STATUS: stopped after Task 2; next = Task 3 Step 1

# Brief 12: FRAMES groundedness test

Task 1 complete. Added optional ranker length overrides, with model caching per (name, length). Defaults remain single 128, ColBERT query/document 32/64, cross 128. Added shared _claude(model, system, text, timeout=120); pick retains its exact CLI flags, system prompt and cache key. Failed CLI responses still raise and are not cached.

TDD: the new length-cap test initially failed with TypeError for _model(single, 256), then passed. Validation: 22 unittest tests passed, including unchanged Haiku and cache-key tests, plus test_nodes.py and test_maxsim.py. git diff --check passed.

Task 2 complete: lens/evidence.py implements raw, rag, lens and rerank evidence with tokenizer-based budgets. Tokenizer is loaded once from the already-populated bge cache; token counting excludes special tokens. Chunks slice original text using offsets, preserving case. Packing skips oversized units and continues to smaller ones; Lens merges selected adjacent sentences, counting the rendered result against the budget. Raw defaults to its registered 150,000-token cap and prefers whole-page truncation before cutting a single oversized page.

TDD: evidence tests failed on the missing module, then passed. All 27 eval/train/evidence tests, node extraction and MaxSim checks passed. Rankers are mocked in evidence-arm tests; no FRAMES answer/judge calls or inference runs have occurred. The tokenizer warning for text longer than its model maximum is expected: counting/chunking intentionally tokenizes full pages without running them through the encoder.

Stopped at the next task boundary under the plan's low-usage handoff instructions. No FRAMES data downloaded or harness implemented. Replacements, judge agreement, arm metrics and D19 outcomes are not yet measured.

Next: Task 3 Step 1, write the FRAMES harness tests. Preserve Task 4's hard STOP for planner labels; do not start Task 5 without those labels and the required agreement.

Deviations: none in Task 1.

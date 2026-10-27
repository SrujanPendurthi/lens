STATUS: stopped after Task 1; next = Task 2 Step 1

# Brief 12: FRAMES groundedness test

Task 1 complete. Added optional ranker length overrides, with model caching per (name, length). Defaults remain single 128, ColBERT query/document 32/64, cross 128. Added shared _claude(model, system, text, timeout=120); pick retains its exact CLI flags, system prompt and cache key. Failed CLI responses still raise and are not cached.

TDD: the new length-cap test initially failed with TypeError for _model(single, 256), then passed. Validation: 22 unittest tests passed, including unchanged Haiku and cache-key tests, plus test_nodes.py and test_maxsim.py. git diff --check passed.

Stopped at a task boundary under the plan's low-usage handoff instructions. No FRAMES data downloaded, no evidence arms or harness implemented, no answer/judge calls made. Replacements, judge agreement, arm metrics and D19 outcomes are not yet measured.

Next: Task 2 Step 1, write the evidence-arm tests. Preserve Task 4's hard STOP for planner labels; do not start Task 5 without those labels and the required agreement.

Deviations: none in Task 1.

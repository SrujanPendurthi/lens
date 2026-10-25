# SDD ledger — plan: docs/superpowers/plans/2026-10-25-cleanup-and-haiku-picker.md

Ruling: Execute in the explicitly specified lens directory with its existing data and virtualenv; no worktree or commits. The plan explicitly changes this checkout and includes uncommitted source documents. The shared parent repository now has commits, contrary to the plan's historical note; the no-commit instruction still stands.
Ruling: Keep this ledger in lens/docs rather than the shared parent .superpowers directory, to respect workspace permissions and avoid sibling state.
Pre-flight: Task 1 cache interfaces feed Task 3; Qwen key compatibility and transient sub-goal miss flags need explicit tests.
Pre-flight: Task 2 incremental schema feeds Task 4 and --summarize; preserve legacy recall rows, infer metadata from saved rows, require maxsim@10 for full picker gate.
Pre-flight: Task 4 multi-k requires retaining --k for summarize compatibility while --ks controls execution; default execution uses 10.
Pre-flight: Tasks 5/6 use the generic watchdog CLI from Task 1; all real-model regression checks also use this runner.
Baseline: 12 evaluator tests, context extraction and toy MaxSim passed before this plan; evaluator baseline rerun below.

Task 1: RED — three cache tests failed on missing _cached, pick_key and transient miss flag. GREEN — 13 evaluator tests and both script tests passed.
Task 1: Regression attempt 1 — 20 rows, zero sub-goal misses; one cross-encoder mismatch on step 8 (old gold rank 12, new 10).
Ruling: Restore cross-encoder batch_size=64 because removing it violates the plan's exact-output regression requirement. Keep the redundant wrapper removed and ColBERT default batching. Regression rerun pending; cost if wrong is one unnecessary batch argument.
Task 2: complete — crash test failed because no output existed, summarize test failed on unknown option; after incremental flush and summarize branch all 15 evaluator tests plus context/MaxSim passed.
Task 3: RED — missing parsers and backend argument; GREEN — 19 evaluator tests plus context/MaxSim passed, including CLI nonzero, is_error, missing-result no-cache cases.
Task 3: CLI 2.1.310; original probe succeeded with [7], but used 954 input tokens. Disabling slash commands also used 954. Hidden-context gate remains open.
Ruling: CLI help says --bare disables OAuth/keychain and requires API-key auth, contradicting the plan's subscription-only/no-API-key requirement. Try documented --safe-mode (preserves auth) before considering the context gate blocked. No auth tokens will be extracted or repurposed.

Task 1: complete — rerun 20/20 rows match all saved gold ranks, subgoal cache misses: 0. Full evaluator suite subsequently 23/23 green plus context/MaxSim scripts.
Task 3: Unit implementation complete; live auth succeeds. Safe-mode probe reports 583 input tokens; hidden-context prerequisite remains blocked (<300 required). User asked asynchronously whether to retain the gate or accept measured overhead. No Haiku evaluation launched.
Task 4: complete — watched unknown flags, missing shown metrics, and incomplete-gate KeyError fail; all 23 tests now pass, plus context and MaxSim scripts. Includes multi-k subset, hallucinated refs and truncated gold checks.
Ruling: Baseline shown refs are intersected with post-truncation visibility for shown, gold_shown, ok and hallucinated metrics. This makes acc|shown reflect the actual prompt, preventing truncated gold from inflating the oracle-in-list denominator.
Task 5: running — 200-step context recall through generic run_evals.py under shared lock/watchdog. No Qwen picker resumed.

Ruling: During this turn, another session initialized lens/.git with bd2f9e6 (capturing the in-progress Task 1–3 code) and committed 08cf94c updating this plan to authorize checkpoint commits and final push. Follow the current plan and the user's earlier commit request. Earlier no-commit ledger text records the superseded instruction, not current policy. Never stage data/.
Task 7: applied authorized obsolete-doc deletions and five-line status section; final report pending Run B and hidden-context gate.

Task 3: complete — neutral-directory safe-mode probe returned [7] using 203 input tokens (<300). Exact final CLI invocation is the safe-mode/disable-slash-commands command, launched with child cwd=/private/tmp. Live subscription auth succeeds; no API credentials or environment auth changes used. Earlier asynchronous overhead question is moot.
Ruling: Only the CLI child runs from neutral /private/tmp; the evaluation/runner still execute from lens as required. This removes injected project metadata without changing prompts, model, auth, tools, MCP or Qwen cache keys.
Final review: fresh read-only reviewer examined full code and uncommitted changes against 08cf94c, independently verified all 80 regression gold ranks, and ran 23 tests. No critical defects.
Final: fixed Important CLI invocation/probe mismatch — augmented Haiku boundary test failed on missing cwd; backend now uses the exact passing 203-token configuration; full suite 24/24 and both scripts green.
Final: Ruling: regrade historical-summary mislabel from Minor to Important because it misattributes experimental evidence. Added saved oracle metadata test, watched query=subgoal fail, then infer saved query_mode (legacy unknown). Full suite 24/24 green.
Final: Ruling: live evaluation quality and usage-limit behavior declined by reviewer remain runtime verification tasks, not waived requirements. Task 5/6 and final report are still pending. Watchdog termination was exercised in prior Brief 3; no extra destructive termination test needed for unchanged watchdog.
Final: Ruling: absent historical 8858909 is outside the new nested repository; current local history starts with bd2f9e6. Review full files and actual worktree diff rather than rewriting history.

Task 5: interrupted at 67/200 rows. Exact log error: MPS backend out of memory (MPS allocated 2.11 GiB, other allocations 16.01 GiB, max allowed 17.76 GiB; requested 11.34 MiB). Runner also raised swap grew by more than 2 GiB; stopped PID 24113. Exit 1; GPU lock released. No retry or watermark bypass.
Task 6: not started — obey stop-on-Metal-blocker. The hidden-context prerequisite passed at 203 tokens, but a full picker run remains outstanding.
Task 7: docs/report complete for the interrupted outcome — authorized obsolete docs removed, guidance condensed, exact errors and 67-row summary saved in BRIEF_5_REPORT.md, matched-prefix/full-reference R@5/10/20/50/100 tables included. Missing full-run and Haiku tables explicitly unavailable. Conditional MaxSim/PyLate removal cannot be concluded because picker comparison is absent.
Runtime verification: --summarize successfully reconstructed 67 flushed rows after actual MPS/watchdog interruption. All 67 task/query/candidate identities and sub-goal token/time fields match the Brief 2 prefix. Partial context recall@10 maxsim=11/67, single=12/67. This does not complete the 200-step gate.
Ruling: Keep saved partial context results as clearly labeled evidence, not a completed result; data/ probe/log artifacts remain unstaged under current plan. Commit docs and partial result, then push the reviewed checkpoint and blocker report as explicitly authorized by the updated plan.

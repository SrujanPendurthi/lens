# SDD ledger — plan: docs/superpowers/plans/2026-10-25-cleanup-and-haiku-picker.md

Ruling: Execute in the explicitly specified lens directory with its existing data and virtualenv; no worktree or commits. The plan explicitly changes this checkout and includes uncommitted source documents. The shared parent repository now has commits, contrary to the plan's historical note; the no-commit instruction still stands.
Ruling: Keep this ledger in lens/docs rather than the shared parent .superpowers directory, to respect workspace permissions and avoid sibling state.
Pre-flight: Task 1 cache interfaces feed Task 3; Qwen key compatibility and transient sub-goal miss flags need explicit tests.
Pre-flight: Task 2 incremental schema feeds Task 4 and --summarize; preserve legacy recall rows, infer metadata from saved rows, require maxsim@10 for full picker gate.
Pre-flight: Task 4 multi-k requires retaining --k for summarize compatibility while --ks controls execution; default execution uses 10.
Pre-flight: Tasks 5/6 use the generic watchdog CLI from Task 1; all real-model regression checks also use this runner.
Baseline: 12 evaluator tests, context extraction and toy MaxSim passed before this plan; evaluator baseline rerun below.

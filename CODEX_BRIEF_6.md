# Codex brief 6: finish brief 5 Tasks 6 + 5 on CPU rankers (D11)

Repo: lens (own git repo). Read PLAN.md D11 and BRIEF_5_REPORT.md. Same Global Constraints as `docs/superpowers/plans/2026-10-25-cleanup-and-haiku-picker.md`; commit + push at the end.

**Code changes:** none expected. If you need any, keep them minimal and tested.

1. **Haiku run first (brief 5 Task 6), rankers on CPU:**
   `LENS_DEVICE=cpu .venv/bin/python run_evals.py data/haiku_53.log -- --n 53 --picker haiku --ks 10,50 --rankers single,cross,maxsim --output results_haiku_53.jsonl`
   - That's 371 sequential CLI calls. On a usage-limit error, stop and report the rows done; rerunning resumes from the cache.
   - Then run `.venv/bin/python eval.py --summarize results_haiku_53.jsonl` and paste the output verbatim.
   - Report median tokens both raw and minus 203 (the fixed CLI overhead).
2. **Then Run B (brief 5 Task 5) from scratch on CPU:**
   `LENS_DEVICE=cpu .venv/bin/python run_evals.py data/recall_subgoal_ctx.log -- --n 200 --no-llm --steps data/steps_ctx.jsonl --node-text context`
   - Keep the previous 67-row MPS file as `data/results_subgoal_ctx_mps67.jsonl` for comparison.
   - Report the CPU-vs-MPS gold-rank agreement on those 67 steps, then the full 200-step side-by-side table (own text from brief 2 vs context).
3. Append both results to `BRIEF_5_REPORT.md` under "Brief 6 continuation", including the brief 5 Task 7 Step 3 conditional (maxsim + pylate removable?). Report only; don't remove anything.
4. Don't stage `data/` or the probe files. Commit the report, then push.

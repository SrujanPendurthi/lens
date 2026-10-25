# Brief 3 execution report — interrupted

**Incomplete.** Run A completed 53 of 200 steps before the required swap watchdog stopped it. Run B did not start. No full-run summary or GATE line was produced. The D8 zero-shot KILL from Brief 2 remains the latest completed decision; D9 remains open.

## Exact failure

```text
RuntimeError: swap grew by more than 2 GiB; stopped PID 2914
```

The watchdog in `run_evals.py:26` terminated the active process group. The evaluation child was PID 2921. The runner exited with status 1, so it did not start the context evaluation. The Metal run used the shared `../.gpu/gpu-run lens` lock and the existing 20 GiB model allocation caps. No GPU retry was attempted after the watchdog stop.

The laptop was also in use. Concurrent activity can affect timings and memory pressure; these observations do not establish the cause of the swap growth. An earlier process check measured zero swap use. Saved timings include concurrent laptop activity and cache hits retain their original generation timings.

## Saved evidence

- `data/llm_subgoal.log`: all 53 completed steps and the shutdown warning.
- `data/llm_cache.jsonl`: cached sub-goals and picker outputs, retained for resumption.
- `results_subgoal.jsonl`: preserved Brief 2 recall results, not overwritten by this run.
- `data/context_audit.json` and `data/context_examples.json`: prepared-data audit and examples.

The evaluator writes result rows and its summary only after the entire run. Consequently `results_picker_subgoal.jsonl` and `results_subgoal_ctx.jsonl` were not created. Full per-ranker picker accuracy and the requested context recall comparisons are unavailable.

## Partial checks — not the planned 200-step evaluation

All four own-text rankers match Brief 2 exactly on all 53 completed steps. The prepared-data audit reports 743 steps, 99.73% gold coverage, identical original nodes and metadata, and matching cached sub-goals.

All 265 saved picker cache entries have 0 parse failures (0.0%). This is below 5% on the saved calls, but does not establish the full-run requirement.

Baseline on the 53 completed steps: accuracy 5.7%; median prompt tokens 11264; truncation 5.7%; median TTFT 55.01 s. These values were reconstructed from the exact baseline prompt hashes and saved cache, not a fresh model run.

| Operation | Partial baseline accuracy |
| --- | ---: |
| CLICK | 3/42 (7.1%) |
| TYPE | 0/9 (0.0%) |
| SELECT | 0/2 (0.0%) |

Five raw baseline `out` strings from the first five sampled steps:

- `"[11005]"`
- `"[25460]"`
- `"[38183]"`
- `"[69817]"`
- `"[11520]"`

## Five context examples

- [11043] button "submit Search": Ballet met Dance Center Cinderella - Ballet American Ballet Theatre Grand Rapids Ballet Washington Ballet Studios The Georgia Ballet Studios
- [40898] a "Buy Now": Ready To Thrill Your Group? Its easy to buy Hurricane Harbor Phoenix group tickets! Purchasing online means fast and flexible planning. Tickets can be purchased up until the date of your visit Print orders up to 99 tickets or get
- [68292] button "button View details": UA883 7:45 am LHR On schedule 10:30 am EWR UA905 9:25 am LHR On schedule 12:35 pm EWR View details UA15 10:00 am LHR On schedule 1:10 pm EWR View details UA17 12:00 pm LHR On schedule 3:10 pm EWR
- [257931] button "Search": Matching Keywords samsung galaxy s23 ultra 5g dual 8gb ram in Cell Phones - Unlocked 8gb ram hyperx 8gb ram 8gb ram laptop hp elitebook i7 8gb ram 8gb ram ddr4 corsair 8gb ram 4 8gb ram neo forza 8gb
- [53878] button "button Continue": Get notified with email When would you like to be notified when your new matches arrive? Instantly Daily summary I don't want alerts

## Implementation and verification

- `lens/nodes.py:13` adds optional context; `:32` selects the smallest qualifying ancestor, removes own text once and caps context at 40 words. `Node.line()` stays unchanged.
- `eval.py:18` saves context during preparation; `:39` accepts a separate steps path and text mode; `:85` passes enriched text only to rankers; `:81` checks baseline visibility after truncation.
- `lens/rank.py:29` uses context ColBERT document length 128 and cross-encoder max length 192. Own-mode caps remain unchanged and single stays at 128.
- `lens/pick.py:70` records refs visible after truncation for baseline scoring.
- `run_evals.py` adds sequential execution with the shared GPU lock, a 2 GiB swap-growth watchdog and process-group termination.
- `.gitignore` excludes virtual environments, bytecode and large regenerable input datasets.
- All 12 `test_eval.py` tests pass. `test_maxsim.py` and `test_nodes.py` pass; context assertions cover nearest ancestor, removal once, 40-word cap and unchanged picker lines.

## Remaining work

Complete Run A and Run B under the required memory controls, then produce the full summary/GATE lines, per-ranker accuracy by operation, baseline statistics, context R@50/R@100 comparison and final D9 verdict. Saved picker outputs are reusable on a resumed run. The sample remains a seed-0 training-shard slice, not an official test split. Do not build PLAID, quantization or CUDA kernels under the existing D8 decision.

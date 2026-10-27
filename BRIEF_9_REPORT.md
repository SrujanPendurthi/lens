STATUS: complete; D14 FAIL; Tasks 1–5 finished.

# Brief 9: ColPali mean pooling and audit

## Box alignment

First five sampled steps with a positive-area gold box; crops use a 40 px margin and a red box. Four of five pass; the pre-registered continuation threshold is at least four.

| UID | Gold node | Result | Observation |
| --- | --- | --- | --- |
| `6ccbade8-62c4-41b8-99bc-2ef2d7c157e1` | `[50374] svg ""` | PASS | Red box frames the visible downward SVG chevron; node text is empty. |
| `050bde66-16bb-4c60-947a-d99b4f0dbd44` | `[380349] button "Ok and Close button OK"` | FAIL | Red box crosses the dimmed area above the Terms and Conditions panel; no OK button is visible inside it. |
| `f4fe43cd-babf-468e-9caf-e5c36c31b19c` | `[178] a "Music"` | PASS | Red box frames the Music navigation link. |
| `62437294-b3b0-443c-9c5c-0eced7350cf2` | `[115330] button "button Select My Car"` | PASS | Red box frames the orange Select My Car button. |
| `eb069158-69f8-4990-9151-b9a81c1c0513` | `[184584] div "Size Size"` | PASS | Red box frames the Size filter row and plus icon. |

Crops and manifest: `data/box_check/`; throwaway script: `/private/tmp/lens-box-check.py`. No alignment code or image artifacts committed.

## Implementation and validation

- Audit: removed picker parameters, truncation/debug output, screenshot path fallback, stale original-plan and handover sections; ignored all `data/*` except the existing LLM cache. Haiku key for `x` remains `699a210c1d6f15d68fc5e6938ce7175bcad71da2`.
- Gate: D12 uses integer hit counts, `(hits - single_hits) * 100 >= 5 * n`. Boundary tests: 63/200 vs 53/200 passes; 62/200 fails. Brief 7's four remaining ranker R@5/10/20 rows reproduce exactly; no `GATE:` output.
- Pooling: concatenate each box's patch similarity columns across all overlapping tiles, then mean over patches and sum over query tokens. Nested-child regression changed from a 1.0/1.0 tie to child 1.0 > container 1/1024.
- Cache: missing model/tile keys mean legacy/current constants; mismatched model or tile re-encodes. New payloads include model and tile metadata. Existing 200 screenshot cache files remain in place.
- Eval timing subtracts tile encoding. The controlled-clock test uses 3 s elapsed minus 2 s encoding and asserts 1,000 ms.
- Plain tests before rerun: 34 run, 32 passed, 2 heavy tests skipped; node and MaxSim checks passed after audit.
- Heavy tests: 14/14 passed through `../.gpu/gpu-run`; Search 7.634003 > Privacy 1.600016 with the real model and mean pooling. No alternative pooling tested.

## Deviations

- Task 2 findings and checklist were committed separately (`de97047`); no alignment code or crops were committed.
- The timing test uses a controlled 3 s clock interval so the old implementation fails the plan's `< 2000 ms` assertion. With the original instantaneous ranker mock, that assertion would pass before the fix.
- Launcher stderr is saved in `/tmp/lens-colpali-mean-watchdog.log` for exact error reporting.

## Results

200/200 rows completed. Subgoal cache misses: 0. Tile cache hits: 200/200. All 200 query strings and single-vector gold ranks exactly match `results_test_200.jsonl`; single R@10 = 53/200. All 200 screenshot cache file sizes and nanosecond modification times are unchanged. Tile re-encodes: 0. Watchdog stderr: empty. Text device: CPU; vision device: MPS.

Verbatim `.venv/bin/python eval.py --summarize results_test_colpali_mean.jsonl`:

```text

query=subgoal
steps=200  median nodes=438.5  textless gold=2.5%
subgoal median tokens=115  ms=806.75
arm             R@5   R@10   R@20  enc ms
single        20.0%  26.5%  33.5%     468
colpali       11.5%  19.5%  29.5%     166
D12 colpali: R@10 = 39/200, single = 53/200, margin = -7.0 pts → FAIL
```

D12 piece 2: colpali-mean R@10 = 39/200 (19.5%), single = 53/200 (26.5%), margin = -7.0 pts → FAIL. Required ColPali hits: 63/200.

| Ranker | R@5 | R@10 | R@20 | R@50 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| single | 40/200 (20.0%) | 53/200 (26.5%) | 67/200 (33.5%) | 94/200 (47.0%) | 119/200 (59.5%) |
| colpali-max | 0/200 (0.0%) | 3/200 (1.5%) | 14/200 (7.0%) | 50/200 (25.0%) | 84/200 (42.0%) |
| colpali-mean | 23/200 (11.5%) | 39/200 (19.5%) | 59/200 (29.5%) | 92/200 (46.0%) | 120/200 (60.0%) |

Size-only floor: 12/200 (6.0%). ColPali-mean: 39/200 (19.5%), +27 hits / +13.5 pts above the floor. The pre-registered “no usable signal” condition (`hits < 12`) is false.

| ColPali-mean subset | Steps | R@10 hits | R@10 |
| --- | ---: | ---: | ---: |
| op=CLICK | 158 | 27 | 17.09% |
| op=TYPE | 30 | 10 | 33.33% |
| op=SELECT | 12 | 2 | 16.67% |
| textless_gold=True | 5 | 1 | 20.00% |
| textless_gold=False | 195 | 38 | 19.49% |

| Pooling | Median gold-rank percentile (`gold_rank / n_nodes`) |
| --- | ---: |
| max | 0.324259 |
| mean | 0.141351 |

## Commits

- Audit cuts: `a2015b9`.
- Alignment findings: `de97047`.
- Mean pooling and cache validation: `881d2a2`.
- Results: this report/results commit. Task 5 removal follows because the gate failed. Results commit: `612807a`.

## Removal after FAIL

Task 5 removed `lens/vision.py`, `test_vision.py`, ColPali dispatch/stat recording, box prep, and the Pillow requirement. Allowed rankers are the four existing text rankers. D12 boundary tests remain, using MaxSim as the comparison name after ColPali removal.

Final verification: 19/19 eval tests, node extraction and MaxSim checks passed. `rg -n 'colpali|vision|boxes' lens eval.py` returned no matches. Brief 7's four remaining ranker R@5/R@10/R@20 rows still match exactly. All 200 ColPali tile cache sizes/mtimes remain unchanged; both ColPali result files still have 200 rows. No `data/` artifacts were staged. Removal is this final cleanup commit.

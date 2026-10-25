# Brief 2 execution report

Both requested 200-step recall runs completed. Sub-goal MaxSim recall@10 is 15.5% (31/200), below single-vector at 24.0% (48/200). D7 is KILL. The condition for the full picker run is false, so it was not run.

Both runs used the shared `../.gpu/gpu-run lens` lock with approved execution outside the sandbox, the configured memory caps, and a PID-based swap watchdog (stops if swap grows by more than 2 GiB). No commits were made and neither sibling project was modified.

## oracle: verbatim summary

```text
query=oracle
steps=200  median nodes=457.0  textless gold=3.5%
arm          R@5   R@10   R@20  enc ms    acc  tok med  ttft s
bm25        4.0%  11.5%  15.5%       1
single      9.0%  15.5%  21.0%     411
maxsim      5.0%   8.5%  12.5%     938
cross       4.5%  10.5%  14.0%     593
GATE: KILL (recall-only)
```

| Ranker | R@50 | R@100 |
| --- | ---: | ---: |
| bm25 | 29.5% | 38.5% |
| single | 39.0% | 56.0% |
| maxsim | 28.0% | 44.0% |
| cross | 28.0% | 44.0% |

## subgoal: verbatim summary

```text
query=subgoal
steps=200  median nodes=457.0  textless gold=3.5%
subgoal median tokens=112  ms=713.32
arm          R@5   R@10   R@20  enc ms    acc  tok med  ttft s
bm25       14.0%  17.5%  22.0%       3
single     18.5%  24.0%  31.5%     469
maxsim      9.0%  15.5%  24.0%     910
cross      16.0%  21.0%  26.5%     679
GATE: KILL (recall-only)
```

| Ranker | R@50 | R@100 |
| --- | ---: | ---: |
| bm25 | 32.5% | 41.5% |
| single | 41.5% | 57.0% |
| maxsim | 37.0% | 47.5% |
| cross | 42.5% | 55.0% |

## Five sub-goal examples

1. Task: Go to the list of ballet event for the next 30 days.
   - Previous: [textbox]  Search for artists, teams or venues... -> TYPE: Ballet
   - Sub-goal: Click the Search button
   - Gold: [11043] button "submit Search"

2. Task: Get the most popular documentary tv series sorted by IMDB rating, see the details of the top result.
   - Previous: [link]  User Rating -> CLICK
   - Sub-goal: Sort the results by IMDB rating
   - Gold: [21333] a "Going Postal"

3. Task: Buy tickets for a group at Hurricane harbor Phoenix on April 22
   - Previous: [link]  Groups  -> CLICK
   - Sub-goal: Type 50 in the group size box
   - Gold: [40898] a "Buy Now"

4. Task: Find the flight schedule between London and New York for April 7 and check the upgrade details of the noon flight.
   - Previous: [button]  Search -> CLICK
   - Sub-goal: Click the noon flight entry to view upgrade details
   - Gold: [68292] button "button View details"

5. Task: Add a pair of diamond stud earrings to the cart.
   - Previous: [span]  diamond stud earrings -> CLICK
   - Sub-goal: Add to cart button
   - Gold: [11504] img "10k Gold 1 Carat T.W. Black Diamond Stud Earrings 10k Gold 1 Carat T.W. Black Diamond Stud Earrings"

## Cost and checks

Median sub-goal prompt plus generated tokens: 112. Median generation time: 713.32 ms (excludes initial model loading/download). Both are stored per step, including on cache hits.

`test_maxsim.py` passes; all nine `test_eval.py` tests pass. Both result files contain 200 rows with the correct query mode; all sub-goals are nonempty and have positive recorded token costs.

## Changes

- `lens/rank.py:18`: enforce the MPS memory cap; `lens/rank.py:28`: ColBERT query length 32, document length still 64.

- `lens/pick.py:28`: cached sub-goal generation using the shared model, 24 generated tokens maximum, whitespace/quote stripping, prompt-plus-output token counts, generation time.

- `eval.py:38`: three query modes, sub-goal default; `eval.py:59`: query construction, operation-and-tag-only oracle, row costs and five examples; `eval.py:87`: add sub-goal tokens to each ranker arm, leaving baseline cost unchanged; `eval.py:91`: mode-specific output; `eval.py:96`: query-labelled summary and median sub-goal cost.

- `test_eval.py`: cache, token accounting, no-picker sub-goal generation, oracle leakage, original task-query behavior, and existing gate tests.

## Interpretation and limits

Sub-goals improve measured MaxSim R@10 from run 1's 4.5% to 15.5%, but single-vector remains ahead. The new runs also change ColBERT query length from 64 to 32, so the improvement cannot be attributed solely to query wording.

The prescribed oracle is weakly specified for click actions: operation and tag often amount to CLICK button, with no target name. Its poor result should not be interpreted as the upper bound for a correctly specified next-element query.

Sub-goals generated without current page content sometimes invent details (group size 50), repeat a completed action (sorting), or skip ahead (add to cart before selecting the product). The five examples expose these errors.

These are recall-only results on the same seed-0 training-shard sample. No picker accuracy, token-savings gate, or baseline truncation measurements were obtained. The enc ms column times the complete ranking call, not isolated encoding.

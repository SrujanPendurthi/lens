STATUS: complete; Tasks 1–4 finished.

# Brief 8: cleanup and ColPali

## Cleanup and regression

Task 1 (`3ff2cb1`): 92 lines added, 1,378 removed across 14 files. Deleted pagemaxsim, context extraction/caps, the Qwen picker branch, train prep, `--k`, and seven specified stale files. Kept subgoal generation, cache keys, default text-ranker caps and D7 logic.

Task 2 (`aea88fc`): bounding-box prep from decoded or JSON-string attributes. Regenerated 1,339 rows. Coverage: 1,257/1,257 labeled actions. Screenshots: 1,338, preserving JPEG bytes under `.png` filenames. All 200 sampled steps have parseable gold boxes; 199/200 have positive-area gold boxes. Twenty-query regression and 80 text-ranker gold ranks match the first 20 rows of `results_test_200.jsonl` exactly; subgoal cache misses: 0.

Task 3 (`f11e7ce`): native transformers ColPali, `vidore/colpali-v1.3-hf`, bf16; fixed 1,280-px tiles, 32×32 patches, white bottom padding, image-token selection, per-candidate maxima across all overlapping patches/tiles, CPU float32 embedding cache, missing-screenshot metadata, per-tile timing.

Sample: seed-0 shuffle followed by gold-survival filtering; first 200. Screenshot widths: 185 at 1,280 px; 15 wider (1,283–1,410 px). Source images span 859 tiles; median 3, maximum 32. No sample screenshot is missing. No gold box extends past x=1,280.


## Results

200/200 steps completed on MPS vision with CPU text rankers, bf16 ColPali. Subgoal cache misses: 0. All 200 query strings and all 800 existing text-ranker gold ranks exactly match `results_test_200.jsonl`. Watchdog stderr: empty; CPU fallback count: 0. Picker calls: 0.

Verbatim `.venv/bin/python eval.py --summarize results_test_colpali.jsonl` output:

```text

query=subgoal
steps=200  median nodes=438.5  textless gold=2.5%
subgoal median tokens=115  ms=806.75
arm             R@5   R@10   R@20  enc ms
bm25          11.0%  16.0%  20.5%       2
single        20.0%  26.5%  33.5%     594
maxsim        11.5%  18.0%  27.0%     668
cross         18.0%  22.0%  28.0%     905
colpali        0.0%   1.5%   7.0%   14707
GATE: KILL (recall-only)
```

D12 piece 2: colpali R@10 = 1.5%, single R@10 = 26.5%, margin = −25.0 pts → FAIL (required margin ≥ 5.0 pts; required colpali R@10 ≥ 31.5%).

| Ranker | R@5 | R@10 | R@20 | R@50 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 11.0% | 16.0% | 20.5% | 29.0% | 42.5% |
| single | 20.0% | 26.5% | 33.5% | 47.0% | 59.5% |
| maxsim | 11.5% | 18.0% | 27.0% | 40.5% | 50.0% |
| cross | 18.0% | 22.0% | 28.0% | 40.5% | 54.0% |
| colpali | 0.0% | 1.5% | 7.0% | 25.0% | 42.0% |

| ColPali subset | Steps | R@10 hits | R@10 |
| --- | ---: | ---: | ---: |
| textless_gold=True | 5 | 0 | 0.00% |
| textless_gold=False | 195 | 3 | 1.54% |
| CLICK | 158 | 1 | 0.63% |
| TYPE | 30 | 2 | 6.67% |
| SELECT | 12 | 0 | 0.00% |

| Encoding/cache metric | Value |
| --- | ---: |
| Device | MPS |
| `no_screenshot` steps | 0 |
| Total encoded tiles | 859 |
| Median tiles per step | 3.0 |
| Median encode seconds per tile | 3.880164 |
| Median tile-encoding seconds per step | 14.311176 |
| Median total ColPali rank milliseconds per step | 14,707 |
| Screenshot cache hits | 0 |
| Cache files | 200 |
| Cache logical bytes | 450,956,845 (430.07 MiB) |
| Cache allocated disk usage (`du -sk`) | 440,672 KiB (430.34 MiB) |

## Recorded differences

Real model test: Search 12.565985 > Privacy 8.825798. Evaluation ColPali R@5: 0/200; R@10: 3/200; R@20: 14/200. Existing single-vector R@10: 53/200. No sampled screenshots were missing, despite one missing screenshot among all 1,339 prepared actions.

## Validation

Task 1: both new cleanup tests failed before implementation. Task 2: the bounding-box assertion failed with `KeyError: boxes` before implementation. Task 3: importing `lens.vision` failed before implementation.

Final lightweight validation: 36 tests run, 34 passed, 2 heavy-model tests skipped; node extraction and MaxSim checks passed. Heavy validation: 10 tests passed on MPS, including both real-model tests. White-tile output: 1,024 normalized float32 CPU vectors; cache tensors identical. Real text-region scores: Search 12.565985 > Privacy 8.825798. The extra-width test was added during the heavy test run and passed in the final lightweight suite.

## Deviations

- Fifteen sample images exceed the assumed 1,280-px width. Vertical tiles retain source coordinates and crop the extra right margin; no resizing or box scaling. No sampled gold box intersects that margin.
- Launcher stderr is saved to `/tmp/lens-colpali-watchdog.log` so watchdog failures can be reported exactly; evaluation output remains in `data/colpali_test.log`.

## Code changes

- `lens/rank.py:9`: allowed ColPali name; four default rankers retained.
- `lens/nodes.py:19`: iframe-aware extraction retained; context and page-word paths removed.
- `lens/pick.py:90`: Haiku-only picker; Qwen remains only for subgoal generation and existing cache-key compatibility.
- `eval.py:19`: test-task steps by default.
- `eval.py:40`: finite four-float candidate rectangles from JSON or decoded attributes.
- `eval.py:61`: log and retain rows with missing screenshot bytes/path.
- `eval.py:149`: ColPali dispatch with UID, candidate refs and boxes; persist encoding metadata.
- `eval.py:167`: summary signature cleanup; unchanged D7 gate.
- `lens/vision.py:36`: positive-area tile/patch intersections.
- `lens/vision.py:50`: content-based image loading, RGB tiles and white padding.
- `lens/vision.py:65`: lazy native transformers model/processor, bf16 on configured device.
- `lens/vision.py:84`: select patch embeddings by image-token ID.
- `lens/vision.py:93`: normalized CPU patch tensors, per-screenshot disk cache, encoding timings and missing-image stats.
- `lens/vision.py:129`: processor query path.
- `lens/vision.py:139`: source-image clipping and maxima across all element patches and tiles.
- `requirements.txt:8`: Pillow.

## Commits and output

- Task 1: `3ff2cb1`.
- Task 2: `aea88fc`.
- Task 3: `f11e7ce`.
- Task 4: this report/results commit; push target `origin/main`.
- Results: `results_test_colpali.jsonl`, 200 rows. No `data/` files staged.

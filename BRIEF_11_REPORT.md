STATUS: complete; Tasks 1–2 finished; D16 GENERALIZES.

# Brief 11: unseen-website recall check

D16: single_ft − single on test_website = +2 hits → GENERALIZES.

The pre-registered rule classifies a positive single-vector lift as GENERALIZES and a nonpositive lift as MEMORIZATION. MaxSim lift: +16 hits. No additional decision rule was applied.

## Leakage and coverage

| Split | Shards | Actions | Distinct websites | Distinct task descriptions |
| --- | ---: | ---: | ---: | ---: |
| train | 27 | 7,775 | 73 | 1,009 |
| test_website | 4 | 1,019 | 10 | 142 |

Shared websites: **0**. Shared task descriptions: **0**. Checks completed before evaluation. Coverage from the actual run: `Gold extraction coverage: 100.00% (975 steps)`. There are 44 unlabeled actions. The seed-0 shuffle plus gold-survival filter selected the first 200 eligible steps; selected task order was verified against prepared steps.

## Results

200/200 rows completed; subgoal cache misses: 200. Device: CPU for all four rankers. No picker calls or training. Watchdog stderr: empty.

Verbatim `.venv/bin/python eval.py --summarize results_website_ft.jsonl`:

```text
steps=200  median nodes=439.0  textless gold=5.5%
subgoal median tokens=112  ms=775.36
arm             R@5   R@10   R@20  enc ms
single        20.5%  26.0%  30.5%     596
maxsim        14.0%  19.5%  27.0%     660
single_ft     18.0%  27.0%  32.5%     635
maxsim_ft     19.0%  27.5%  33.5%     646
D12 maxsim_ft: R@10 = 55/200, single_ft = 54/200, margin = +0.5 pts → FAIL
```

| Ranker | R@5 | R@10 | R@20 | R@50 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| single | 41/200 (20.5%) | 52/200 (26.0%) | 61/200 (30.5%) | 84/200 (42.0%) | 109/200 (54.5%) |
| maxsim | 28/200 (14.0%) | 39/200 (19.5%) | 54/200 (27.0%) | 70/200 (35.0%) | 86/200 (43.0%) |
| single_ft | 36/200 (18.0%) | 54/200 (27.0%) | 65/200 (32.5%) | 87/200 (43.5%) | 116/200 (58.0%) |
| maxsim_ft | 38/200 (19.0%) | 55/200 (27.5%) | 67/200 (33.5%) | 91/200 (45.5%) | 110/200 (55.0%) |

## Fine-tuning lift

| Split | Model | Zero-shot R@10 | Fine-tuned R@10 | Lift, hits | Lift, pts |
| --- | --- | ---: | ---: | ---: | ---: |
| test_task | single | 53/200 | 60/200 | +7 | +3.5 |
| test_task | maxsim | 36/200 | 60/200 | +24 | +12.0 |
| test_website | single | 52/200 | 54/200 | +2 | +1.0 |
| test_website | maxsim | 39/200 | 55/200 | +16 | +8.0 |

## Cleanup and verification

The new D12 filter test failed before implementation; afterward all 21 eval/train tests, node extraction and MaxSim checks passed. The unchanged fine-tuned checkpoint/cap test passed. Brief 10 recall rows reproduce exactly, with only the maxsim_ft D12 line. Task 1 cleanup `121312c` was pushed after verification.

Cleanup removed the ks alias, shortened checkpoint mapping, used one rank kind, removed redundant fit seeding/invalid-name handling and spent FitTests, recorded the D15 budget caveat, and deleted the two specified spent files.

D15 remained frozen: no training was run; all checkpoint file sizes and nanosecond modification times are unchanged, the training-triplet SHA-256 is unchanged, and triplets() source matches its pre-cleanup version. Old untracked data is retained. Only the new raw website download is removed after verification as instructed; prepared steps, overlap metadata and subgoal cache remain locally.

## Deviations

The first remote two-column check read all train shards but exhausted HTTP 429 retries on the website split. A paced, resumable check read only website/confirmed_task remotely from train and projected the same columns from the already-downloaded website shards locally. No full train download was made for this check. Both overlap counts were zero before model execution.

Watchdog launcher stderr was retained in /tmp/ for error reporting. Temporary metadata scripts and data artifacts were not committed.

## Commits

Task 1: `121312c`. Task 2: this report/results/PLAN/checklist commit, plus a separate `chore: llm cache` commit. Both tasks are pushed to origin/main. Work stops here; the thesis re-grill remains with the planner and user.

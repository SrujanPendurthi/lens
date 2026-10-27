STATUS: complete; Tasks 1–4 finished; D15 FAIL.

# Brief 10: fine-tuned ColBERT versus single-vector

D12 piece 3: maxsim_ft R@10 = 60/200 (30.0%) vs single_ft 60/200 (30.0%), margin +0.0 pts → FAIL. Required margin: ≥ 5.0 pts, or 70/200 MaxSim hits against the observed baseline.

## Train data and leakage check

Multimodal-Mind2Web `train`: 27 parquet shards, 7,775 prepared actions, 1,009 distinct `confirmed_task` strings. Test-task prep: 177 distinct task strings. Train/test task overlap: **0** (checked before subgoal generation). Gold survives extraction in 7,362 train rows.

| Triplet preparation | Count |
| --- | ---: |
| Triplets | 7,163 |
| Skipped: no surviving gold | 413 |
| Skipped: textless gold | 199 |
| Skipped: no textful negative | 0 |
| Empty positive / negative / query strings | 0 / 0 / 0 |

Positive: first textful gold node in node order. Negative: a textful non-gold candidate on the same page, chosen with `random.Random(f"{seed}\n{task}\n{len(prev)}")`, seed 0. Subgoals use the existing Qwen prompt/cache. Both models read the identical `data/triplets_train.jsonl` file. The verified raw download `data/mm_train/` was deleted as instructed; prepared steps and triplets remain locally.

## Training

One training run per model; no tuning, dev evaluation, early stopping, or repeated training. Shared settings: seed/data seed 0, 1 epoch, learning rate 2e-5, batch 32, warmup ratio 0.1, CPU, no intermediate checkpoints. There were 224 optimizer steps, including 23 warmup steps. Epoch logging was used without an external reporter.

| Model | Loss | Length caps | Final train loss | Train seconds |
| --- | --- | --- | ---: | ---: |
| single | MultipleNegativesRankingLoss | max_seq_length=128 | 3.087001528058733 | 197.3 |
| maxsim | pylate Contrastive + ColBERTCollator | query_length=32, document_length=64 | 3.2462501525878906 | 251.9 |

Checkpoints: `models/single_ft` and `models/maxsim_ft` (ignored by git). Both passed a separate CPU load/cap check through the shared lock/watchdog (exit 0). Training logs show epoch 1 and 224/224 steps with final losses; watchdog stderr is empty. No training was repeated after the server restart.

## Evaluation

Same 200 seed-0, gold-surviving test-task steps as `results_test_200.jsonl`. Completed rows: 200. Subgoal cache misses: 0. All 200 queries and all 400 zero-shot single/MaxSim gold ranks match the reference exactly. Zero-shot R@10: single 53/200, maxsim 36/200. Rankers run on CPU; encoding times are CPU measurements. No picker calls. Watchdog stderr is empty.

Verbatim `.venv/bin/python eval.py --summarize results_test_ft.jsonl`:

```text
steps=200  median nodes=438.5  textless gold=2.5%
subgoal median tokens=115  ms=806.75
arm             R@5   R@10   R@20  enc ms
single        20.0%  26.5%  33.5%     496
maxsim        11.5%  18.0%  27.0%     541
single_ft     21.0%  30.0%  38.0%     498
maxsim_ft     22.5%  30.0%  37.5%     550
D12 single: R@10 = 53/200, single_ft = 60/200, margin = -3.5 pts → FAIL
D12 maxsim: R@10 = 36/200, single_ft = 60/200, margin = -12.0 pts → FAIL
D12 maxsim_ft: R@10 = 60/200, single_ft = 60/200, margin = +0.0 pts → FAIL
```

| Ranker | R@5 | R@10 | R@20 | R@50 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| single | 40/200 (20.0%) | 53/200 (26.5%) | 67/200 (33.5%) | 94/200 (47.0%) | 119/200 (59.5%) |
| maxsim | 23/200 (11.5%) | 36/200 (18.0%) | 54/200 (27.0%) | 81/200 (40.5%) | 100/200 (50.0%) |
| single_ft | 42/200 (21.0%) | 60/200 (30.0%) | 76/200 (38.0%) | 105/200 (52.5%) | 128/200 (64.0%) |
| maxsim_ft | 45/200 (22.5%) | 60/200 (30.0%) | 75/200 (37.5%) | 103/200 (51.5%) | 135/200 (67.5%) |

| Fine-tuning lift at R@10 | Extra hits | Percentage points |
| --- | ---: | ---: |
| single_ft − single | +7 | +3.5 |
| maxsim_ft − maxsim | +24 | +12.0 |

| maxsim_ft operation | Steps | R@10 hits | R@10 |
| --- | ---: | ---: | ---: |
| CLICK | 158 | 47 | 29.75% |
| TYPE | 30 | 8 | 26.67% |
| SELECT | 12 | 5 | 41.67% |

## Code and verification

- Cleanup removed screenshot prep/UID output, query-mode metadata and summary arguments, duplicate survival bookkeeping, obsolete guards/tests and the finished brief 9 plan. `ALLOWED_RANKERS` remains and now permits the two `_ft` names.
- Task 1 eval/test lines: 191 + 282 = 473 before, 167 + 230 = 397 after; reduction 76. The brief 7 four-ranker R@5/10/20 rows reproduced exactly. The moved Haiku cache-key test passed.
- `train.py` implements triplets and the two registered losses/configurations. It refuses to fit into an existing output directory.
- `run_evals.py` accepts an explicit script and retains the shared lock and swap watchdog.
- `lens/rank.py` loads local `_ft` checkpoints with the same caps as their zero-shot twins; default rankers are unchanged.
- `summarize` compares D12 margins against `single_ft` when present, using integer hit counts at the 5.0-point boundary.
- TDD: the triplet test initially failed with missing `train`; cap and baseline-switch tests failed before implementation; the mocked training-configuration test failed with missing `fit` before implementation. All 21 tests, node extraction and MaxSim checks passed before training. Mocked trainer tests did not optimize any weights.
- Installed versions: torch 2.11.0, sentence-transformers 5.3.0, pylate 1.6.0, datasets 5.1.0, accelerate 1.15.0, transformers 5.3.0.

## Deviations

- The 27 train shards are under `data/train-*`; the plan's root `train-*` pattern matches none. Download used the actual `data/train-*` paths.
- sentence-transformers 5.3.0 on transformers 5.3.0 accepts the specified `warmup_ratio=0.1`, then internally moves it to `warmup_steps=0.1`. The configuration test checks the effective 10% ratio and 23/224 warmup steps; no hyperparameter changed.
- Watchdog launcher stderr was saved under `/tmp/` for exact error reporting.
- The server restarted after the ColBERT training finished and interrupted verification. Saved logs/checkpoints were inspected and loaded through the wrapper; neither training run was repeated.

## Commits and artifacts

- Task 1 cleanup: `d6edbd0`; separate cache: `8c15dab`; pushed after Task 1.
- Task 2 triplets/generic runner: `cb59d9d`; separate generated cache: `1241272`.
- Task 3 fine-tuning and rankers: `0434128`.
- Task 4: this report, `results_test_ft.jsonl`, D15 outcome, and completed checklist. The final evaluation added no cache entries, so no additional cache commit was needed.
- Only `data/llm_cache.jsonl` was staged under `data/`; raw data, prepared data and models were not staged.

Work stops after Task 4 as instructed. The next decision belongs to the planner.

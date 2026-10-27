STATUS: complete; Tasks 1–4 finished.

# Brief 7: test split and whole-page contextual MaxSim

Dataset: `osunlp/Multimodal-Mind2Web`, official `test_task`, five parquet shards. Prepared rows: 1,339. Actions with no positive candidates: 82. Labeled actions: 1,257. Gold extraction coverage: 1,257/1,257 = 100.00% (1,257/1,339 = 93.88% of all actions). Screenshots saved: 1,338. Missing screenshot UID: `99e6a37a-fad2-449a-a359-8aa93687cb5b`. `du -sk data/mm_test_task`: 1,266,300 KiB (1.21 GiB allocated).

Dry run: `Gold extraction coverage: 100.00% (1257 steps)`; one completed row; one subgoal cache miss; `GATE: KILL (recall-only)`.

Sample: seed-0 shuffle of all prepared actions, then first 200 with gold surviving extraction. Rankers use own-text inputs; pagemaxsim uses page words. Fixed values: 64 words/window, 32-word stride, 128 document tokens, 32 query tokens. CPU rankers; encoding times are CPU measurements. No picker calls.


## Results

200/200 rows completed. Subgoal cache misses: 199 (dry run populated one). Cache-only replay of all 200 selected queries: 0 misses; all query texts matched saved results. No model load was allowed during that replay.

Verbatim `.venv/bin/python eval.py --summarize results_test_200.jsonl` output:

```text

query=subgoal
node_text=own
steps=200  median nodes=438.5  textless gold=2.5%
subgoal median tokens=115  ms=806.75
arm             R@5   R@10   R@20  enc ms
bm25          11.0%  16.0%  20.5%       3
single        20.0%  26.5%  33.5%     558
maxsim        11.5%  18.0%  27.0%     570
cross         18.0%  22.0%  28.0%     826
pagemaxsim    13.5%  17.5%  23.0%     388
GATE: KILL (recall-only)
```

| Ranker | R@5 | R@10 | R@20 | R@50 | R@100 | Median CPU ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 11.0% | 16.0% | 20.5% | 29.0% | 42.5% | 2.84 |
| single | 20.0% | 26.5% | 33.5% | 47.0% | 59.5% | 557.87 |
| maxsim | 11.5% | 18.0% | 27.0% | 40.5% | 50.0% | 570.10 |
| cross | 18.0% | 22.0% | 28.0% | 40.5% | 54.0% | 825.85 |
| pagemaxsim | 13.5% | 17.5% | 23.0% | 35.0% | 54.5% | 387.85 |

D12 piece 1: pagemaxsim R@10 = 17.5%, single R@10 = 26.5%, margin = −9.0 pts → FAIL (required margin ≥ 5.0 pts; required pagemaxsim R@10 ≥ 31.5%).

Pagemaxsim median CPU time: 387.85 ms/step. Median nodes: 438.5. Median page words: 1,534.5; maximum: 18,007.

| Gold text definition | Textless steps | Textless R@10 | Steps with text | With-text R@10 |
| --- | ---: | ---: | ---: | ---: |
| Existing extracted own-text (`textless_gold`) | 5 | 0/5 = 0.0% | 195 | 35/195 = 17.95% |
| Assigned page words (`page_textless_gold`) | 16 | 0/16 = 0.0% | 184 | 35/184 = 19.02% |

## Comparison with brief 2

Brief 2 used the train shard, own-text nodes and subgoal queries on 200 steps. This run uses the official test-task split, own-text nodes for the four existing rankers, and subgoal queries on 200 steps. Brief 2 median nodes: 457.0; this run: 438.5. Brief 2 textless gold: 3.5%; this run: 2.5%. Brief 2 ranker timing used MPS; this run uses CPU.

| Ranker | Brief 2 R@10 | Test R@10 | Change, pts |
| --- | ---: | ---: | ---: |
| bm25 | 17.5% | 16.0% | −1.5 |
| single | 24.0% | 26.5% | +2.5 |
| maxsim | 15.5% | 18.0% | +2.5 |
| cross | 21.0% | 22.0% | +1.0 |

## Code and validation

- `lens/nodes.py:20`: parse dataset-expanded iframe children as DOM elements, preserving iframe tags.
- `lens/nodes.py:52`: document-order page words, candidate attributes, innermost candidate ownership, unowned context.
- `eval.py:33`: parquet prep, JSON field/entry parsing, all action rows, screenshot saving and missing-image handling.
- `eval.py:107`: coverage over labeled actions; filter gold-surviving actions after seeded shuffle.
- `lens/rank.py:9`: pagemaxsim is allowed but absent from the default ranker tuple.
- `lens/rank.py:52`: fixed-length pre-skiplist normalized token embeddings with word IDs and explicit document marker alignment.
- `lens/rank.py:79`: per-query-token maxima over each candidate's tokens across all overlapping windows; tokenless candidates score negative infinity.
- `eval.py:134`: page-textless gold metadata; page-ranker dispatch and summary inclusion follow.
- `requirements.txt:7`: pyarrow.

TDD: Task 1 failed with missing `page_words`; Task 2 failed with missing `prep_mm`; Task 3 failed with missing `encode_doc_tokens`/`rank_page`. Final validation: 32 unittest tests passed, `test_nodes.py` passed, `test_maxsim.py` passed. Real model tests ran on CPU. D7 gate code, existing ranker caps, cache keys, and default ranker tuple were unchanged.

## Deviations

- The inspected parquet row stores `operation` as a JSON string, candidate collections as lists of individual JSON strings, and `target_action_index` as a string. Prep supports decoded objects as well.
- One screenshot is null. Its action remains prepared; no PNG is fabricated. All 1,338 saved images have JPEG signatures (`ffd8ffe000104a46`); original bytes are preserved under the plan's `.png` filenames.
- lxml's HTML parser treated expanded iframe children as raw text. The iframe parsing regression fix raised all-action gold coverage from 92.83% to 93.88%; all remaining 82 actions have empty positive candidates. Coverage uses labeled actions, with all-action counts reported separately.
- The plan's real-checkpoint toy relevance assertion failed: for `click the search button` on `Search button Privacy policy`, Search scored 21.854803 and Privacy scored 24.218493. Aligned document vectors matched pylate's standard pre-filter output within 1.8e-7. The relevance ordering test uses controlled vectors; a real-checkpoint equivalence test covers subwords and punctuation. The scoring algorithm and fixed values were unchanged.
- Sandboxed Hugging Face metadata requests failed with `[Errno 8] nodename nor servname provided, or not known`. The required tests were rerun with approved network access and passed.
- The first nohup launch created no log/results. Relaunching with a two-second initialization wait created both.

## Commits

- Task 1: `b033259`.
- Task 2: `08a9b3e`.
- Task 3: `488719e`.
- Task 4: this report/results commit. Push target: `origin/main`.

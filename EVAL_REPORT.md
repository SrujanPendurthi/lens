# TokenLens v1 evaluation — 2026-10-23

Verdict: **KILL at the recall-only gate (D7)**. MaxSim recall@10 is
4.5% (9/200), below single-vector recall@10 of 14.0% (28/200).
The plan explicitly permits this early kill before spending time on the picker.
The current text-only setup does not justify proceeding to PLAID, quantization,
the asynchronous protocol, or RL.

## Evidence

Opus's existing `eval.py --n 200 --no-llm` process finished during handover.
Its per-step outputs are in `results.jsonl`; the original console output is
in `data/recall.log`. No replacement model evaluation was needed.

| Ranker | Recall@5 | Recall@10 | Recall@20 | Median ranking ms/page |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 8.5% | 13.0% | 19.0% | 2 |
| BGE small (single vector) | 10.0% | 14.0% | 19.5% | 410 |
| AnswerAI ColBERT (MaxSim) | 1.5% | 4.5% | 9.0% | 946 |
| MiniLM cross-encoder | 10.5% | 16.5% | 24.5% | 812 |

The deterministic seed-0 sample contains 200 steps across 83 distinct task
descriptions from the downloaded Mind2Web training shard, with a median of
457 candidate nodes per step. This is a sample from training data, not an
official test-split result. Seven sampled steps (3.5%) have textless gold nodes.

Gold extraction coverage is 741/743 prepared steps (99.73%), above the 98%
requirement. All 200 sampled steps retain at least one gold candidate. There
are no duplicate candidate refs in the prepared data.

## Checks completed

- `test_maxsim.py`: exact MaxSim arithmetic and planted-node selection pass.
- Verified the loaded ColBERT projection equals the checkpoint's trained
  `linear.weight` tensor exactly (96 × 384). The base encoder's unused-weight
  warning does not mean PyLate discards the projection.
- Verified query/document prefixes `[unused0]` and `[unused1]`, repeatable
  ranking, and a small semantic sanity example using cached model weights,
  the shared GPU lock, a memory cap, and a swap watchdog.
- Recomputed the summary directly from all 200 saved result rows.
- Four evaluator regression tests pass: recall alone cannot PASS, a recall
  tie with single-vector kills, the recall gate always uses 10, and a full
  PASS requires a top-10 picker.

## Limits and remaining work

No picker run, prefill curve, accuracy comparison, prompt-token reduction,
or full end-to-end smoke was performed; the early kill makes those unnecessary
for this decision. `lens/pick.py` remains unvalidated end to end, including its
handling of gold elements removed by baseline truncation.

The timing column measures the complete ranking call (including encoding and
scoring, and first-call model initialization), not isolated encoder time.
The run uses the existing 64-token ColBERT query/document limits and text-only
candidate representations. The weak cross-encoder result also warrants
examining query construction and candidate text if this experiment is reopened.
This result applies to the measured setup, not all late-interaction methods.

## Harness changes after the run

`eval.py` now checks prepared gold coverage, accepts `--output`, treats missing
gold candidates as recall misses, validates positive sample sizes, and avoids
claiming PASS without all D7 evidence. D7 recall always uses k=10. These changes
do not alter the saved 200-step ranking results or their KILL verdict.

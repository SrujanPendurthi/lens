"""Mind2Web offline eval: baseline (all candidate nodes) vs top-k per ranker. Prints D12 recall margins against single-vector.

usage: python eval.py --prep-mm data/mm_test_task
       python eval.py --n 200 [--no-llm]
"""
import argparse
import json
from pathlib import Path
import random
import statistics as st
import time

from lens.nodes import Node, extract
from lens.rank import ALLOWED_RANKERS, RANKERS, rank

STEPS = "data/steps_test.jsonl"


def prep_mm(parquet_dir: str, steps_path: str) -> None:
    import pyarrow.parquet as pq

    def parsed(value):
        return json.loads(value) if isinstance(value, str) else value

    with open(steps_path, "w") as out:
        for path in sorted(Path(parquet_dir).rglob("*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
                for action in batch.to_pylist():
                    pos = parsed(action["pos_candidates"])
                    neg = parsed(action["neg_candidates"])
                    # Some shards store each candidate as its own JSON string.
                    pos, neg = [parsed(c) for c in pos], [parsed(c) for c in neg]
                    gold = sorted({str(c["backend_node_id"]) for c in pos})
                    refs = list(dict.fromkeys(str(c["backend_node_id"]) for c in pos + neg))
                    nodes = extract(action["cleaned_html"], refs)
                    row = {"task": action["confirmed_task"],
                           "prev": action["action_reprs"][:int(action["target_action_index"])][-3:],
                           "op": parsed(action["operation"]), "gold": gold,
                           "nodes": [n.__dict__ for n in nodes]}
                    out.write(json.dumps(row) + "\n")


def has_gold(step):
    return bool(set(step["gold"]) & {n["ref"] for n in step["nodes"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-mm")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ks", type=lambda s: sorted({int(k) for k in s.split(",")}), default=[10], help="comma-separated picker cutoffs")
    ap.add_argument("--rankers", default=",".join(RANKERS))
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--output")
    ap.add_argument("--summarize")
    ap.add_argument("--steps", default=STEPS)
    a = ap.parse_args()
    if a.summarize:
        with open(a.summarize) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if not rows:
            ap.error("no result rows to summarize")
        return summarize(rows, "baseline" in rows[0])
    if a.prep_mm:
        return prep_mm(a.prep_mm, a.steps)
    rankers = list(dict.fromkeys(a.rankers.split(",")))
    if not rankers or any(name not in ALLOWED_RANKERS for name in rankers):
        ap.error("--rankers must contain names from " + ",".join(ALLOWED_RANKERS))
    with open(a.steps) as f:
        steps = [json.loads(l) for l in f]
    if not steps:
        ap.error("no prepared steps; run --prep-mm first")
    labeled = [s for s in steps if s["gold"]]
    if not labeled:
        ap.error("no labeled steps to evaluate")
    print(f"Prepared actions: {len(steps)}; unlabeled: {len(steps) - len(labeled)}", flush=True)
    coverage = sum(has_gold(s) for s in labeled) / len(labeled)
    print(f"Gold extraction coverage: {coverage:.2%} ({len(labeled)} steps)", flush=True)
    if coverage < 0.98:
        ap.error("gold extraction coverage is below 98%; fix extraction before evaluating")
    random.Random(0).shuffle(steps)  # one shard holds whole tasks; shuffle so n steps span many sites
    steps = [s for s in steps if has_gold(s)]
    rows = []
    misses = 0
    with open(a.output or "results_subgoal.jsonl", "w") as f:
        for s in steps[:a.n]:
            nodes, gold = [Node(**n) for n in s["nodes"]], set(s["gold"])
            # random.shuffle gives baseline a neutral order; Mind2Web lists pos_candidates first
            random.Random(s["task"]).shuffle(nodes)
            from lens.pick import subgoal
            goal = subgoal(s["task"], s["prev"])
            misses += bool(goal.get("miss"))
            query = goal["text"]
            row = {"task": s["task"], "op": s["op"]["op"], "query": query,
                   "subgoal_tokens": goal["tokens"], "subgoal_ms": goal["ms"],
                   "n_nodes": len(nodes), "textless_gold": not any(n.text for n in nodes if n.ref in gold)}
            def pick_nodes(shown_nodes, cost):
                from lens.pick import pick, prompt
                p = pick(prompt(s["task"], s["prev"], shown_nodes))
                refs = {n.ref for n in shown_nodes}
                return p | {"ok": p["ref"] in gold and p["ref"] in refs,
                            "tokens": p["prompt_tokens"] + cost, "shown": len(refs),
                            "gold_shown": bool(gold & refs),
                            "hallucinated": p["ref"] not in (None, "none") and p["ref"] not in refs}

            if not a.no_llm:
                row["baseline"] = pick_nodes(nodes, 0)
            texts = [n.text for n in nodes]
            for name in rankers:
                t0 = time.perf_counter()
                scores = rank(name, query, texts)
                order = scores.argsort(descending=True).tolist()
                row[name] = {"ms": (time.perf_counter() - t0) * 1000,
                             "gold_rank": min((order.index(i) for i, n in enumerate(nodes) if n.ref in gold), default=2**63 - 1)}
                if not a.no_llm:
                    for k in a.ks:
                        row[f"{name}@{k}"] = pick_nodes([nodes[i] for i in order[:k]], goal["tokens"])
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(len(rows), {k: row[k].get("gold_rank", row[k].get("ok")) for k in ("baseline",) + ALLOWED_RANKERS if k in row}, flush=True)
    print(f"subgoal cache misses: {misses}")
    summarize(rows, not a.no_llm)


def summarize(rows, llm):
    n = len(rows)
    rankers = [name for name in ALLOWED_RANKERS if all(name in r for r in rows)]
    rec = lambda name, kk: sum(r[name]["gold_rank"] < kk for r in rows) / n
    print(f"steps={n}  median nodes={st.median(r['n_nodes'] for r in rows)}  textless gold={sum(r['textless_gold'] for r in rows) / n:.1%}")
    if all("subgoal_tokens" in r for r in rows):
        print(f"subgoal median tokens={st.median(r['subgoal_tokens'] for r in rows):.0f}  ms={st.median(r['subgoal_ms'] for r in rows):.2f}")
    print(f"{'arm':12}{'R@5':>7}{'R@10':>7}{'R@20':>7}{'enc ms':>8}")
    for name in rankers:
        print(f"{name:12}" + "".join(f"{rec(name, kk):7.1%}" for kk in (5, 10, 20)) +
              f"{st.median(r[name]['ms'] for r in rows):8.0f}")
    if llm:
        pick_arms = [name for name in rows[0] if name == "baseline" or "@" in name]
        pick_arms = [name for name in pick_arms if all(name in r for r in rows)]
        print(f"{'pick arm':12}{'acc':>8}{'acc|shown':>11}{'halluc':>9}{'tok med':>10}{'API s':>9}{'CLICK':>9}{'TYPE':>9}{'SELECT':>9}")
        for name in pick_arms:
            picks = [r[name] for r in rows]
            shown = [p for p in picks if p["gold_shown"]]
            conditional = f"{sum(p['ok'] for p in shown)/len(shown):.1%}" if shown else "n/a"
            line = (f"{name:12}{sum(p['ok'] for p in picks)/n:8.1%}{conditional:>11}"
                    f"{sum(p['hallucinated'] for p in picks)/n:9.1%}"
                    f"{st.median(p['tokens'] for p in picks):10.0f}{st.median(p['ttft'] for p in picks):9.2f}")
            for op in ("CLICK", "TYPE", "SELECT"):
                subset = [r[name] for r in rows if r.get("op") == op]
                value = f"{sum(p['ok'] for p in subset)/len(subset):.1%}" if subset else "n/a"
                line += f"{value:>9}"
            print(line)

    base = "single_ft" if "single_ft" in rankers else "single"
    if base in rankers:
        single_hits = sum(r[base]["gold_rank"] < 10 for r in rows)
        for name in rankers:
            if name == base or (base == "single_ft" and not name.endswith("_ft")):
                continue
            hits = sum(r[name]["gold_rank"] < 10 for r in rows)
            margin = (hits - single_hits) * 100 / n
            verdict = "PASS" if (hits - single_hits) * 100 >= 5 * n else "FAIL"
            print(f"D12 {name}: R@10 = {hits}/{n}, {base} = {single_hits}/{n}, margin = {margin:+.1f} pts → {verdict}")


if __name__ == "__main__":
    main()

"""Mind2Web offline eval: baseline (all candidate nodes) vs top-k per ranker. Prints PASS/KILL per plan gate D7.

usage: python eval.py --prep data/train_0.json   # build data/steps.jsonl once
       python eval.py --n 200 [--k 10] [--no-llm]
"""
import argparse
import json
import random
import statistics as st
import time

from lens.nodes import Node, extract
from lens.rank import RANKERS, rank

STEPS = "data/steps.jsonl"


def prep(path, steps_path=STEPS):
    with open(steps_path, "w") as out:
        for t in json.load(open(path)):
            for i, a in enumerate(t["actions"]):
                gold = sorted({c["backend_node_id"] for c in a["pos_candidates"]})
                if not gold:
                    continue
                refs = [c["backend_node_id"] for c in a["pos_candidates"] + a["neg_candidates"]]
                nodes = extract(a["cleaned_html"], refs)
                out.write(json.dumps({"task": t["confirmed_task"], "prev": t["action_reprs"][max(0, i - 3):i],
                                      "op": a["operation"], "gold": gold, "nodes": [n.__dict__ for n in nodes]}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--output")
    ap.add_argument("--summarize")
    ap.add_argument("--steps", default=STEPS)
    ap.add_argument("--node-text", choices=("own", "context"), default="own")
    a = ap.parse_args()
    if a.summarize:
        with open(a.summarize) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if not rows:
            ap.error("no result rows to summarize")
        return summarize(rows, a.k, "baseline" in rows[0], "subgoal", rows[0].get("node_text", "own"))
    if a.prep:
        return prep(a.prep, a.steps)
    if a.n < 1 or a.k < 1:
        ap.error("--n and --k must be positive")

    with open(a.steps) as f:
        steps = [json.loads(l) for l in f]
    if not steps:
        ap.error("no prepared steps; run --prep first")
    coverage = sum(bool(set(s["gold"]) & {n["ref"] for n in s["nodes"]}) for s in steps) / len(steps)
    print(f"Gold extraction coverage: {coverage:.2%} ({len(steps)} steps)", flush=True)
    if coverage < 0.98:
        ap.error("gold extraction coverage is below 98%; fix extraction before evaluating")
    random.Random(0).shuffle(steps)  # one shard holds whole tasks; shuffle so n steps span many sites
    rows = []
    misses = 0
    suffix = "_ctx" if a.node_text == "context" else ""
    with open(a.output or f"results_subgoal{suffix}.jsonl", "w") as f:
        for s in steps[:a.n]:
            nodes, gold = [Node(**n) for n in s["nodes"]], set(s["gold"])
            # random.shuffle gives baseline a neutral order; Mind2Web lists pos_candidates first
            random.Random(s["task"]).shuffle(nodes)
            from lens.pick import subgoal
            goal = subgoal(s["task"], s["prev"])
            misses += bool(goal.get("miss"))
            query = goal["text"]
            gold_node = next((n for n in nodes if n.ref in gold), None)
            row = {"task": s["task"], "op": s["op"]["op"], "query_mode": "subgoal", "node_text": a.node_text, "query": query,
                   "subgoal_tokens": goal["tokens"], "subgoal_ms": goal["ms"],
                   "n_nodes": len(nodes), "textless_gold": not any(n.text for n in nodes if n.ref in gold)}
            if len(rows) < 5:
                example = {"task": s["task"], "prev": s["prev"][-1] if s["prev"] else "none",
                           "subgoal": query, "gold": gold_node.line() if gold_node else "missing"}
                row["example"] = example
                print("EXAMPLE:", json.dumps(example), flush=True)
            if not a.no_llm:
                from lens.pick import pick, prompt
                p = pick(prompt(s["task"], s["prev"], nodes))
                visible = p.get("visible_refs")
                row["baseline"] = p | {"ok": p["ref"] in gold and (visible is None or p["ref"] in visible), "tokens": p["prompt_tokens"]}
            for name in RANKERS:
                t0 = time.perf_counter()
                texts = [f"{n.text} | {n.context}" if a.node_text == "context" else n.text for n in nodes]
                order = rank(name, query, texts, node_text=a.node_text).argsort(descending=True).tolist()
                r = {"ms": (time.perf_counter() - t0) * 1000,
                     "gold_rank": min((order.index(i) for i, n in enumerate(nodes) if n.ref in gold), default=2**63 - 1)}
                if not a.no_llm:
                    p = pick(prompt(s["task"], s["prev"], [nodes[i] for i in order[:a.k]]))
                    r |= p | {"ok": p["ref"] in gold, "tokens": p["prompt_tokens"] + goal["tokens"]}
                row[name] = r
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(len(rows), {k: row[k].get("gold_rank", row[k].get("ok")) for k in ("baseline",) + RANKERS if k in row}, flush=True)
    print(f"subgoal cache misses: {misses}")
    summarize(rows, a.k, not a.no_llm, "subgoal", a.node_text)


def summarize(rows, k, llm, query="task", node_text="own"):
    n = len(rows)
    rec = lambda name, kk: sum(r[name]["gold_rank"] < kk for r in rows) / n
    print(f"\nquery={query}")
    print(f"node_text={node_text}")
    print(f"steps={n}  median nodes={st.median(r['n_nodes'] for r in rows)}  textless gold={sum(r['textless_gold'] for r in rows) / n:.1%}")
    if query == "subgoal":
        print(f"subgoal median tokens={st.median(r['subgoal_tokens'] for r in rows):.0f}  ms={st.median(r['subgoal_ms'] for r in rows):.2f}")
    print(f"{'arm':9}{'R@5':>7}{'R@10':>7}{'R@20':>7}{'enc ms':>8}{'acc':>7}{'tok med':>9}{'ttft s':>8}")
    arms = (["baseline"] if llm else []) + list(RANKERS)
    for name in arms:
        line = f"{name:9}"
        line += "".join(f"{rec(name, kk):7.1%}" for kk in (5, 10, 20)) if name != "baseline" else " " * 21
        line += f"{st.median(r[name]['ms'] for r in rows):8.0f}" if name != "baseline" else " " * 8
        if llm:
            line += f"{sum(r[name]['ok'] for r in rows) / n:7.1%}{st.median(r[name]['tokens'] for r in rows):9.0f}{st.median(r[name]['ttft'] for r in rows):8.2f}"
        print(line)
    if llm:
        print(f"baseline truncated: {sum(r['baseline']['truncated'] for r in rows) / n:.1%}")

    # gate D7
    r_max, r_single = rec("maxsim", 10), rec("single", 10)
    kill = r_max <= r_single
    passed = r_max >= 0.85
    if llm:
        acc = lambda name: sum(r[name]["ok"] for r in rows) / n
        tok = lambda name: st.median(r[name]["tokens"] for r in rows)
        kill |= acc("maxsim") < acc("baseline") - 0.05
        passed &= acc("maxsim") >= acc("baseline") - 0.02 and tok("maxsim") <= 0.2 * tok("baseline")
    # D7 always uses recall@10 and a top-10 picker. Recall alone cannot prove PASS.
    verdict = "KILL" if kill else "PASS" if passed and llm and k == 10 else "INCOMPLETE" if not llm or k != 10 else "NEITHER"
    print("GATE:", verdict, "(recall-only)" if not llm else "")


if __name__ == "__main__":
    main()

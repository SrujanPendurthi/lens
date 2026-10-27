"""Mind2Web offline eval: baseline (all candidate nodes) vs top-k per ranker. Prints PASS/KILL per plan gate D7.

usage: python eval.py --prep data/train_0.json   # build data/steps.jsonl once
       python eval.py --n 200 [--k 10] [--no-llm]
"""
import argparse
import json
import random
import statistics as st
import time

from pathlib import Path

from lens.nodes import Node, extract, page_words
from lens.rank import ALLOWED_RANKERS, RANKERS, rank, rank_page

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


def prep_mm(parquet_dir: str, steps_path: str, shots_dir: str = "data/shots") -> None:
    import pyarrow.parquet as pq

    def parsed(value):
        return json.loads(value) if isinstance(value, str) else value

    shots = Path(shots_dir)
    shots.mkdir(parents=True, exist_ok=True)
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
                    uid = action["action_uid"]
                    shot = shots / f"{uid}.png"
                    if not shot.exists() and action["screenshot"] is not None:
                        screenshot = action["screenshot"]
                        data = screenshot.get("bytes")
                        shot.write_bytes(data if data is not None else Path(screenshot["path"]).read_bytes())
                    row = {"task": action["confirmed_task"],
                           "prev": action["action_reprs"][:int(action["target_action_index"])][-3:],
                           "op": parsed(action["operation"]), "gold": gold,
                           "nodes": [n.__dict__ for n in nodes], "uid": uid,
                           "page": page_words(action["cleaned_html"], refs)}
                    out.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep")
    ap.add_argument("--prep-mm")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=10, help="summary compatibility; D7 always uses top 10")
    ap.add_argument("--ks", default="10", help="comma-separated picker cutoffs")
    ap.add_argument("--rankers", default=",".join(RANKERS))
    ap.add_argument("--picker", choices=("qwen", "haiku"), default="qwen")
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
        return summarize(rows, a.k, "baseline" in rows[0], rows[0].get("query_mode", "unknown"), rows[0].get("node_text", "own"))
    if a.prep_mm:
        return prep_mm(a.prep_mm, a.steps)
    if a.prep:
        return prep(a.prep, a.steps)
    try:
        ks = list(dict.fromkeys(int(k) for k in a.ks.split(",")))
    except ValueError:
        ap.error("--ks must be comma-separated positive integers")
    rankers = list(dict.fromkeys(a.rankers.split(",")))
    if not rankers or any(name not in ALLOWED_RANKERS for name in rankers):
        ap.error("--rankers must contain names from " + ",".join(ALLOWED_RANKERS))
    if not ks or min(ks) < 1:
        ap.error("--ks must be positive")
    if a.n < 1 or a.k < 1:
        ap.error("--n and --k must be positive")

    with open(a.steps) as f:
        steps = [json.loads(l) for l in f]
    if not steps:
        ap.error("no prepared steps; run --prep first")
    labeled = [s for s in steps if s["gold"]]
    if not labeled:
        ap.error("no labeled steps to evaluate")
    print(f"Prepared actions: {len(steps)}; unlabeled: {len(steps) - len(labeled)}", flush=True)
    coverage = sum(bool(set(s["gold"]) & {n["ref"] for n in s["nodes"]}) for s in labeled) / len(labeled)
    print(f"Gold extraction coverage: {coverage:.2%} ({len(labeled)} steps)", flush=True)
    if coverage < 0.98:
        ap.error("gold extraction coverage is below 98%; fix extraction before evaluating")
    random.Random(0).shuffle(steps)  # one shard holds whole tasks; shuffle so n steps span many sites
    steps = [s for s in steps if set(s["gold"]) & {n["ref"] for n in s["nodes"]}]
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
            row = {"task": s["task"], "op": s["op"]["op"], "query_mode": "subgoal", "node_text": a.node_text, "picker": a.picker, "query": query,
                   "subgoal_tokens": goal["tokens"], "subgoal_ms": goal["ms"],
                   "n_nodes": len(nodes), "textless_gold": not any(n.text for n in nodes if n.ref in gold)}
            if "pagemaxsim" in rankers:
                page_refs = {ref for _, ref in s["page"] if ref is not None}
                row["page_textless_gold"] = not bool(gold & page_refs)
            if len(rows) < 5:
                example = {"task": s["task"], "prev": s["prev"][-1] if s["prev"] else "none",
                           "subgoal": query, "gold": gold_node.line() if gold_node else "missing"}
                row["example"] = example
                print("EXAMPLE:", json.dumps(example), flush=True)
            def pick_nodes(shown_nodes, cost):
                from lens.pick import pick, prompt
                p = pick(prompt(s["task"], s["prev"], shown_nodes), a.picker)
                refs = {n.ref for n in shown_nodes}
                if p.get("visible_refs") is not None:
                    refs &= set(p["visible_refs"])
                return p | {"ok": p["ref"] in gold and p["ref"] in refs,
                            "tokens": p["prompt_tokens"] + cost, "shown": len(refs),
                            "gold_shown": bool(gold & refs),
                            "hallucinated": p["ref"] not in (None, "none") and p["ref"] not in refs}

            if not a.no_llm:
                row["baseline"] = pick_nodes(nodes, 0)
            texts = [f"{n.text} | {n.context}" if a.node_text == "context" else n.text for n in nodes]
            for name in rankers:
                t0 = time.perf_counter()
                scores = (rank_page(query, s["page"], [n.ref for n in nodes]) if name == "pagemaxsim"
                          else rank(name, query, texts, node_text=a.node_text))
                order = scores.argsort(descending=True).tolist()
                row[name] = {"ms": (time.perf_counter() - t0) * 1000,
                             "gold_rank": min((order.index(i) for i, n in enumerate(nodes) if n.ref in gold), default=2**63 - 1)}
                if not a.no_llm:
                    for k in ks:
                        row[f"{name}@{k}"] = pick_nodes([nodes[i] for i in order[:k]], goal["tokens"])
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(len(rows), {k: row[k].get("gold_rank", row[k].get("ok")) for k in ("baseline",) + ALLOWED_RANKERS if k in row}, flush=True)
    print(f"subgoal cache misses: {misses}")
    summarize(rows, a.k, not a.no_llm, "subgoal", a.node_text)


def summarize(rows, k, llm, query="subgoal", node_text=None):
    if not rows:
        raise ValueError("no result rows to summarize")
    n = len(rows)
    rankers = [name for name in ALLOWED_RANKERS if all(name in r for r in rows)]
    rec = lambda name, kk: sum(r[name]["gold_rank"] < kk for r in rows) / n
    print(f"\nquery={query}")
    print(f"node_text={node_text or rows[0].get('node_text', 'own')}")
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
        haiku = rows[0].get("picker") == "haiku"
        timing = "API s" if haiku else "ttft s"
        if haiku:
            print("Haiku timing: total API duration, not TTFT")
        print(f"{'pick arm':12}{'acc':>8}{'acc|shown':>11}{'halluc':>9}{'tok med':>10}{timing:>9}{'CLICK':>9}{'TYPE':>9}{'SELECT':>9}")
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
        if "baseline" in pick_arms:
            print(f"baseline truncated: {sum(r['baseline']['truncated'] for r in rows)/n:.1%}")

    # D7 always uses recall@10 and the top-10 MaxSim picker.
    if "maxsim" not in rankers or "single" not in rankers or (llm and
            not all("maxsim@10" in r and "baseline" in r for r in rows)):
        verdict = "INCOMPLETE"
    else:
        kill = rec("maxsim", 10) <= rec("single", 10)
        passed = rec("maxsim", 10) >= 0.85
        if llm:
            acc = lambda name: sum(r[name]["ok"] for r in rows) / n
            tok = lambda name: st.median(r[name]["tokens"] for r in rows)
            kill |= acc("maxsim@10") < acc("baseline") - 0.05
            passed &= acc("maxsim@10") >= acc("baseline") - 0.02 and tok("maxsim@10") <= 0.2 * tok("baseline")
        verdict = "KILL" if kill else "PASS" if passed and llm else "INCOMPLETE" if not llm else "NEITHER"
    print("GATE:", verdict, "(recall-only)" if not llm else "")


if __name__ == "__main__":
    main()

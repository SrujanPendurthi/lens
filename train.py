"""Prepare shared train triplets and fit the two D15 rankers."""
import argparse
import json
from pathlib import Path
import random

from eval import has_gold
import lens.pick


def triplets(steps: list[dict], seed: int = 0) -> tuple[list[dict], dict]:
    rows = []
    skips = {"textless_gold": 0, "no_negative": 0, "no_gold": 0}
    for index, step in enumerate(steps):
        if not has_gold(step):
            skips["no_gold"] += 1
            continue
        gold = set(step["gold"])
        positive = next((n["text"] for n in step["nodes"] if n["ref"] in gold and n["text"].strip()), None)
        if positive is None:
            skips["textless_gold"] += 1
            continue
        negatives = [n["text"] for n in step["nodes"] if n["ref"] not in gold and n["text"].strip()]
        if not negatives:
            skips["no_negative"] += 1
            continue
        task, prev = step["task"], step["prev"]
        negative = random.Random(f"{seed}\n{task}\n{len(prev)}").choice(negatives)
        goal = lens.pick.subgoal(task, prev)
        rows.append({"query": goal["text"], "positive": positive, "negative": negative})
        if len(rows) % 100 == 0:
            print(f"triplets: {len(rows)}; steps: {index + 1}/{len(steps)}", flush=True)
    return rows, skips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets", nargs=2, metavar=("STEPS", "OUT"))
    args = parser.parse_args()
    if args.triplets:
        source, destination = args.triplets
        with open(source) as stream:
            rows, skips = triplets([json.loads(line) for line in stream])
        with open(destination, "w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        print(f"triplet count: {len(rows)}; skips: {json.dumps(skips, sort_keys=True)}", flush=True)
    else:
        parser.error("specify --triplets STEPS OUT")


if __name__ == "__main__":
    main()

"""Prepare shared train triplets and fit the two D15 rankers."""
import argparse
import json
from pathlib import Path
import random

from eval import has_gold
import lens.pick

TRIPLETS = Path("data/triplets_train.jsonl")
MODELS = Path("models")


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


def fit(name):
    from datasets import Dataset
    from transformers import set_seed
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)

    output = MODELS / f"{name}_ft"
    if output.exists():
        raise RuntimeError(f"refusing a second training run: {output} already exists")
    set_seed(0)
    with TRIPLETS.open() as stream:
        dataset = Dataset.from_list([json.loads(line) for line in stream])
    kwargs = {}
    if name == "single":
        model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        model.max_seq_length = 128
        loss = losses.MultipleNegativesRankingLoss(model)
    elif name == "maxsim":
        from pylate import models, losses as colbert_losses, utils
        model = models.ColBERT("answerdotai/answerai-colbert-small-v1", device="cpu",
                              query_length=32, document_length=64)
        loss = colbert_losses.Contrastive(model)
        kwargs["data_collator"] = utils.ColBERTCollator(model.tokenize)
    else:
        raise ValueError(f"unknown training model: {name}")
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output), seed=0, data_seed=0, num_train_epochs=1,
        learning_rate=2e-5, per_device_train_batch_size=32, warmup_ratio=0.1,
        use_cpu=True, save_strategy="no", logging_strategy="epoch", report_to="none")
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset,
                                         loss=loss, **kwargs)
    result = trainer.train()
    model.save_pretrained(str(output))
    print(f"final train loss ({name}): {result.metrics['train_loss']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--triplets", nargs=2, metavar=("STEPS", "OUT"))
    mode.add_argument("--fit", choices=("single", "maxsim"))
    args = parser.parse_args()
    if args.triplets:
        source, destination = args.triplets
        with open(source) as stream:
            rows, skips = triplets([json.loads(line) for line in stream])
        with open(destination, "w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        print(f"triplet count: {len(rows)}; skips: {json.dumps(skips, sort_keys=True)}", flush=True)
    elif args.fit:
        fit(args.fit)


if __name__ == "__main__":
    main()

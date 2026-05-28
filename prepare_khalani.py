"""Build en->khalani jsonl + 5-fold split from khalani_dataset.json (55 translated pairs)."""
import json, argparse, random, os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/khalani_dataset.json")
    p.add_argument("--out_dir", default="data/khalani")
    p.add_argument("--k", type=int, default=5, help="number of CV folds")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    d = json.load(open(args.src))
    pairs = [x for x in d if x.get("has_translation") and x.get("english")]
    rows = [{"id": x["id"], "section": x.get("section"),
             "instruction": f"Translate to Khalani: {x['english'].strip()}",
             "output": x["khalani"].strip(),
             "en": x["english"].strip(), "khalani": x["khalani"].strip(),
             "sources": x.get("sources", [])} for x in pairs]

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["fold"] = i % args.k

    os.makedirs(args.out_dir, exist_ok=True)
    all_path = os.path.join(args.out_dir, "all.jsonl")
    with open(all_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for fold in range(args.k):
        test = [r for r in rows if r["fold"] == fold]
        train = [r for r in rows if r["fold"] != fold]
        with open(os.path.join(args.out_dir, f"fold{fold}_train.jsonl"), "w") as f:
            for r in train:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(args.out_dir, f"fold{fold}_test.jsonl"), "w") as f:
            for r in test:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(rows)} pairs -> {args.out_dir}  ({args.k}-fold)")
    for fold in range(args.k):
        n = sum(1 for r in rows if r["fold"] == fold)
        print(f"  fold{fold}: test={n} train={len(rows)-n}")


if __name__ == "__main__":
    main()

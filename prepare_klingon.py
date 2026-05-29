"""Build en->klingon jsonl from OPUS Tatoeba en-tlh (CC-BY).

Produces (in data/klingon/):
  all.jsonl, train.jsonl, val.jsonl, test.jsonl   (full split, for fine-tuning)
  fold{0-4}_{train,test}.jsonl                     (ICL subset, mirrors khalani 5-fold)

Source: https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-tlh.txt.zip
"""
import json, argparse, random, os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--en", default="data/klingon_raw/Tatoeba.en-tlh.en")
    p.add_argument("--tlh", default="data/klingon_raw/Tatoeba.en-tlh.tlh")
    p.add_argument("--out_dir", default="data/klingon")
    p.add_argument("--n_test", type=int, default=500)
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument("--icl_n", type=int, default=110, help="size of ICL CV subset")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    en = [l.rstrip("\n").strip() for l in open(args.en)]
    tlh = [l.rstrip("\n").strip() for l in open(args.tlh)]
    assert len(en) == len(tlh)

    # dedup exact (en, tlh) pairs; keep first
    seen, rows = set(), []
    for e, k in zip(en, tlh):
        if not e or not k:
            continue
        key = (e, k)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"instruction": f"Translate to Klingon: {e}", "output": k,
                     "en": e, "klingon": k})

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    def dump(name, data):
        with open(os.path.join(args.out_dir, name), "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("all.jsonl", rows)
    test, val, train = rows[:args.n_test], rows[args.n_test:args.n_test+args.n_val], rows[args.n_test+args.n_val:]
    dump("test.jsonl", test); dump("val.jsonl", val); dump("train.jsonl", train)

    # ICL CV subset (mirror khalani): k-fold over first icl_n rows
    icl = rows[:args.icl_n]
    for i, r in enumerate(icl):
        r = dict(r); r["fold"] = i % args.k
        icl[i] = r
    for fold in range(args.k):
        dump(f"fold{fold}_train.jsonl", [r for r in icl if r["fold"] != fold])
        dump(f"fold{fold}_test.jsonl", [r for r in icl if r["fold"] == fold])

    print(f"{len(rows)} unique pairs (from {len(en)} lines)")
    print(f"  full: train={len(train)} val={len(val)} test={len(test)}")
    print(f"  ICL {args.k}-fold over {len(icl)}: ~{len(icl)//args.k} test/fold")


if __name__ == "__main__":
    main()

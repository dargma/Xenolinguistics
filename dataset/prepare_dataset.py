"""Prepare OPUS-100 en-fi data splits: train/val/test at 1k, 10k, 100k.

Generates dataset/data/{train,val,test}_{1k,10k,100k}.jsonl with format:
  {"instruction": "Translate to Finnish: <EN>", "output": "<FI>",
   "en": "<EN>", "fi": "<FI>"}

For other language pairs, change `LANG_PAIR` and `LANG_NAME` below.
"""
import json, os, argparse
from datasets import load_dataset

LANG_PAIR = "en-fi"     # OPUS-100 config name
SRC, TGT = "en", "fi"
LANG_NAME = "Finnish"   # used in instruction


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="1k,10k,100k",
                   help="comma-separated. tag → N: 1k=1000, 10k=10000, 100k=100000")
    p.add_argument("--out_dir", default="dataset/data")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    size_map = {"1k": 1000, "10k": 10000, "100k": 100000}
    ds = load_dataset("Helsinki-NLP/opus-100", LANG_PAIR, split="train")
    print(f"full train rows: {len(ds)}")

    for tag in args.sizes.split(","):
        N = size_map[tag]
        subset = ds.select(range(N))
        splits = {
            "train": subset.select(range(int(N*0.8))),
            "val":   subset.select(range(int(N*0.8), int(N*0.9))),
            "test":  subset.select(range(int(N*0.9), N)),
        }
        for split_name, data in splits.items():
            path = os.path.join(args.out_dir, f"{split_name}_{tag}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for ex in data:
                    f.write(json.dumps({
                        "instruction": f"Translate to {LANG_NAME}: {ex['translation'][SRC]}",
                        "output":      ex['translation'][TGT],
                        "en":          ex['translation'][SRC],
                        "fi":          ex['translation'][TGT],
                    }, ensure_ascii=False) + "\n")
            print(f"  {path}: {len(data)} pairs")
    print("done.")


if __name__ == "__main__":
    main()

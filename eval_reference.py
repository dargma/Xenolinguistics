"""Eval Helsinki-NLP/opus-mt-tc-big-en-fi (dedicated NMT baseline)."""
import os, json, argparse, torch, sacrebleu
from transformers import MarianMTModel, MarianTokenizer

MODEL_ID = "Helsinki-NLP/opus-mt-tc-big-en-fi"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_file", default="data/test_1k.jsonl")
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--out", default="outputs/reference/eval_results.json")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    tok = MarianTokenizer.from_pretrained(MODEL_ID)
    model = MarianMTModel.from_pretrained(MODEL_ID).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    rows = [json.loads(l) for l in open(args.test_file)][:args.n_eval]
    preds, refs, examples = [], [], []
    for i, ex in enumerate(rows):
        inputs = tok([ex["en"]], return_tensors="pt", padding=True,
                     truncation=True, max_length=256)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, num_beams=args.num_beams)
        pred = tok.decode(out[0], skip_special_tokens=True).strip()
        preds.append(pred); refs.append(ex["fi"])
        if i < 10:
            examples.append({"en": ex["en"], "ref": ex["fi"], "pred": pred})

    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    result = {"model": MODEL_ID, "role": "reference_nmt", "n": len(preds),
              "chrF": round(chrf,2), "BLEU": round(bleu,2), "examples": examples}
    json.dump(result, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"chrF={chrf:.2f}  BLEU={bleu:.2f}  → {args.out}")


if __name__ == "__main__":
    main()

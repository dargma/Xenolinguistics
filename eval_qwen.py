"""Eval Qwen2.5-7B + LoRA on en→fi test set (greedy)."""
import os, json, argparse, torch, sacrebleu
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_ID = "Qwen/Qwen2.5-7B-Instruct"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="outputs/qwen_100k/adapter")
    p.add_argument("--base", default=BASE_ID)
    p.add_argument("--test_file", default="data/test_1k.jsonl")
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out_path = args.out or os.path.join(os.path.dirname(args.adapter) or ".", "eval_results.json")

    is_lora = os.path.exists(os.path.join(args.adapter, "adapter_config.json"))
    tok_src = args.base if is_lora and not os.path.exists(os.path.join(args.adapter, "tokenizer_config.json")) else args.adapter
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if is_lora:
        base = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, args.adapter).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.adapter, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()

    rows = [json.loads(l) for l in open(args.test_file)][:args.n_eval]
    preds, refs, examples = [], [], []
    for i, ex in enumerate(rows):
        prompt = (f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        pred = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
        preds.append(pred); refs.append(ex["fi"])
        if i < 5:
            print(f"[{i}] EN: {ex['en'][:80]}\n    REF: {ex['fi'][:80]}\n    PRED: {pred[:80]}")
        if i < 10:
            examples.append({"en": ex["en"], "ref": ex["fi"], "pred": pred})

    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    result = {"adapter": args.adapter, "base": args.base, "n": len(preds),
              "chrF": round(chrf,2), "BLEU": round(bleu,2), "examples": examples}
    json.dump(result, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nchrF={chrf:.2f}  BLEU={bleu:.2f}  → {out_path}")


if __name__ == "__main__":
    main()

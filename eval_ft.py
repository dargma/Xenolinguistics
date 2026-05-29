"""Eval a FINE-TUNED model (AR or diffusion) on a held-out test set, 0-shot (instruction only).

Fairness (per methodology audit): both models generate FREE-length (no gt_length oracle).
Metrics: chrF, BLEU, exact-match, and word-order tau (vocabulary-decoupled word-order signal).
Reuses helpers from eval_khalani_icl.py.
"""
import os, json, argparse, types, torch, sacrebleu
from eval_khalani_icl import clean_pred, word_order_tau, QWEN_ID, FASTDLLM_ID


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["ar", "diffusion"], required=True)
    p.add_argument("--model_path", required=True, help="fine-tuned model dir (full-FT 'final/') or base id")
    p.add_argument("--test_file", default="/content/local_fast/data/klingon/test.jsonl")
    p.add_argument("--tgt_field", default="klingon")
    p.add_argument("--lang_name", default="Klingon")
    p.add_argument("--n_eval", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--mode", default="free", choices=["free", "gt_length"],
                   help="free = symmetric free-length (cut at EOS); "
                        "gt_length = cap generated tokens to reference token length (oracle).")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_path = args.out or f"outputs/{args.tgt_field}_ft_eval_{args.model_type}_{args.mode}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.model_type == "ar":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()
    else:
        import fastdllm_generation as gf
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
        model.mdm_sample = types.MethodType(gf.Fast_dLLM_QwenForCausalLM.batch_sample, model)
        mask_id = tok.encode("|<MASK>|", add_special_tokens=False)[0]

    rows = [json.loads(l) for l in open(args.test_file)][:args.n_eval]
    preds, refs, examples = [], [], []
    for i, ex in enumerate(rows):
        msgs = [{"role": "user", "content": ex["instruction"]}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        # gt_length oracle: cap generated tokens to reference token length (same budget either way).
        gt_tok = len(tok.encode(ex[args.tgt_field], add_special_tokens=False)) if args.mode == "gt_length" else None
        if args.model_type == "ar":
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            if gt_tok is not None:
                gen_ids = gen_ids[:gt_tok]
            raw = tok.decode(gen_ids, skip_special_tokens=True)
        else:
            input_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            seq_len = torch.tensor([input_ids.shape[1]], device=model.device)
            bs = args.block_size
            mnt = max(((args.max_new_tokens + bs - 1) // bs) * bs, bs)  # same denoise budget for both modes
            out = model.mdm_sample(input_ids, tokenizer=tok, block_size=bs, small_block_size=bs,
                                   max_new_tokens=mnt, mask_id=mask_id, min_len=input_ids.shape[1],
                                   seq_len=seq_len, use_block_cache=True, threshold=args.threshold)
            gen_ids = out[0][input_ids.shape[1]:]
            if gt_tok is not None:
                gen_ids = gen_ids[:gt_tok]
            raw = tok.decode(gen_ids, skip_special_tokens=True)
        pred = clean_pred(raw, args.lang_name)
        preds.append(pred); refs.append(ex[args.tgt_field])
        if i < 10:
            examples.append({"en": ex.get("en"), "ref": ex[args.tgt_field], "pred": pred})

    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    em = 100.0 * sum(p == r for p, r in zip(preds, refs)) / len(preds)
    tau, tau_n = word_order_tau(preds, refs)
    result = {"model_type": args.model_type, "model_path": args.model_path,
              "test_file": args.test_file, "n": len(preds), "mode": args.mode,
              "max_new_tokens": args.max_new_tokens,
              "chrF": round(chrf, 2), "BLEU": round(bleu, 2), "exact_match": round(em, 2),
              "word_order_tau": round(tau, 3) if tau is not None else None, "word_order_tau_n": tau_n,
              "examples": examples}
    json.dump(result, open(out_path, "w"), indent=2, ensure_ascii=False)
    taustr = f"{tau:.3f}(n={tau_n})" if tau is not None else "n/a"
    print(f"\n[{args.model_type}] chrF={chrf:.2f} BLEU={bleu:.2f} EM={em:.1f} word_order_tau={taustr}  n={len(preds)} -> {out_path}")


if __name__ == "__main__":
    main()

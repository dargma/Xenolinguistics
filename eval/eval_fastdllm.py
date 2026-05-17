"""Eval Fast-dLLM v2 7B on en→fi test set using official mdm_sample."""
import os, json, argparse, types, torch, sacrebleu
from transformers import AutoTokenizer, AutoModelForCausalLM
import fastdllm_generation as gf

MODEL_ID = "Efficient-Large-Model/Fast_dLLM_v2_7B"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=MODEL_ID)
    p.add_argument("--test_file", default="dataset/data/test_1k.jsonl")
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--small_block_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--mode", default="free", choices=["free", "gt_length"],
                   help="free = first-EOS cut, max_new_tokens=256 cap (no length info). "
                        "gt_length = generate ceil(len_tok(fi_ref)/block_size)*block_size masks "
                        "and truncate to exactly len_tok(fi_ref) tokens (oracle length, EOS ignored).")
    p.add_argument("--out", default="eval_fastdllm.json")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    mask_id = tok.encode("|<MASK>|", add_special_tokens=False)[0]
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()
    model.mdm_sample = types.MethodType(gf.Fast_dLLM_QwenForCausalLM.batch_sample, model)

    rows = [json.loads(l) for l in open(args.test_file)][:args.n_eval]
    preds, refs, examples = [], [], []
    for i, ex in enumerate(rows):
        msgs = [{"role": "user", "content": ex["instruction"]}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        seq_len = torch.tensor([input_ids.shape[1]], device=model.device)
        if args.mode == "gt_length":
            gt_tok = len(tok.encode(ex["fi"], add_special_tokens=False))
            mnt = ((gt_tok + args.block_size - 1) // args.block_size) * args.block_size
            mnt = max(mnt, args.block_size)
        else:
            mnt = args.max_new_tokens
        out = model.mdm_sample(
            input_ids, tokenizer=tok,
            block_size=args.block_size, small_block_size=args.small_block_size,
            max_new_tokens=mnt, mask_id=mask_id,
            min_len=input_ids.shape[1], seq_len=seq_len,
            use_block_cache=True, threshold=args.threshold,
        )
        gen_ids = out[0][input_ids.shape[1]:]
        if args.mode == "gt_length":
            # Oracle cut at exact gt token length, ignore EOS
            gen_ids = gen_ids[:gt_tok]
        else:
            # free: cut at first EOS, fall back to mnt cap
            eos_id = tok.eos_token_id
            if eos_id is not None:
                eos_pos = (gen_ids == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    gen_ids = gen_ids[:eos_pos[0]]
        pred = tok.decode(gen_ids, skip_special_tokens=True).strip()
        preds.append(pred); refs.append(ex["fi"])
        if i < 5:
            print(f"[{i}] EN: {ex['en'][:80]}\n    REF: {ex['fi'][:80]}\n    PRED: {pred[:80]}")
        if i < 10:
            examples.append({"en": ex["en"], "ref": ex["fi"], "pred": pred})

    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    result = {"ckpt": args.ckpt, "mode": args.mode, "n": len(preds),
              "chrF": round(chrf,2), "BLEU": round(bleu,2),
              "block_size": args.block_size, "small_block_size": args.small_block_size,
              "threshold": args.threshold, "max_new_tokens_cap": args.max_new_tokens,
              "examples": examples}
    json.dump(result, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nchrF={chrf:.2f}  BLEU={bleu:.2f}  → {args.out}")


if __name__ == "__main__":
    main()

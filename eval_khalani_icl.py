"""Few-shot ICL eval (no training) for en->khalani: AR (Qwen) vs Diffusion (Fast-dLLM v2).

Headline axis of the conlang experiment. For each CV fold: draw k few-shot examples from
that fold's train split (same examples for both models), prompt with the held-out English,
generate Khalani. Convention (per CLAUDE.md): Qwen = greedy free-length; Fast-dLLM = gt_length.
"""
import os, json, argparse, types, random, torch, sacrebleu

# Compat: transformers 5.x dropped "default" rope_type key (needed by Fast-dLLM v2)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
def _default_rope(config, device=None, seq_len=None, **kwargs):
    base = config.rope_theta
    dim = int(config.head_dim if getattr(config, "head_dim", None) else config.hidden_size // config.num_attention_heads)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, 1.0
ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope)

QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
FASTDLLM_ID = "Efficient-Large-Model/Fast_dLLM_v2_7B"

def word_order_tau(preds, refs):
    """Word-order signal decoupled from vocabulary: for each pair, take words the
    prediction got right (present in ref), and measure whether their relative ORDER
    matches the reference via Kendall tau. Mean over pairs with >=2 matched words.
    Returns (mean_tau, n_scored). tau in [-1,1]; 1 = same order, -1 = reversed."""
    taus = []
    for pred, ref in zip(preds, refs):
        rw = ref.split()
        pw = pred.split()
        ref_pos = {}
        for i, w in enumerate(rw):
            ref_pos.setdefault(w, i)  # first occurrence
        matched = [ref_pos[w] for w in pw if w in ref_pos]
        # dedup preserving pred order, keep words matched once
        seen, seq = set(), []
        for w in pw:
            if w in ref_pos and w not in seen:
                seen.add(w); seq.append(ref_pos[w])
        if len(seq) < 2:
            continue
        conc = disc = 0
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                if seq[i] < seq[j]:
                    conc += 1
                elif seq[i] > seq[j]:
                    disc += 1
        tot = conc + disc
        if tot:
            taus.append((conc - disc) / tot)
    if not taus:
        return None, 0
    return sum(taus) / len(taus), len(taus)


def clean_pred(pred, lang_name):
    if not pred:
        return pred
    label = lang_name.lower() + ":"
    pred = pred.strip().strip("`").strip()
    for line in pred.splitlines():
        line = line.strip()
        low = line.lower()
        if low.startswith(label):
            line = line[len(label):].strip()
        if low.startswith("english:"):
            continue
        if line:
            return line.strip().strip('"').strip("`").strip()
    return ""


def build_messages(shots, test_en, instruction, lang_name, tgt_field):
    """Multi-turn chat: few-shot pairs as user/assistant turns so the assistant
    turn is pure target language (no trailing label to echo). Same for both models."""
    msgs = []
    for i, ex in enumerate(shots):
        user = (f"{instruction}\n\nEnglish: {ex['en']}" if i == 0 else f"English: {ex['en']}")
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": ex[tgt_field]})
    msgs.append({"role": "user", "content": f"English: {test_en}"})
    return msgs


def run_fold(model, tok, model_type, train, test, k, seed, gen_kwargs, cfg):
    instruction, lang_name, tgt_field = cfg["instruction"], cfg["lang_name"], cfg["tgt_field"]
    rng = random.Random(seed)
    shots = rng.sample(train, min(k, len(train)))
    preds, refs, examples = [], [], []
    for ex in test:
        msgs = build_messages(shots, ex["en"], instruction, lang_name, tgt_field)
        if model_type == "ar":
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=gen_kwargs["max_new_tokens"],
                                     do_sample=False, pad_token_id=tok.eos_token_id)
            pred = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        else:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            input_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            seq_len = torch.tensor([input_ids.shape[1]], device=model.device)
            bs = gen_kwargs["block_size"]
            if gen_kwargs["diff_mode"] == "gt_length":
                budget = len(tok.encode(ex[tgt_field], add_special_tokens=False))  # oracle (asymmetric!)
            else:
                budget = gen_kwargs["max_new_tokens"]  # free, symmetric with AR
            mnt = max(((budget + bs - 1) // bs) * bs, bs)
            out = model.mdm_sample(
                input_ids, tokenizer=tok, block_size=bs, small_block_size=bs,
                max_new_tokens=mnt, mask_id=gen_kwargs["mask_id"],
                min_len=input_ids.shape[1], seq_len=seq_len,
                use_block_cache=True, threshold=gen_kwargs["threshold"])
            pred = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred = clean_pred(pred, lang_name)
        preds.append(pred); refs.append(ex[tgt_field])
        examples.append({"en": ex["en"], "ref": ex[tgt_field], "pred": pred})
    return preds, refs, examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["ar", "diffusion"], required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--data_dir", default="data/khalani")
    p.add_argument("--tgt_field", default="khalani", help="jsonl key for the target text")
    p.add_argument("--lang_name", default="Khalani")
    p.add_argument("--lang_desc", default="the language of the Protoss")
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--n_shot", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--diff_mode", default="free", choices=["free", "gt_length"],
                   help="free (default) = symmetric with AR free-length. gt_length = oracle target length (ASYMMETRIC, supplementary only).")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ckpt = args.ckpt or (QWEN_ID if args.model_type == "ar" else FASTDLLM_ID)
    out_path = args.out or f"outputs/{args.tgt_field}_icl_{args.model_type}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cfg = {"tgt_field": args.tgt_field, "lang_name": args.lang_name,
           "instruction": (f"Translate the following English into {args.lang_name}, {args.lang_desc}. "
                           f"Output only the {args.lang_name} translation, nothing else.")}

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    gen_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.model_type == "ar":
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="auto").eval()
    else:
        import fastdllm_generation as gf
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
        model.mdm_sample = types.MethodType(gf.Fast_dLLM_QwenForCausalLM.batch_sample, model)
        gen_kwargs.update({"block_size": args.block_size, "threshold": args.threshold,
                           "diff_mode": args.diff_mode,
                           "mask_id": tok.encode("|<MASK>|", add_special_tokens=False)[0]})

    all_preds, all_refs, fold_scores, fold_examples = [], [], [], []
    for fold in range(args.k_folds):
        train = [json.loads(l) for l in open(f"{args.data_dir}/fold{fold}_train.jsonl")]
        test = [json.loads(l) for l in open(f"{args.data_dir}/fold{fold}_test.jsonl")]
        preds, refs, examples = run_fold(model, tok, args.model_type, train, test,
                                         args.n_shot, args.seed + fold, gen_kwargs, cfg)
        chrf = sacrebleu.corpus_chrf(preds, [refs]).score
        em = 100.0 * sum(p == r for p, r in zip(preds, refs)) / len(preds)
        fold_scores.append({"fold": fold, "chrF": round(chrf, 2),
                            "exact_match": round(em, 2), "n": len(preds)})
        print(f"[{args.model_type}] fold{fold}: chrF={chrf:.2f} EM={em:.1f} n={len(preds)}")
        all_preds += preds; all_refs += refs
        fold_examples.append({"fold": fold, "examples": examples[:5]})

    import statistics
    chrfs = [s["chrF"] for s in fold_scores]
    micro_chrf = sacrebleu.corpus_chrf(all_preds, [all_refs]).score
    micro_bleu = sacrebleu.corpus_bleu(all_preds, [all_refs]).score
    micro_em = 100.0 * sum(p == r for p, r in zip(all_preds, all_refs)) / len(all_preds)
    tau, tau_n = word_order_tau(all_preds, all_refs)
    result = {"model_type": args.model_type, "ckpt": ckpt, "n_shot": args.n_shot,
              "k_folds": args.k_folds, "n_total": len(all_preds),
              "diff_mode": args.diff_mode if args.model_type == "diffusion" else None,
              "chrF_macro_mean": round(statistics.mean(chrfs), 2),
              "chrF_macro_std": round(statistics.pstdev(chrfs), 2),
              "chrF_micro": round(micro_chrf, 2), "BLEU_micro": round(micro_bleu, 2),
              "exact_match_micro": round(micro_em, 2),
              "word_order_tau": round(tau, 3) if tau is not None else None,
              "word_order_tau_n": tau_n,
              "fold_scores": fold_scores, "fold_examples": fold_examples}
    json.dump(result, open(out_path, "w"), indent=2, ensure_ascii=False)
    taustr = f"{tau:.3f}(n={tau_n})" if tau is not None else "n/a"
    print(f"\n[{args.model_type}] chrF macro={result['chrF_macro_mean']}±{result['chrF_macro_std']} "
          f"micro={micro_chrf:.2f}  BLEU={micro_bleu:.2f}  EM={micro_em:.1f}  word_order_tau={taustr}  -> {out_path}")


if __name__ == "__main__":
    main()

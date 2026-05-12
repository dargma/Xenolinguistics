"""
Dream-7B MBR (Minimum Bayes Risk) Decoding
Generate N candidates per input, select best by chrF against others.
No retraining needed.
"""

import json
import torch
import sacrebleu
import argparse
import time
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel


def mbr_select(candidates, metric="chrf"):
    """Select candidate with highest average pairwise chrF against all others."""
    n = len(candidates)
    if n == 1:
        return 0, candidates[0]

    scores = []
    for i, cand in enumerate(candidates):
        pairwise = []
        for j, other in enumerate(candidates):
            if i == j:
                continue
            if metric == "chrf":
                s = sacrebleu.sentence_chrf(cand, [other]).score
            else:
                s = sacrebleu.sentence_bleu(cand, [other]).score
            pairwise.append(s)
        scores.append(sum(pairwise) / len(pairwise))

    best_idx = max(range(n), key=lambda i: scores[i])
    return best_idx, candidates[best_idx]


def evaluate(args):
    test_data = [json.loads(l) for l in open(args.test_file)][:args.n_eval]

    print(f"MBR Decoding | N={args.n_candidates} | temp={args.temperature}")
    print(f"Adapter: {args.adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    base = AutoModel.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    MASK_ID = tokenizer.mask_token_id
    print(f"Evaluating {len(test_data)} samples...\n")

    predictions = []
    references = []
    examples = []
    greedy_predictions = []  # for comparison
    start = time.time()

    for i, ex in enumerate(test_data):
        messages = [{"role": "user", "content": ex["instruction"]}]
        prompt_str = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompt_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        input_ids = prompt_ids["input_ids"].to(model.device)
        attention_mask = prompt_ids["attention_mask"].to(model.device)
        prompt_len = input_ids.shape[1]

        candidates = []
        for c in range(args.n_candidates):
            with torch.no_grad():
                temp = 0.0001 if c == 0 else args.temperature  # first is greedy
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    output_history=False,
                    steps=args.steps,
                    temperature=temp,
                    top_p=args.top_p,
                    alg=args.alg,
                    alg_temp=0.0,
                )
                gen_ids = output.sequences[0] if hasattr(output, "sequences") else output[0]

            gen_text = tokenizer.decode(gen_ids[prompt_len:], skip_special_tokens=True).strip()
            for stop in ["<|im_end|>", "<|endoftext|>", "\n\n\n"]:
                if stop in gen_text:
                    gen_text = gen_text[:gen_text.index(stop)].strip()
            candidates.append(gen_text)

        # MBR selection
        best_idx, best_text = mbr_select(candidates)
        predictions.append(best_text)
        greedy_predictions.append(candidates[0])
        references.append(ex["fi"])

        if i < 5:
            print(f"[{i}] EN:   {ex['en']}")
            print(f"    REF:  {ex['fi']}")
            print(f"    GREEDY: {candidates[0]}")
            print(f"    MBR({best_idx}): {best_text}")
            if len(set(candidates)) > 1:
                print(f"    (unique candidates: {len(set(candidates))}/{len(candidates)})")
            print()

    elapsed = time.time() - start

    # MBR metrics
    chrf_mbr = sacrebleu.corpus_chrf(predictions, [references])
    bleu_mbr = sacrebleu.corpus_bleu(predictions, [references])
    # Greedy metrics (for comparison)
    chrf_greedy = sacrebleu.corpus_chrf(greedy_predictions, [references])
    bleu_greedy = sacrebleu.corpus_bleu(greedy_predictions, [references])

    print(f"\n{'='*50}")
    print(f"Greedy:  chrF={chrf_greedy.score:.2f}, BLEU={bleu_greedy.score:.2f}")
    print(f"MBR(N={args.n_candidates}): chrF={chrf_mbr.score:.2f}, BLEU={bleu_mbr.score:.2f}")
    print(f"Delta:   chrF={chrf_mbr.score - chrf_greedy.score:+.2f}, BLEU={bleu_mbr.score - bleu_greedy.score:+.2f}")
    print(f"Time: {elapsed:.0f}s ({elapsed/len(test_data):.1f}s/sample)")
    print(f"{'='*50}")

    result = {
        "method": "MBR",
        "n_candidates": args.n_candidates,
        "temperature": args.temperature,
        "greedy_chrF": round(chrf_greedy.score, 2),
        "greedy_BLEU": round(bleu_greedy.score, 2),
        "mbr_chrF": round(chrf_mbr.score, 2),
        "mbr_BLEU": round(bleu_mbr.score, 2),
        "delta_chrF": round(chrf_mbr.score - chrf_greedy.score, 2),
    }
    out_path = f"{args.adapter_path}/../mbr_results.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--test_file", default="data/test_1k.jsonl")
    parser.add_argument("--n_eval", type=int, default=25)
    parser.add_argument("--n_candidates", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--alg", default="entropy")
    args = parser.parse_args()
    evaluate(args)

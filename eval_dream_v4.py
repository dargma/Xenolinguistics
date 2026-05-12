"""
Dream-7B v4 — Unified Evaluation
Evaluates any Dream adapter using diffusion_generate
"""

import json
import torch
import sacrebleu
import argparse
import time
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel


def evaluate(args):
    test_data = [json.loads(l) for l in open(args.test_file)][:args.n_eval]

    print(f"Loading model: {args.model_id}")
    print(f"Adapter: {args.adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    base = AutoModel.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    MASK_ID = tokenizer.mask_token_id
    print(f"mask_token_id: {MASK_ID}")
    print(f"Evaluating {len(test_data)} samples (steps={args.steps}, temp={args.temperature})...\n")

    predictions = []
    references = []
    examples = []
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

        with torch.no_grad():
            try:
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    output_history=False,
                    steps=args.steps,
                    temperature=args.temperature if args.temperature > 0 else 0.0001,
                    top_p=args.top_p,
                    alg=args.alg,
                    alg_temp=0.0,
                )
                gen_ids = output.sequences[0] if hasattr(output, "sequences") else output[0]
            except Exception as e:
                if i == 0:
                    print(f"diffusion_generate failed: {e}")
                    print("Falling back to manual denoising...")

                n_masks = args.max_new_tokens
                full_ids = torch.cat([
                    input_ids,
                    torch.full((1, n_masks), MASK_ID, dtype=torch.long, device=model.device)
                ], dim=1)
                full_attn = torch.ones_like(full_ids)
                attn_4d = torch.logical_and(
                    full_attn.unsqueeze(1).unsqueeze(-2),
                    full_attn.unsqueeze(1).unsqueeze(-1),
                )

                for step in range(min(args.steps, 32)):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(
                            input_ids=full_ids,
                            attention_mask=attn_4d,
                            use_cache=False,
                        ).logits

                    shift_logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1)
                    pred_ids = shift_logits.argmax(dim=-1)
                    mask_pos = (full_ids == MASK_ID)
                    full_ids = torch.where(mask_pos, pred_ids, full_ids)

                gen_ids = full_ids[0]

        gen_text = tokenizer.decode(gen_ids[prompt_len:], skip_special_tokens=True).strip()
        for stop in ["<|im_end|>", "<|endoftext|>", "\n\n\n"]:
            if stop in gen_text:
                gen_text = gen_text[:gen_text.index(stop)].strip()

        predictions.append(gen_text)
        references.append(ex["fi"])

        if i < 10:
            examples.append({
                "en": ex["en"],
                "fi_ref": ex["fi"],
                "fi_pred": gen_text,
            })
            if i < 5:
                print(f"[{i}] EN:   {ex['en']}")
                print(f"    REF:  {ex['fi']}")
                print(f"    PRED: {gen_text}")
                print()

    elapsed = time.time() - start
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    print(f"\n{'='*50}")
    print(f"chrF={chrf.score:.2f}, BLEU={bleu.score:.2f} ({elapsed:.0f}s)")
    print(f"{'='*50}")

    result = {
        "model": "Dream-7B + LoRA (v4)",
        "adapter": args.adapter_path,
        "data_tag": args.test_file,
        "n_eval": len(predictions),
        "chrF": round(chrf.score, 2),
        "BLEU": round(bleu.score, 2),
        "steps": args.steps,
        "temperature": args.temperature,
        "alg": args.alg,
        "max_new_tokens": args.max_new_tokens,
        "eval_time_sec": round(elapsed, 1),
        "examples": examples,
    }
    out_path = f"{args.adapter_path}/../eval_results.json"
    json.dump(result, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"Results saved -> {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dream-7B v4 Evaluation")
    parser.add_argument("--model_id", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--test_file", default="data/test_1k.jsonl")
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--alg", default="entropy")
    args = parser.parse_args()
    evaluate(args)

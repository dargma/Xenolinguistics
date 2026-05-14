"""
Dream-7B — DiffusionLLM-style Oracle Length Evaluation

Follows the oracle length protocol from "Diffusion Language Models Can Perform
Many Tasks with Scaling and Instruction-Finetuning" (Ye et al. 2023):

  input = [prompt] + [MASK × gt_len] + [EOS]

The EOS at the end is FIXED (not masked, not predicted) — it acts as an
explicit "stop here" anchor for the diffusion process. Only the gt_len
mask block is denoised. Decoding takes only the gt_len region.

This differs from our previous approach which used Dream's diffusion_generate
with max_new_tokens=gt_len (no explicit terminal EOS).

Reference: github.com/yegcjs/DiffusionLLM src/model/dd_model.py:162-172
"""
import argparse
import json
import os
import time

import sacrebleu
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"


def dream_oracle_generate(model, tokenizer, instruction, gt_len,
                          mask_id, eos_id, steps=DENOISE_STEPS):
    """DiffusionLLM oracle-length generation:
       seq = [prompt(chat-template)] + [MASK × gt_len] + [EOS]
       Only the mask block is denoised; prompt and trailing EOS are fixed.
    """
    messages = [{"role": "user", "content": instruction}]
    prompt_str = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer(
        prompt_str, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    mask_block = torch.full(
        (1, gt_len), mask_id, dtype=torch.long, device=model.device
    )
    eos_tok = torch.tensor([[eos_id]], dtype=torch.long, device=model.device)
    input_ids = torch.cat([prompt_ids, mask_block, eos_tok], dim=1)
    mask_end = prompt_len + gt_len  # exclusive

    # Dream uses bidirectional attention (4D mask)
    attn_1d = torch.ones_like(input_ids)
    attn_4d = torch.logical_and(
        attn_1d.unsqueeze(1).unsqueeze(-2),
        attn_1d.unsqueeze(1).unsqueeze(-1),
    )

    for step in range(steps):
        is_mask = input_ids == mask_id
        n_remaining = is_mask.sum().item()
        if n_remaining == 0:
            break

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attn_4d,
                use_cache=False,
            ).logits

        # Dream logit shift: predict position n from position n-1
        shift_logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1)
        probs = torch.softmax(shift_logits.float(), dim=-1)
        max_probs, pred_ids = probs.max(dim=-1)

        # Number to unmask this step (linear schedule)
        n_to_unmask = max(1, n_remaining // (steps - step))

        # Only consider mask positions; pick highest-confidence ones
        conf = max_probs.clone()
        conf[~is_mask] = -float("inf")
        _, top_idx = conf.view(-1).topk(n_to_unmask)

        flat_in = input_ids.view(-1)
        flat_pred = pred_ids.view(-1)
        flat_in[top_idx] = flat_pred[top_idx]
        input_ids = flat_in.view(input_ids.shape)

    # Decode only the mask-region (oracle GT-length window, EOS excluded)
    response_ids = input_ids[0, prompt_len:mask_end]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Dream DiffusionLLM-oracle eval")
    parser.add_argument("--adapter_path", required=True,
                        help="Path to LoRA adapter (e.g., outputs/dream_qkvo_100k_ep1/adapter_best)")
    parser.add_argument("--test_file", default="data/test_1k.jsonl")
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--steps", type=int, default=64,
                        help="Iterative denoising steps")
    parser.add_argument("--output_path", default=None,
                        help="Where to save eval JSON (default: <adapter_dir>/../eval_diffusionllm_oracle.json)")
    args = parser.parse_args()

    test_data = [json.loads(line) for line in open(args.test_file)][:args.n_eval]

    print(f"Loading Dream: {MODEL_ID}")
    print(f"Adapter: {args.adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    base = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    print(f"mask_token_id={mask_id}, eos_token_id={eos_id}")
    print(f"Evaluating {len(test_data)} samples — DiffusionLLM oracle protocol")
    print(f"  layout: [prompt] + [MASK × gt_len] + [EOS], denoise={args.steps} steps")

    predictions, references, examples = [], [], []
    start = time.time()

    for i, ex in enumerate(test_data):
        gt_len = len(tokenizer.encode(ex["fi"], add_special_tokens=False))

        with torch.no_grad():
            gen_text = dream_oracle_generate(
                model, tokenizer,
                instruction=ex["instruction"],
                gt_len=gt_len,
                mask_id=mask_id,
                eos_id=eos_id,
                steps=args.steps,
            )

        # Defensive post-processing (model may leak chat tokens)
        for stop in ["<|im_end|>", "<|endoftext|>", "\n\n\n"]:
            if stop in gen_text:
                gen_text = gen_text[:gen_text.index(stop)].strip()

        predictions.append(gen_text)
        references.append(ex["fi"])

        if i < 10:
            examples.append({"en": ex["en"], "fi_ref": ex["fi"], "fi_pred": gen_text})
            if i < 5:
                print(f"[{i}] EN:   {ex['en']}")
                print(f"    REF:  {ex['fi']}")
                print(f"    PRED: {gen_text}")
                print()

    elapsed = time.time() - start
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    print(f"\n{'=' * 60}")
    print(f"[Dream DiffusionLLM-oracle] chrF={chrf.score:.2f}  BLEU={bleu.score:.2f}  ({elapsed:.0f}s)")
    print(f"{'=' * 60}")

    result = {
        "model": "Dream-7B + LoRA",
        "adapter": args.adapter_path,
        "mode": "diffusionllm_oracle",
        "protocol": "[prompt] + [MASK x gt_len] + [EOS], fixed prompt+EOS, denoise mask block only",
        "n_eval": len(predictions),
        "denoise_steps": args.steps,
        "chrF": round(chrf.score, 2),
        "BLEU": round(bleu.score, 2),
        "eval_time_sec": round(elapsed, 1),
        "examples": examples,
    }
    if args.output_path:
        out_path = args.output_path
    else:
        adapter_dir = os.path.dirname(args.adapter_path.rstrip("/"))
        out_path = os.path.join(adapter_dir, "eval_diffusionllm_oracle.json")
    json.dump(result, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"Results saved -> {out_path}")


if __name__ == "__main__":
    main()
